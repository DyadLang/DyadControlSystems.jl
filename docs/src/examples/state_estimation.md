# State estimation for ModelingToolkit models

This tutorial will demonstrate how to perform *state estimation* using a model built using ModelingToolkit. State estimation is the process of estimating the state of a system from noisy measurements. For practical systems occurring in engineering and industrial practice, we *almost never* have measurement access to the true state of the system. We may use sensors to measure parts of the state, but sensors are noisy, and it is often impractical to measure all components of the state. Some examples of variables that are difficult to measure are
- The state of charge of a battery
- The temperature of the lubrication in a gearbox
- Velocities and angular velocities of moving parts
- In industrial robots, it is common to only be able to measure the angle of a joint on the motor side. The angle of the joint on the other side of the gearbox is often not accessible. The gearbox may be flexible and have backlash, causing the arm-side joint angle to be very different from the motor-side joint angle.
- The internal temperature of the walls in a building.


When designing a controller that uses a model including state that is not measurable, or the measurements are very noisy, we must make use of a state estimator, commonly referred to as a *state observer* in the field of automatic control. A commonly used state observer is the [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter), this is an *optimal* state observer if the system is linear and noise is Gaussian. However, for nonlinear systems, we may have to use a more sophisticated state observer. In this tutorial, we will design an [Unscented Kalman filter (UKF)](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter) to estimate the position of a 2D mass-spring system. The construction of the system model is detailed in [Component-Based Modeling a Spring-Mass System](https://docs.sciml.ai/ModelingToolkit/stable/examples/spring_mass/#Component-Based-Modeling-a-Spring-Mass-System), we reproduce this code below without any details, and then move on to designing the state observer.

## Building the system model
The following code builds the model of the **2D spring-mass system** and simulates it using OrdinaryDiffEq.jl.
```@example STATE_ESTIMATION
using DyadControlSystems, ModelingToolkit
using ModelingToolkit, Plots, OrdinaryDiffEq, LinearAlgebra
gr(fmt=:png) # hide

@variables t
D = Differential(t)

function Mass(; name, m = 1.0, xy = [0., 0.], u = [0., 0.])
    ps = @parameters m=m
    sts = @variables pos(t)[1:2]=xy v(t)[1:2]=u
    eqs = collect(D.(pos) .~ v)
    ODESystem(eqs, t, [pos..., v...], ps; name)
end

function Spring(; name, k = 1e4, l = 1.)
    ps = @parameters k=k l=l
    @variables x(t), dir(t)[1:2]
    ODESystem(Equation[], t, [x, dir...], ps; name)
end

function connect_spring(spring, a, b)
    [
        spring.x ~ norm(collect(a .- b))
        collect(spring.dir .~ collect(a .- b))
    ]
end

spring_force(spring) = -spring.k .* collect(spring.dir) .* (spring.x - spring.l)  ./ spring.x

m = 1.0
xy = [1., -1.]
k = 1e4
l = 1.
center = [0., 0.]
g = [0., -9.81]
@named mass = Mass(m=m, xy=xy)
@named spring = Spring(k=k, l=l)

eqs = [
    connect_spring(spring, mass.pos, center)
    collect(D.(mass.v) .~ spring_force(spring) / mass.m .+ g)
]

@named _model = ODESystem(eqs, t, [spring.x; spring.dir; mass.pos], [])
@named model = compose(_model, mass, spring)
sys = structural_simplify(model)

prob = ODEProblem(sys, [], (0., 2.))
sol = solve(prob, Rosenbrock23())
plot(sol, layout=4, plot_title="Simulation")
```

## State estimation

To build the state observer, we first construct a [`FunctionSystem`](@ref) from our `ODESystem`. When doing this, we indicate what our inputs and outputs are. This particular system has no inputs, and we are only able to measure the 2D position of the mass, i.e., we are unable to measure the velocities. 
```@example STATE_ESTIMATION
cmodel = complete(model)
inputs = []
outputs = collect(cmodel.mass.pos)
funcsys = FunctionSystem(model, inputs, outputs)
p = ModelingToolkit.varmap_to_vars(ModelingToolkit.defaults(model), funcsys.p)
```

The UKF will operate in discrete time with a sample interval ``T_s = 5``ms, and the system will be discretized using a 4th order Runge-Kutta method. 
```@example STATE_ESTIMATION
Ts = 0.005 # Sample time
discrete_dynamics = DyadControlSystems.rk4(funcsys, Ts, supersample=2)
nothing # hide
```

The state observer comes from the library [LowLevelParticleFilters](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/). This library also contains a function to compute the discrete-time covariance matrices for a double-integrator model, which we will use to pick the dynamics-noise covariance matrix for the system, ``R_1``. We also select a measurement-noise covariance matrix ``R_2`` and an initial state distribution ``d_0``.

The covariance matrices ``R_1`` and ``R_2`` are defined as
```@example STATE_ESTIMATION
using LowLevelParticleFilters
Rdi = LowLevelParticleFilters.double_integrator_covariance(Ts, 1)
R1 = cat(Rdi, Rdi, dims=(1,2)) + 1e-9I
R2 = 0.005I(funcsys.ny)
x0 = sol(0, idxs=funcsys.x) 
d0 = MvNormal(x0, R1)
nothing # hide
```

We use the simulated solution from above to generate some data to use for state estimation. We add measurement noise to the simulated data to make our experiment more realistic.
```@example STATE_ESTIMATION
Tf = sol.t[end]                     # Final time
timevec = 0:Ts:Tf
u = fill([], length(timevec))       # No inputs
y0 = sol(timevec, idxs=outputs).u   # Noise-free output
y = [y0[i] + rand(MvNormal(R2)) for i in 1:length(y0)] # Add measurement noise
nothing # hide
```

We are now ready to construct the the state estimator by calling the constructor [`UnscentedKalmanFilter`](@ref) with our discretized function system. We use the [`forward_trajectory`](@ref) function to run the state estimation for a full trajectory. The function returns a filtering-solution object which contains the state estimates and the log-likelihood of the solution. We plot the state estimates and the simulated solution together with the noisy measurements.
```@example STATE_ESTIMATION
ukf = UnscentedKalmanFilter(discrete_dynamics, R1, R2, d0; p)
filtersol = forward_trajectory(ukf, u, y)
plot(timevec, filtersol, ploty=false, plotx=false, plotu=false)
plot!(sol, idxs=funcsys.x, plot_title="State estimation using UKF")
plot!(timevec, reduce(hcat, y)', sp=[3 4], lab="Measurements", alpha=0.5)
```
If everything went well, we should see that the state estimates ``x(t|t)`` track the true data closely.
```@example STATE_ESTIMATION
using Test
@test filtersol.ll > 500
filtersol.ll
```

## Concluding remarks
This tutorial demonstrated state estimation using a ModelingToolkit model. Even though we simulated some measurement noise, this was a fairly ideal scenario in the sense that we had no *model error*. State estimation can sometimes be difficult in practice due to model mismatch, making tuning the state estimator a challenging task. We also do not have access to the "true state" in practice, further complicating the tuning of the estimator. The tutorial on [Parameter estimation for observers](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/) contains various strategies that may help you solve this tuning problem.

Finally, we note that this example used a nonlinear observer (the UKF) even though the system was actually linear. For a linear system, we could have made use of a standard Kalman filter and obtained a more efficient solution. However, the UKF is a general-purpose state estimator that can be used for both linear and nonlinear systems, and since MTK allows us to model nonlinear systems, we chose to use the UKF in this example.