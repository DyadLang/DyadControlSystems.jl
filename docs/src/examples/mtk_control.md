# Modeling for control using ModelingToolkit
Modeling is an integral part in several control-design workflows, indeed, control is often the primary reason for performing modeling in the first place. This page aims at providing a high-level overview of how we can make good use of the acausal modeling language of ModelingToolkit for control design and analysis purposes. We will look at things like
- Simulating a closed-loop system with a PID controller.
- Building a state-feedback controller in ModelingToolkit.
- Making use of an inverse model for improved reference tracking.
- Perform closed-loop analysis of sensitivity functions.

## PID control with ModelingToolkit

This example will demonstrate a simple workflow for simulation of a control system.

### The system model
We will consider a system consisting of **two masses connected by a flexible element**.
![Double-mass model](figs/double_mass.png)

The code belows defines the system model, which also includes a damper to model dissipation of energy by the flexible element. 

```@example mtk_control
using ModelingToolkit, OrdinaryDiffEq, Plots, LinearAlgebra
using ModelingToolkitStandardLibrary.Mechanical.Rotational
using ModelingToolkitStandardLibrary.Blocks: Sine
using ModelingToolkit: connect
gr(fmt=:png) # hide
import ModelingToolkitStandardLibrary.Blocks
t = Blocks.t

# Parameters
m1 = 1
m2 = 1
k = 1000 # Spring stiffness
c = 10   # Damping coefficient

@named inertia1 = Inertia(; J = m1, phi=0, w=0)
@named inertia2 = Inertia(; J = m2, phi=0, w=0)

@named spring = Spring(; c = k)
@named damper = Damper(; d = c)

@named torque = Torque(use_support=false)

function SystemModel(u=nothing; name=:model)
    eqs = [
        connect(torque.flange, inertia1.flange_a)
        connect(inertia1.flange_b, spring.flange_a, damper.flange_a)
        connect(inertia2.flange_a, spring.flange_b, damper.flange_b)
    ]
    if u !== nothing 
        push!(eqs, connect(torque.tau, u.output))
        return @named model = ODESystem(eqs, t; systems = [torque, inertia1, inertia2, spring, damper, u])
    end
    ODESystem(eqs, t; systems = [torque, inertia1, inertia2, spring, damper], name)
end
```
We are now ready to simulate the system. We simulate the response of the system to a sinusoidal input and plot how the states evolve in time

```@example mtk_control
model = SystemModel(Sine(frequency=30/2pi, name=:u))
sys = structural_simplify(model)
prob = ODEProblem(sys, Pair[], (0.0, 1.0))
sol = solve(prob, Rodas5())
plot(sol)
```

The next step is to **introduce a controller** and close the loop. We will look at how the closed-loop system behaves with a step on the reference. To handle the discontinuous reference step, we tell the solver to continue even though the step size gets small using `force_dtmin=true`. We also show how to connect a **reference filter** to shape the reference (feedforward). The closed-loop system will thus look like this
```
                                  d│                   
                                   │                   
    ┌───────┐        ┌────────┐    │   ┌────────┐      
 r  │       │      e │        │ u  ▼   │        │        y
────► filter├────+───►  PID   ├────+───►   P    ├───────┬───►
    │       │   -▲   │        │        │        │       │
    └───────┘    │   └────────┘        └────────┘       │
                 │                                      │
                 └──────────────────────────────────────┘
```
```@example mtk_control
using ModelingToolkitStandardLibrary.Blocks: LimPID, SecondOrder, Step, RealOutput
using ModelingToolkitStandardLibrary.Mechanical.Rotational: AngleSensor

@named r = Step(start_time=1)
model = SystemModel()
@named pid = LimPID(k = 400, Ti = 0.5, Td = 1, u_max=350)
@named filt = SecondOrder(d = 0.9, w = 10, x=0, xd=0)
@named sensor = AngleSensor()

connections = [
    connect(r.output, filt.input)
    connect(filt.output, pid.reference)
    connect(pid.ctr_output, model.torque.tau)
    connect(model.inertia1.flange_b, sensor.flange)
    connect(pid.measurement, sensor.phi)
]
closed_loop = structural_simplify(
    ODESystem(connections, t, systems = [model, pid, filt, sensor, r], name = :closed_loop),
)
prob = ODEProblem(closed_loop, Pair[], (0.0, 4.0))
sol = solve(prob, Rodas5())
plot(
    plot(sol, vars = [filt.y, model.inertia1.phi, model.inertia2.phi]),
    plot(sol, vars = [pid.ctr_output.u], title = "Control signal"),
    legend = :bottomright,
)
```

The parameters in the filter and the PID controller were tuned manually here, for automatic tuning of parameters in MTK models, see [Automatic tuning of structured controllers](@ref).

## State feedback using ModelingToolkit
In this example we will instead simulate a state-feedback interconnection ``u = -Lx``. We assume for now that all states are measurable[^observer] and that a suitable feedback-gain matrix ``L`` is calculated using LQR ([`lqr`](@ref)). The scenario this time is **disturbance rejection**, i.e., we simulate that a unit disturbance acts as a force on the second mass between `1 < t < 3`.

The system can be depicted like this, where the `+` node is represented by the `Add` block in the code.
```
       ┌─────┐  y
d      │     ├─────►
───►+──►model│
    ▲  │     ├──┐
    │  └─────┘  │x
    │           │
    │   ┌───┐   │
    └───┤ L │◄──┘
        └───┘
```

[^observer]: For a tutorial building a state estimator, see [State estimation for ModelingToolkit models](@ref) as well as several of the tutorials on MPC.

```@example mtk_control
using ModelingToolkitStandardLibrary.Blocks: MatrixGain, Add
using ControlSystems

@named d = Step(start_time=1, duration=2) # Disturbance
model = SystemModel() # Model with load disturbance
outputs = [inertia1.w, inertia2.w, inertia1.phi, inertia2.phi]
inputs = [torque.tau.u]
op = Dict(inputs .=> 0)
matrices, ssys = ModelingToolkit.linearize(model, inputs, outputs; op) # Specify operating point with keyword op
# matrices = ModelingToolkit.reorder_unknowns(matrices, unknowns(ssys), outputs) 
linsys = ss(matrices...) # Create a StateSpace object

# Design state-feedback gain using LQR
C = linsys.C # Penalizing the output rather than the state since we cannot rely on the state order from MTK
L = -lqr(linsys, 100C'I(4)*C, I(1)) * C
@named state_feedback = MatrixGain(K=L)
@named add = Add() # To add the control signal and the disturbance

model_outputs = [model.inertia1.w, model.inertia2.w, model.inertia1.phi, model.inertia2.phi]
connections = [
    [state_feedback.input.u[i] ~ model_outputs[i] for i in 1:4]
    connect(add.input1, d.output)
    connect(add.input2, state_feedback.output)
    connect(add.output, model.torque.tau)
]
closed_loop = ODESystem(connections, t, systems=[model, state_feedback, add, d], name=:closed_loop)
closed_loop = structural_simplify(closed_loop)
prob = ODEProblem(closed_loop, Pair[], (0., 10.))
sol = solve(prob, Rodas5(), dtmax=0.1); # set dtmax to prevent the solver from overstepping the entire disturbance

plot(plot(sol, idxs=[model.inertia1.phi, model.inertia2.phi]), plot(sol, idxs=[state_feedback.output.u], title="Control signal"))
```

```@example mtk_control
using Test # hide
@test minimum(sol[model.inertia1.phi]) == 0 # hide
@test maximum(sol[model.inertia1.phi]) ≈ 0.05985007413537076 rtol=1e-2 # hide
nothing # hide
```

## Feedforward using an inverse model
In the examples above, we filtered the reference step using a second-order filter. This is important in order to get a dynamically feasible reference trajectory, indeed, a step in a positional reference is discontinuous, requiring unbounded velocities and accelerations to realize.[^TrajectoryLimiters] With the second-order filter, the filtered reference will have continuous position and velocity profiles. However, we relied solely on *error-driven feedback* to follow the reference, implying that an error has to arise before the controller performs any action. In order to increase the reference-following performance, we make use of *feedforward*, i.e., given a model of the controlled system, we can calculate ahead-of-time what control input we ought to send to the plant in order for it to follow the reference. A model that goes from desired output to an input is called an *inverse model*, i.e., it goes from output to input, as opposed to the standard simulation model which is in this context called a forward model. 

[^TrajectoryLimiters]: See also [TrajectoryLimiters.jl](https://github.com/baggepinnen/TrajectoryLimiters.jl) for nonlinear filters that produce dynamically feasible reference trajectories with bounded velocity and acceleration.

Employing an inverse model in ModelingToolkit is straightforward due to its acausal nature. We need to create an identical copy of the model, we call it `inverse_model`, and then we connect it between the filtered reference and the input of the forward model. We connect it "in reverse", i.e., we connect the torque of the inverse to the torque of the forward model, and connect the "sensor output" of the inverse model to the desired sensor output that comes from the filtered reference. 

The interconnection diagram will look like below, where `M` denotes `model` and `iM` denotes the inverse model. `F` denotes the reference filter.
```
                    ┌─────┐
                    │     │
            ┌──────►│ iM  ├─┐
            │       │     │ │
            │       └─────┘ │
            │               │
    ┌─────┐ │       ┌─────┐ │  ┌─────┐
 r  │     │ │       │     │ ▼  │     │
────┤  F  ├─┴──►+──►│ PID ├─+─►│  M  ├─┐y
    │     │     ▲   │     │   u│     │ │
    └─────┘     │   └─────┘    └─────┘ │
                │                      │
                │   ┌─────┐            │
                │   │     │            │
                └───┤ -1  ◄────────────┘
                    │     │
                    └─────┘
```


```@example mtk_control
@named add = Add() # To add the feedback and feedforward control signals
@named inverse_model = SystemModel()
@named inverse_sensor = AngleSensor()
connections = [
    connect(r.output, :r, filt.input) # Name connection r to form an analysis point
    connect(inverse_model.inertia1.flange_b, inverse_sensor.flange) # Attach the inverse sensor to the inverse model
    connect(filt.output, pid.reference, inverse_sensor.phi) # the filtered reference now goes to both the PID controller and the inverse model input
    connect(inverse_model.torque.tau, add.input1)
    connect(pid.ctr_output, add.input2)
    connect(add.output, :u, model.torque.tau) # Name connection u to form an analysis point
    connect(model.inertia1.flange_b, sensor.flange)
    connect(sensor.phi, :y, pid.measurement)  # Name connection y to form an analysis point
]
closed_loop = ODESystem(connections, t, systems = [model, inverse_model, pid, filt, sensor, inverse_sensor, r, add], name = :closed_loop)
prob = ODEProblem(structural_simplify(closed_loop), Pair[], (0.0, 3.0))
sol = solve(prob, Rodas5())
plot(
    plot(sol, vars = [filt.y, model.inertia1.phi, model.inertia2.phi]),
    plot(sol, vars = [pid.ctr_output.u], title = "Feedback control signal", ylims=(-1,1)),
    plot(sol, vars = [inverse_model.torque.tau.u], title = "Feedforward control signal"),
    legend = :bottomright,
)
```

This time, we get perfect reference tracking and the PID controller has zero output!
In practice, the result is not expected to be this perfect due to mismatch between the model used for feedforward and the actual plant, but this approach is nevertheless very useful in improving reference tracking in servo applications. It also has the benefit of *decoupling* the disturbance-rejection properties of the loop from the command-following properties, i.e., *regulation and servoing has been decoupled*. This generally makes it much easier to tune a control system.



## Linear analysis with analysis points
An analysis point is a component inserted in a model at a point of interest for linear analysis. This allows the control designer to derive transfer functions of relevance for the analysis of the system, such as sensitivity functions and loop-transfer functions. Analysis points may also be used to open closed-loop connections without manually redefining the model.

For more details and examples using analysis points, see [Linear analysis with analysis points.](http://mtkstdlib.sciml.ai/dev/API/linear_analysis/), the documentation on [Automatic tuning of structured controllers](@ref) as well as the [instructional video](https://youtu.be/-XOux-2XDGI).


### Example:
In the code that formed a closed-loop system with an inverse model above, analysis points `:r, :u, :y` were inserted when the blocks were connected (indicated in the block diagram). These can be used to derive, e.g., sensitivity functions at the plant output like so:
```@example mtk_control
using ModelingToolkitStandardLibrary.Blocks
matrices, simplified_sys = Blocks.get_sensitivity(closed_loop, :y)
So = ss(matrices...)
matrices, simplified_sys = Blocks.get_comp_sensitivity(closed_loop, :y)
To = ss(matrices...)
bodeplot([So, To], label=["S" "" "T" ""], plot_title="Sensitivity functions")
```
To get the closed-loop transfer function from reference to output, we call `linearize` with the analysis-point names like this
```@example mtk_control
matrices, simplified_sys = ModelingToolkit.linearize(closed_loop, :r, :y)
TF = ss(matrices...)
bodeplot(TF, label="TF", plot_title="Closed-loop transfer function")
```
We see that this transfer function differs from the complementary sensitivity function $T$, this is due to the inverse-model path and the reference pre-filter $F$.

## Further reading
### Batch Linearization and gain scheduling
See [ControlSystemsMTK: Batch Linearization and gain scheduling](https://juliacontrol.github.io/ControlSystemsMTK.jl/dev/batch_linearization/)