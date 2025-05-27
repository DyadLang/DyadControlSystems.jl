# MPC with model estimated from data
Sometimes, the control engineer finds themselves in a situation in which no model of the process they have to control is available. This is often the case in industry, since plants may be complicated, and creating models from first principles is an expensive endeavour. If the task is *regulation*, .i.e., keep some output at a desired set point, a linear model is often sufficient. Indeed, the controller is doing all that it can to keep the process operating around the operating point, so a linear assumption may be valid despite the overall process being nonlinear. 

In situations like these, we may be able to estimate a linear model from input-output data coming from the system during operation, i.e., through *system identification*. In this example, we will consider an industrial heating process, similar in nature to a hair dryer, controlled using linear MPC.

The data we will use for this identification comes from [STADIUS's Identification Database](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html), and they give the following description
> Laboratory setup acting like a hair dryer. Air is fanned through a tube and heated at the inlet. The air temperature is measured by a thermocouple at the output. The input is the voltage over the heating device (a mesh of resistor wires).

More details around the system-identification part in this example is available in the extended tutorial [System identification for an industrial dryer](https://baggepinnen.github.io/ControlSystemIdentification.jl/stable/examples/hair_dryer/). The tutorial you are reading now will only give sparse details on the identification, and then move on to the control design.

This tutorial is also available in video form:
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/z8o83UORuqQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

## System identification
We start by downloading the data, reading it using `readdlm` and packaging the data into an [`iddata`](@ref) object:
```@example DATA_MPC
using DelimitedFiles, Plots, Statistics, LinearAlgebra
using DyadControlSystems, ControlSystemIdentification
using DyadControlSystems.MPC
gr(fmt=:png) # hide

url = "https://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/dryer.dat.gz"
zipfilename = "/tmp/dryer.dat.gz"
path = Base.download(url, zipfilename)
run(`gunzip -f $path`)
data = readdlm(path[1:end-3])
Ts = 0.08
u = data[:, 1]' # voltage
y = data[:, 2]' # air temp
d = iddata(y, u, Ts)

plot(plot(d), coherenceplot(d))
```
The figures above show the data available for identification, as well as the estimated [coherence function](https://en.wikipedia.org/wiki/Coherence_(signal_processing)).[^coherence]

[^coherence]: The coherence function gives an indication on the power transfer between output and input that can be explained by a linear system. This function indicates for which frequencies a linear model may be valid, and it is good system-identification practice to inspect this function before proceeding to estimate an LTI model.

To get a feeling for the dynamics of the process, we estimate an plot its impulse response. Before doing this, we save the mean input and then *detrend* the data, i.e., remove the operating point around which the process is operating.
```@example DATA_MPC
u0 = mean(u, dims = 2)[:]
d = detrend(d)
impulseestplot(d, 40, σ = 3)
```

The system appears to have a couple of samples input delay, this must be handled in one way or another, we choose to handle it explicitly by specifying the input delay when we estimate the model below.

In this case, we fit the model using all available training data. See the [system-identification tutorial](https://baggepinnen.github.io/ControlSystemIdentification.jl/stable/examples/hair_dryer/) mentioned above for more details on how to split up data for training and validation, a good practice for model selection.

We estimate a model using the `arx` routine, and specify that we want two free parameters in both numerator and denominator
```@example DATA_MPC
model_arx = arx(d, 2, 2, inputdelay = 3)
```
This gave us a discrete-time transfer function, we will later convert this to a statespace model using the [`ss`](@ref) function before we use it for MPC design.

We may plot the prediction and simulation performance of the estimated model:
```@example DATA_MPC
predplot(model_arx, d, sysname = "ARX")
simplot!(model_arx, d, ploty = false, sysname = "ARX")
```
This looks reasonably good, so we move on!

## Control design

Next up. we define operating points for the linear MPC controller. The input operating point `u0` was extracted from the data above, to compute the corresponding operating point for the states and outputs, we perform a steady-state simulation using [`step`](@ref) (a step-response simulation), and extract the operating point from the final values.[^1] We use this information to create an [`OperatingPoint`](@ref) as well as create reference arrays for the MPC controller (these will be the same as the operating point in this case). These are the set points we want the controller to maintain the output at.

[^1]: The stationary point``x_0`` when ``u_0`` is given can also be computed from ``0 = \dot x = Ax_0 + Bu_0``.

```@example DATA_MPC
sys = named_ss(ss(model_arx), u = :voltage, y = :temp) # give names to signals for nicer plot labels

stepres = step(sys, 100)
x0 = stepres.x[:, end] * u0[]
y0 = stepres.y[:, end] * u0[]
op = OperatingPoint(x0, u0, y0)

xr = copy(x0)
ur = copy(u0)[:]
yr = sys.C * xr
nothing # hide
```


The data we used for identification only included inputs in a narrow range of voltages. Without further information on how the process works, we take the safe route and constrain the inputs available for the MPC controller to be within this range. We do this by constructing a [`MPCConstraints`](@ref) object:
```@example DATA_MPC
umin = minimum(u, dims = 2)[:]
umax = maximum(u, dims = 2)[:]

constraints = MPCConstraints(; umin, umax)
nothing # hide
```

We now move on to define the solver parameters and the cost function for the MPC controller. In this case, we are interested in *penalizing the outputs* of the process rather than the states, since we estimated a black-box LTI model, we don't even know what the states represent! Our cost-function matrix ``Q_1`` is thus of size ``n_y``, i.e., the number of outputs of the process. We also specify covariance matrices ``R_1, R_2`` for a Kalman filter that will be used to estimate the state of the process from the available measurements.
```@example DATA_MPC
solver = OSQPSolver(
    verbose = false,
    eps_rel = 1e-6,
    max_iter = 5000,
    check_termination = 5,
    polish = true,
)

Q1 = Diagonal(ones(sys.ny))   # output cost matrix
Q2 = 0.01 * spdiagm(ones(sys.nu)) # control cost matrix

R1 = I(sys.nx)
R2 = I(sys.ny)
kf = KalmanFilter(ssdata(sys)..., R1, R2, MvNormal(x0, R1))
nothing # hide
```
The last step before simulating the MPC controller is to define the prediction model and the MPC-problem structure. We create a [`LinearMPCModel`](@ref) containing the system model and the Kalman filter. This object also keeps track of the operating point. Since we want to control the *output* of the system rather than the state of the system, we also pass the ``C`` matrix of the system using `z = sys.C`. This will tell the prediction model that we are interested in regulating ``z = Cx`` instead of ``x`` directly, and we may thus provide references for ``z = y`` instead of for ``x``. We do this by passing the argument `r = yr` when creating the problem.

We also define the prediction horizon for the MPC controller, ``N``. We set this to `N = 10` based on the impulse-response we saw above, after about 10 samples, the impulse-response has mostly dissipated.
```@example DATA_MPC
N = 10
predmodel = LinearMPCModel(sys, kf; constraints, op, x0, z = sys.C)
prob = LQMPCProblem(predmodel; Q1, Q2, N, solver, r = yr)
nothing # hide
```

## Simulation

Our MPC controller is now setup, and we specify some parameters related to the simulation. Let's simulate `T = 80` time steps from a random initial condition:
```@example DATA_MPC
T = 80 # Simulation length (time steps)
x0sim = 0 * x0 .+ randn.()
hist = MPC.solve(prob; x0 = x0sim, T)
plot(hist, ploty = true)
hline!([umin[] umax[]], sp = 2, l = (:black, :dash), primary = false)
```
Looks pretty good! but this was an easy task, we did not simulate any model error and there were no disturbances acting on the system.

In the next scenario, we simulate a periodic disturbance acting on the process input. The disturbance is given by
```math
d(t) = \sin(ωt)
```
```@example DATA_MPC
ω = 2π
disturbance = function (u, i)
    sin(ω * i * Ts)
end
histd = MPC.solve(prob; x0 = x0sim, T, verbose = false, disturbance)
plot(histd, ploty = true)
hline!([umin[] umax[]], sp = 2, l = (:black, :dash), primary = false)
```
This time, we see that the output oscillates around the set-point. Not too bad, but not great either. Can we do better?

## Add disturbance model
If the disturbance acting on the system is measurable, we can include it as an input to the system and thus improve the regulation performance. However, even if we cannot measure the disturbance, there's still hope to perform better as long as we know some properties of the disturbance!

Say that we know the frequency of the disturbance, but it is otherwise unknown. In this scenario, which is common in practice, we can augment the system model with a model of the disturbance. When we apply a state estimator to the augmented system, the state estimator will be nice enough to estimate the disturbance for us, based on its affect on the observable measurements. To do this, we make use of the function [`add_resonant_disturbance`](@ref). We indicate the frequency of the disturbance as well as the relative damping ``ζ`` which we set to something small. Lastly, we indicate how the disturbance enters the system, since it enters at the input, it has the same input matrix as the system, `sys.B`.
```@example DATA_MPC
ζ = 1e-3
sys2_ = add_resonant_disturbance(sys, ω, ζ, sys.B)
sys2 = named_ss(sys2_, u = :voltage, y = :temp)
```
This will create an augmented model with two additional states, modeling a resonance.

With the augmented system `sys2`, we must provide an augmented covariance matrix `R_1` for the Kalman filter. The covariance we assign to the two additional states will determine how fast the observer is at estimating the disturbance. Higher values gives faster estimation, but also more sensitivity to measurement noise.

We also create new initial conditions since the state of the system now has two additional components. 
```@example DATA_MPC
R12 = cat(R1, 0.2I(2), dims = (1, 2))
kf2 = KalmanFilter(ssdata(sys2)..., R12, R2)

x02 = [x0; zeros(2)]
x02sim = [x0sim; zeros(2)]
op2 = OperatingPoint(x02, u0, y0)

predmodel2 = LinearMPCModel(sys2, kf2; constraints, op = op2, x0 = x02, z = sys2.C)
prob2 = LQMPCProblem(predmodel2; Q1, Q2, N, solver, r = yr)
```
This time, we need to provide two different models to the `MPC.solve` function. The prediction model that the MPC controller uses is augmented with a disturbance model, but the "true system" that we are controlling does not have any such state components. We thus define the three arguments `dyn_actual, x0_actual` and `Cz_actual` which specifies the dynamics, initial state and output matrix for the true system. This mechanism allows us to simulate MPC controllers where the prediction model of the controller has model errors or a different number of states, like here.
```@example DATA_MPC
dyn_actual = sys
x0_actual  = x0sim
Cz_actual  = sys.C

hist2 = MPC.solve(
    prob2;
    x0 = x02sim,
    T,
    verbose = false,
    disturbance,
    dyn_actual,
    x0_actual,
    Cz_actual,
)
plot(hist2, ploty = true, legend=:bottomright)
hline!([umin[] umax[]], sp = 2, l = (:black, :dash), primary = false)
```
This time, we see that the controller does a much better job at rejecting the periodic disturbance. We also see that it takes a while before the observer (state estimator) has estimated the disturbance, this is because the covariance for the disturbance states in `R12` was set very low, `0.2I`, try increasing this value and the controller will reject the disturbance much faster!

## Adding output constraints

Since we are dealing with a black-box model identified from data, we also show how one can add *output constraints* to the problem (for simplicity, we go back to use the problem formulation without the disturbance), where the *constrained output* $v_{min} ≤ v ≤ v_{max}$ can be any linear function of the state and the input, $v = C_vx + D_vu$. In this case, we want to limit the rate at which the *output* changes, i.e., constrain
```math
\begin{aligned}
\Delta y &= y_{k+1} - y_k \\
~ &= C(Ax_k + Bu_k) - Cx_k \\
~ &=  C(A-I)x_k + CBu_k
\end{aligned}
```

For this, we make use of [`MPC.LinearMPCConstraints`](@ref) which expects the matrices ``C_v`` and ``D_v`` as well as the upper and lower bounds for ``v``. Since we want to keep our constraints on ``u``, we let ``v = [\Delta y, u]``, and assemble the matrices ``C_v`` and ``D_v`` accordingly:
```@example DATA_MPC
Δymin = [-1*Ts]
Δymax = [1*Ts]
vmin  = [Δymin; umin]
vmax  = [Δymax; umax]

soft_indices = 1:1 # Indicate that we consider the Δy constraint a soft constraint
A,B,C,D = ssdata(sys)
Cv = [C*(A-I); zeros(sys.nu, sys.nx)]
Dv = [C*B; I(sys.nu)]
constraints3 = MPC.LinearMPCConstraints(; vmin, vmax, Cv, Dv, soft_indices)
predmodel3   = LinearMPCModel(sys, kf; constraints = constraints3, op, x0, z = sys.C)
prob3        = LQMPCProblem(predmodel3; Q1, Q2, N, solver, r = yr)
hist3        = MPC.solve(prob3; x0 = x0sim, T)

plot(hist, ploty = true, lab="Original problem")
plot!(hist3, ploty = true, lab="With Δy constraint")
hline!([umin[] umax[]], sp = 2, l = (:black, :dash), primary = false)
```

In this case, we see that the controller actually works in the opposite direction in the beginning in order to slow down the rate of change of the output, thus fighting the natural tendency of the system to approach the set point. Since the control signal is also constrained, the controller cannot meet the ``\Delta y`` constraint during the first 0.7s of the simulation. After this, the rate of change of the output remains constant for a while until it has reached the set point. 

## Summary
This tutorial has gone through the steps of
- Estimating a model from data
- Designing an MPC controller
- Simulating a disturbance response
- Augmenting the model with a disturbance model

The final product was an MPC controller that successfully rejected the periodic disturbance, without having access to measurements of it.