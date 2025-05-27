# PID Autotuning

The [PID controller](https://en.wikipedia.org/wiki/PID_controller) is a classical control method that is widely used due to its simplicity and ability to control many practically occurring systems. The parameters of a PID controller can be tuned in multiple different ways:
- Trial and error, in simulation or on the real system
- Loop shaping
- Automatic tuning

This page details the automatic tuning features offered by DyadControlSystems. Automatic tuning offers the possibility of achieving an optimal response, subject to constraints on closed-loop robustness with respect to uncertainty and noise amplification. While this page refers to the automatic tuning of PID controllers in isolation, we refer the reader to [Automatic tuning of structured controllers](@ref) for tuning of more general structured controllers.


The PID autotuning in DyadControlSystems is based on disturbance step-response optimization, subject to robustness constraints on closed-loop sensitivity functions. We currently implement two methods, one that optimizes a PID controller on the form
```math
K(s) = (k_p + k_i/s + k_d s)
```
by solving
```math
\operatorname{minimize}_K \int e(t) dt
```
```math
\text{subject to} \\
||S(s)||_\infty \leq M_S \\
||T(s)||_\infty \leq M_T \\
||KS(s)||_\infty \leq M_{KS}
```
where $e(t) = r(t) - y(t)$ is the control error.

The second method performs joint optimization of a PID controller and a measurement filter on the form
```math
K(s) = C(s) F(s) = (k_p + k_i/s + k_d s)  \dfrac{1}{(sT)^2 + 2ζTs + 1}, ζ = 1/√2
```
by solving
```math
\operatorname{minimize}_{C, F} \int |e(t)| dt
```
```math
\text{subject to} \\
||S(s)||_\infty \leq M_S \\
||T(s)||_\infty \leq M_T\\
||KS(s)||_\infty \leq M_{KS}
```

The autotuning functions operate on [SISO](https://en.wikipedia.org/wiki/Single-input_single-output_system) `LTISystem`s from [ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl). If you have a ModelingToolkit model, you may obtain a linear system model using [linearization](https://help.juliahub.com/DyadControlSystems/dev/linear_analysis/). The general workflow for autotuning is
1. Define a plant model, ``P``
2. Define desired constraints on the maximum magnitudes $M_S, M_T, M_{KS}$ of the sensitivity functions $S = \dfrac{1}{1+ PK}$, $T = \dfrac{PK}{1+ PK}$ and $KS = \dfrac{K}{1+ PK}$.
3. Choose problem and solver parameters and create an [`AutoTuningProblem`](@ref).
4. Call [`solve`](@ref).
5. Plot the result.

If you want to use the optimized controller in a ModelingToolkit simulation, see [`OptimizedPID`](@ref).

Examples of this are shown below.

Functions:
- [`solve`](@ref): solve an autotuning problem.
- [`AutoTuningProblem`](@ref): define an autotuning problem.
- [`OptimizedPID`](@ref): obtain an `ODESystem` representing the tuned PID controller.

## Getting started
DyadControlSystems contains a Pluto-based graphical app (GUI) for PID-controller tuning using the two methods below, usage of this app is documented under [PID autotuning GUI](@ref). This example demonstrates the non-graphical interface.

### Integrated absolute error (IAE)
The following example optimizes a PID controller with a low-pass filter using the method from
> K. Soltesz, C. Grimholt, S. Skogestad. Simultaneous design of proportional–integral–derivative controller and measurement filter by optimisation. Control Theory and Applications. 11(3), pp. 341-348. IET. 2017.

```@example iae
using DyadControlSystems, Plots
gr(fmt=:png) # hide
# Process model (continuous time LTI SISO system).
T = 4 # Time constant
L = 1 # Delay
K = 1 # Gain
P = tf(K, [T, 1.0])*delay(L) # process dynamics
Ts = 0.1 # Discretization time
Tf = 25  # Simulation time

# Robustness constraints
Ms = 1.2   # Maximum allowed sensitivity function magnitude
Mt = Ms    # Maximum allowed complementary sensitivity function magnitude
Mks = 10.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
w = 2π .* exp10.(LinRange(-2, 2, 200)) # frequency grid

prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts, Tf, metric = :IAE)

# p0 = Float64[1, 1, 0, 0.001] # Initial parameter guess can be optionally supplied, kp, ki, kd, T_filter
solve(prob) # hide
res = solve(prob)
plot(res)
```

The figure shows the Nyquist curve of the loop-transfer function $P(s)K(s)$ using the optimized controller, as well as circles corresponding to the chosen constraints. The top figures show Bode plots of the closed-loop sensitivity functions together with the constrains, and the lower left figure shows the response to a unit load-disturbance step as well as a reference-step response. Note, the response to a reference step is not part of the optimization criterion, and optimized suppression of load disturbances often leads to a  suboptimal response to reference steps. If steps are expected in the reference signal, reference shaping using a pre-filter is recommended (called a 2-DOF design, realized, for example, by the introduction of ``W_r`` in the diagram of the following [design example](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/hinf_connection/)). 

### Integrated error (IE)
The following example optimizes a PID controller using the method from
> M. Hast, K. J. Astrom, B. Bernhardsson, S. Boyd. PID design by convex-concave optimization. European Control Conference. IEEE. Zurich, Switzerland. 2013.

This method optimizes integrated error (not integrated *absolute* error). This problem is relatively easy to solve and corresponds well to IAE if the system is well damped. If convergence of the above method (IAE) appears difficult, this method can be used as initialization by choosing `metric = :IEIAE`.

```@example ie
using DyadControlSystems, Plots
T = 4 # Time constant
K = 1 # Gain
P = tf(K, [T, 1.0]) # process dynamics

## Robustness constraints
Ms  = 1.2  # Maximum allowed sensitivity function magnitude
Mt  = Ms   # Maximum allowed complementary sensitivity function magnitude
Mks = 10.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
w   = 2π .* exp10.(LinRange(-2, 2, 50)) # frequency vector

p0 = Float64[1, 1, 0.1] # Initial guess. Use only two parameters to tune a PI instead of PID controller
prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.1, Tf=25.0, metric = :IE) # Set the metric here

solve(prob, p0) # hide
res = solve(prob, p0)
plot(res)
```

## Choosing metric
The `metric = :IE` problem optimizes integrated error $\int e(t) dt$ (not integrated *absolute* error). This problem is relatively easy and fast to solve and corresponds well to IAE if the system is well damped. If this metric is chosen, a PI or PID controller is tuned, determined by the number of parameters in the initial guess. The method requires a stabilizing controller as an initial guess. If the plant is stable, the zero controller is okay. If the initial guess is not stabilizing, an attempt at automatically finding a feasible initial guess is made.

If the response is oscillatory, the `metric = :IE` metric is expected to perform poorly and `metric = :IAE` is recommended. If `metric = :IAE` is chosen, a PID controller with a low-pass filter is tuned by minimizing $\int |e(t)| dt$. This problem is nonconvex and can be difficult to solve. This method can be initialized with the `IE` method by selecting `metric = :IEIAE`.


## Index
```@index
Pages = ["autotuning.md"]
```
```@autodocs
Modules = [DyadControlSystems]
Pages = ["autotuning.jl"]
Private = false
```
```@docs
DyadControlSystems.AutoTuningResult
```