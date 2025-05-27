# PID Autotuning

The [PID controller](https://en.wikipedia.org/wiki/PID_controller) is a classical control method that is widely used due to its simplicity and ability to control many practically occurring systems. The parameters of a PID controller can be tuned in multiple different ways:
- Trial and error, in simulation or on the real system
- Loop shaping
- Automatic tuning

This page details the automatic tuning features offered by DyadControlSystems. Automatic tuning offers the possibility of achieving an optimal response, subject to constraints on closed-loop robustness with respect to uncertainty and noise amplification. While this page refers to the automatic tuning of PID controllers in isolation, we refer the reader to [Automatic tuning of structured controllers](@ref) for tuning of more general structured controllers.


The PID autotuning in DyadControlSystems is based on step-response optimization, subject to robustness constraints on closed-loop sensitivity functions.  The method performs joint optimization of a PID controller and a measurement filter on the form
```math
K(s) = C(s) F(s) = (k_p + k_i/s + k_d s)  \dfrac{1}{(sT_f)^2/(4ζ^2) + T_fs + 1}, \quad ζ = 1
```
by solving
```math
\operatorname{minimize}_{C, F} \int m(e(t)) dt
```
```math
\text{subject to} \\
||S(s)||_\infty \leq M_S \\
||T(s)||_\infty \leq M_T\\
||KS(s)||_\infty \leq M_{KS}
```
where $e(t) = r(t) - y(t)$ is the control error and $m$ is a user-defined metric (defaults to `abs2`). The method allows the user to make several choices, such as
- What are the input and output of the system of which the response is optimized? This allows the user to optimize both reference and disturbance-step responses. These inputs and outputs may be multivariate.
- What are the input and output to the controller, this may differ from the input and output of the system.
- The reference value `r(t)` .

The autotuning functions operate on [SISO](https://en.wikipedia.org/wiki/Single-input_single-output_system) `LTISystem`s from [ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl). If you have a ModelingToolkit model, you may obtain a linear system model using [linearization](https://help.juliahub.com/DyadControlSystems/dev/linear_analysis/). The general workflow for autotuning is
1. Define a plant model, ``P(s)``
2. If the model `P` is MIMO, define which inputs and outputs are connected to the controller, and between which signals the response is optimized.
3. Define desired constraints on the maximum magnitudes $M_S, M_T, M_{KS}$ of the sensitivity functions $S(s) = \dfrac{1}{1 + P(s)K(s)}$, $T(s) = \dfrac{P(s)K(s)}{1+ P(s)K(s)}$ and $K(s)S(s) = \dfrac{K(s)}{1+ P(s)K(s)}$. See sections below for guidance.
4. Choose solver parameters (optional) and create an [`AutoTuningProblem`](@ref).
5. Call [`solve`](@ref).
6. Plot the result.

If you want to use the optimized controller in a ModelingToolkit simulation, see [`OptimizedPID`](@ref).

Examples of this are shown below.

Functions:
- [`solve`](@ref): solve an autotuning problem.
- [`AutoTuningProblem`](@ref): define an autotuning problem.
- [`OptimizedPID`](@ref): obtain an `ODESystem` representing the tuned PID controller.

## Getting started
DyadControlSystems contains a Pluto-based graphical app (GUI) for PID-controller tuning using the two methods below, usage of this app is documented under [PID autotuning GUI](@ref). This example demonstrates the non-graphical interface.

### A simple first-order plant with dead time
The following example optimizes a PID controller with a low-pass filter. We specify the minimum number of options in this example, see [`AutoTuningProblem2`](@ref) for a complete list of user options.

```@example iae
using DyadControlSystems, Plots
using DyadControlSystems: AutoTuningProblem2, OptimizedPID2 # These are experimental and not yet exported
gr(fmt=:png) # hide
# Process model (continuous time LTI SISO system).
τ = 4 # Time constant
L = 1 # Delay
K = 1 # Gain
P = tf(K, [τ, 1.0])*delay(L) # process dynamics
Ts = 0.1 # Discretization time
Tf = 25  # Simulation time

# Robustness constraints
Ms = 1.2   # Maximum allowed sensitivity function magnitude
Mt = Ms    # Maximum allowed complementary sensitivity function magnitude
Mks = 10.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.

prob1 = AutoTuningProblem2(P; Ms, Mt, Mks, Ts, Tf)

# p0 = Float64[1, 1, 0, 0.001] # Initial parameter guess can be optionally supplied, kp, ki, kd, T_filter
solve(prob1) # hide
res1 = solve(prob1)
plot(res1)
```

The figure shows the Nyquist curve of the loop-transfer function $P(s)K(s)$ using the optimized controller, as well as circles corresponding to the chosen constraints. The top figures show Bode plots of the closed-loop sensitivity functions together with the constrains, and the lower left figure shows the response to a unit load-disturbance step. Note, the response to a _reference_ step is not part of the optimization criterion, and optimized suppression of load disturbances often leads to a  suboptimal response to reference steps. If steps are expected in the reference signal, you may opt to instead optimize a reference-step response by providing the keyword argument `step_input = :reference_input`, more on this in the following examples. 

## Reference vs. disturbance step response
The default behavior is to optimize the closed-loop response to a disturbance step appearing at the plant input. If you want to optimize the response to a reference step, you may provide the keyword argument `step_input = :reference_input`. The following example demonstrates this using the same plant model as above:

```@example iae
prob2 = AutoTuningProblem2(P; Ms, Mt, Mks, Ts, Tf, step_input = :reference_input)
solve(prob2) # hide
res2 = solve(prob2)
plot(res2)
```
Note how the plot in the lower-left corner now shows the response to a reference step instead.

To select a custom input of `P` as the input of the step, make sure `P` is an instance of `NamedStateSpace` and pass the name for the input of choice.

## Selecting between P, PI, PD, PID controllers
You may select which type of controller to optimize by providing bounds for the parameter array. The optimizer uses the parameter array
```julia
p = [kp, ki, kd, Tf]
```
where `kp`, `ki`, and `kd` are the proportional, integral, and derivative gains, respectively (parallel form), and `Tf` is the time constant of the low-pass filter. By providing the lower and upper bounds for this array, `lb, ub`, you can force some parameters to be zero. To optimize a PI controller, set `kd` to zero, and to optimize a PD controller, set `ki` to zero. The following example demonstrates this:

```@example iae
prob3 = AutoTuningProblem2(P; Ms, Mt, Mks, Ts, Tf, lb = [0.0, 0.0, 0.0, 0.0], ub = [Inf, Inf, 0, Inf]) # No derivative gain
solve(prob3) # hide
res3 = solve(prob3)
plot(res3)
```
Notice how the controller transfer function is now much smaller. We can also inspect the optimized parameters by looking at the `res` object:

```@example iae
res3.p
```
and we see that the derivative term is zero.

## Initial guess

When we have called `solve(prob)` above, we have not provided any initial guess, and instead relied on an automatically generated initial guess. If the problem appears to be difficult to solve, an initial guess on the form
```julia
p0 = [kp, ki, kd, Tf]
```
may be provided to `solve(prob, p0)`.


## Performance vs. robustness

The optimization problem optimizes performance, subject to robustness constraints. Lower values of `Ms`, `Mt`, and `Mks` increase the robustness of the controller, but generally decreases performance. The following example modifies these constraints to demonstrate the trade-off between performance and robustness. We compare the result to the first example above:


### Increased robustness
```@example iae
prob4 = AutoTuningProblem2(P; Ms = 1.1, Mt = 1.1, Mks, Ts, Tf) # A very robust and conservative controller with low values of Ms and Mt
res4 = solve(prob4)
plot(plot(res1, plot_title="Result 1"), plot(res4, plot_title="Result 4"), size=(800,500), titlefontsize=8, labelfontsize=8)
```
In this case, the disturbance rejection is slower and the peak disturbance response is higher, but the robustness margins are increased:
```@example iae
plot(diskmargin((prob1.P*res1.K).sys), label="Result 1")
plot!(diskmargin((prob4.P*res4.K).sys), label="Result 4")
```

### Increased performance
Below, we demonstrate the opposite, where we relax the robustness constraints to obtain a less robust controller with higher performance:
```@example iae
prob5 = AutoTuningProblem2(P; Ms = 1.5, Mt = 1.5, Mks, Ts, Tf) # A less robust controller with high values of Ms and Mt
res5 = solve(prob5)
plot(plot(res1, plot_title="Result 1"), plot(res5, plot_title="Result 5"), size=(800,500), titlefontsize=8, labelfontsize=8)
```
In this case, the disturbance rejection is faster and the peak disturbance response is lower, but the robustness margins are decreased:
```@example iae
plot(diskmargin((prob1.P*res1.K).sys), label="Result 1")
plot!(diskmargin((prob5.P*res5.K).sys), label="Result 5")
```

## Limiting noise amplification
The constraint `Mks` limits the peak amplification of noise from the process output to the control signal. The following example demonstrates the effect of decreasing `Mks`:

```@example iae
prob6 = AutoTuningProblem2(P; Ms, Mt, Mks = 1.0, Ts, Tf) # A very conservative controller with low value of Mks
res6 = solve(prob6)
plot(plot(res1, plot_title="Result 1"), plot(res6, plot_title="Result 6"), size=(800,500), titlefontsize=8, labelfontsize=8)
```
In this case, the disturbance rejection is slower and the peak disturbance response is higher, but the noise amplification is limited:
```@example iae
timevec = 0:Ts:Tf
noisy_measurements = randn(1, length(timevec))
plot(lsim(res1.G[:u_controller_output_C, :reference_input], noisy_measurements, timevec, method=:zoh), label="Result 1")
plot!(lsim(res6.G[:u_controller_output_C, :reference_input], noisy_measurements, timevec, method=:zoh), label="Result 6", title="Control-signal response to measurement noise")
```
here, we used the closed-loop transfer function from reference input to control signal output, which is the same transfer function as that from measurement noise to control signal (the sign is reversed). Notice how the controller from `res6` is amplifying measurement noise much less.

Another strategy to limit noise amplification is to place a bound on the filter time constant ``T_f``.

## Filter options
The user may choose between a first-order filter and a second-order filter (default). The second-order filter is defined as
```math
F(s) = \dfrac{1}{(sT_f)^2/(4ζ^2) + T_fs + 1}
```
where `T_f` is the filter time constant and `ζ` is the damping ratio (which defaults to 1). The second-order filter is connected in series with the controller, and thus acts on all terms (P,I,D). The first-order filter is defined as
```math
F(s) = \dfrac{1}{sT_f + 1}
```
and only acts on the derivative term.

The filter order is chosen by providing the keyword argument `filter_order = 1` or `filter_order = 2`. If a second-order filter is chosen, the damping ratio may be optionally optimized by passing the keyword argument `optimize_d = true`, this expands the parameter vector to `p = [kp, ki, kd, Tf, d]`. The following example demonstrates this:

```@example iae
prob7 = AutoTuningProblem2(P; Ms, Mt, Mks, Ts, Tf, filter_order = 1) # First-order filter
res7 = solve(prob7)
plot(plot(res1, plot_title="Result 1"), plot(res7, plot_title="Result 7"), size=(800,500), titlefontsize=8, labelfontsize=8)
```
Notice how result 1 has high-frequency [roll-off](https://en.wikipedia.org/wiki/Roll-off) in the controller, while result 7 does not, due to having a single filter pole only.

Below, we optimize the damping parameter `d` of the second-order filter:

```@example iae
ub = [Inf, Inf, Inf, Inf, 2.5] # Upper bounds for the parameters
lb = [0.0, 0.0, 0.0, 0.0, 0.7] # Lower bounds for the parameters
prob8 = AutoTuningProblem2(P; Ms, Mt, Mks, Ts, Tf, filter_order = 2, optimize_d = true, ub, lb) # Second-order filter with optimized damping
res8 = solve(prob8)
plot(plot(res1, plot_title="Result 1"), plot(res8, plot_title="Result 8"), size=(800,500), titlefontsize=8, labelfontsize=8)
```
The result in this case is very similar to the first result, you may notice a _slightly_ wider high-frequency peak in the controller transfer function in the latter result. Here, we provided custom parameter bounds to allow the damping parameter to vary between 0.7 and 2.5., the default bounds if none is provided are $1/\sqrt(2) \leq d \leq 1$. A damping ratio above 1 leads to an over-damped filter, where one of the poles go towards infinity, allowing the controller high-frequency peak to become wider.


## Selecting the solver

The optimization problem is solved using [Optimization.jl](https://docs.sciml.ai/Optimization/stable/), and any solver within Optimization.jl that supports nonlinear constraints may be used. The default solver is Ipopt, constructed using the convenience function [`IpoptSolver`](@ref) like this `solver = IpoptSolver(exact_hessian=false)` (use of exact Hessian is not supported). The function as several arguments that allow you to customize the solution process.

## The `AutoTuningResult`
The structure returned by `solve(prob)` contains the following fields:
- `K`: The optimized controller on state-space form.
- `G`: The closed-loop transfer function used to compute the constrained transfer functions.
- `p`: The optimized parameters.
- `prob`: The `AutoTuningProblem` used to solve the problem.
- `sol`: The solution object returned by the Optimization.jl

The object can be plotted using
```julia
using Plots
plot(res)
```


## Making use of the optimized controller
The optimized parameters are available as the field `result.p`, which is a vector on the form `[kp, ki, kd, Tf]` for the transfer function
```math
K(s) = C(s) F(s) = (k_p + k_i/s + k_d s)  \dfrac{1}{(sT_f)^2/(4ζ^2) + T_fs + 1}, \quad ζ = 1
```
or `[kp, ki, kd, Tf, d]` for the transfer function
```math
K(s) = C(s) F(s) = (k_p + k_i/s + k_d s)  \dfrac{1}{(sT_f)^2/(4ζ^2) + T_fs + 1}, \quad ζ = d
```

To convert the PID parameters to _standard form_, use [`convert_pidparams_from_parallel
`](https://juliacontrol.github.io/ControlSystems.jl/dev/lib/synthesis/#ControlSystemsBase.convert_pidparams_from_parallel-Tuple{Any,%20Any,%20Any,%20Symbol}) like this
```julia
using ControlSystemsBase
kp, ki, kd, Tf = res.p
K, Ti, Td = convert_pidparams_from_parallel(kp, ki, kd, :standard)
```

The optimized controller is also available as a `NamedStateSpace` object through the field `result.K`.

### In ModelingToolkit
If you want to use the optimized controller in a ModelingToolkit model, you may use the convenience function [`OptimizedPID2`](@ref) that returns an `ODESystem` representing the tuned PID controller.



## Advanced usage
### Penalizing control action
The constraint `Mks` effectively limits the _peak amplification_ of noise from the process output to the control signal. If you want to _penalize_ the control signal the optimization, you may include the control signal among the minimized outputs by augmenting `P` with an output corresponding to the control input. If the transfer function from control signal input to penalized control signal output is static, the optimized controller will typically have a very small integral action resulting in steady-state errors. Instead, we suggest including a high-pass filtered version of the control signal to the outputs, demonstrated below. The high-pass filter allows us to tune the control-signal penalty be tuning the cutoff frequency and the gain.

In the example below, we assign names to the outputs of `P` in order to be able to address them when assigning the `step_output` and `measurement` arguments. Since `P` is a system with delay, we form a Padé approximation of order 3 of `P` explicitly before constructing the `NamedStateSpace` object (delay systems cannot be represented as state-space systems, they are infinite-dimensional.). Earlier, we did not have to perform this step manually, it was done automatically by the constructor of [`AutoTuningProblem2`](@ref).
```@example iae
high_pass_filter = 100tf([1,0], [1, 100]) #* tf(100^2, [1, 2*100, 100^2])
Paugmented = [pade(P, 3); high_pass_filter]
Pud = named_ss(Paugmented, y = [:y, :du]) # Give names to the outputs so that we may refer to them when specifying the step output and measurement
step_output = [:y, :du]
measurement = :y
ref = [0.0, 0.0] # Since we have more than one output, we must specify the reference for both
prob9 = AutoTuningProblem2(Pud; prob1.w, Ms, Mt, Mks, Ts, Tf, step_output, measurement, ref)
res9 = solve(prob9)
plot(plot(res1, plot_title="Result 1"), plot(res9, plot_title="Result 9"), size=(800,500), titlefontsize=8, labelfontsize=8)
```
The step-response plot now shows both the measured output and the high-pass filtered control signal, the sum of which are minimized. Notice also how the transfer function $KS$ is significantly smaller than its constraint allows due to the control-signal penalty. We may compare the weighted $\mathcal{H}_2$ norm of the controller noise amplification in result 1 and 9
```@example iae
(
    norm(high_pass_filter*res1.G[:u_controller_output_C, :reference_input]), # The measurement noise enters at the same place as the reference
    norm(high_pass_filter*res9.G[:u_controller_output_C, :reference_input])
)
```
where the latter should be smaller. We can also simulate the response to measurement noise:
```@example iae
plot(lsim(res1.G[:u_controller_output_C, :reference_input], noisy_measurements, timevec, method=:zoh), label="Result 1")
plot!(lsim(res9.G[:u_controller_output_C, :reference_input], noisy_measurements, timevec, method=:zoh), label="Result 9", title="Control-signal response to measurement noise")
```
and notice that the last controller indeed amplifies measurement noise less.



## Index
```@index
Pages = ["autotuning2.md"]
```
```@autodocs
Modules = [DyadControlSystems]
Pages = ["autotuning2.jl"]
Private = true
```
