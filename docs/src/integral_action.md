# Integral action
Integral action refers to an often desired property of the controller, which is the ability to *eliminate the steady-state error* and achieve *offset-free tracking*. This is achieved by, in one way or another, adding an integral term to the controller. This page will discuss the addition of integral action to various forms of controllers, starting with the simplest form, explicit integration, and moving on to more complex forms such as disturbance-model augmentation and output-error augmentation.

The familiar PI(D) controller achieves integral action by explicitly integrating the error. Consider the PI controller on the form
$$C(s) = k_p + k_i/s$$, when the loop is closed around this controller
```
               │d
      ┌─────┐  │  ┌─────┐
r   e │     │u ▼  │     │ y
──+──►│  C  ├──+─►│  P  ├─┬─►
  ▲   │     │     │     │ │
  │-  └─────┘     └─────┘ │
  │                       │
  └───────────────────────┘
```
we get the following transfer function from the reference input to the control error
```math
E(s) = \dfrac{1}{I + PC}R(s)
```
This transfer function is called the *(output) sensitivity function*, ``S(s)``, and we may determine the steady-state tracking error by taking the limit of this function as $s \to 0$. If either ``P`` or ``C`` has an integral term (and none of them have a zero in the origin), ``S(s) \to 0`` as $s \to 0$, and the steady-state error is eliminated. However, it is not enough to consider the steady-state error in response to a reference input, as the controller is also be used to *reject disturbances*. The transfer function from a disturbance appearing at the plant input to the control error is given by
```math
E(s) = \dfrac{P}{I + PC}D(s)
```
and in this case, it is not enough for ``P`` to contain an integrator in order to eliminate steady-state errors. We can easily see this if we take ``C(s) = 1``, in which case ``P(s) \to ∞`` and ``\frac{P}{I + PC} \to 1`` as $s \to 0$, we thus conclude that the controller must contain integral action in order to fully reject low-frequency disturbances.

We may perform a more formal analysis using the *final-value theorem*, that states that the steady-state error is given by
```math
e_{∞} = \lim_{t → \infty} e(t) = \lim_{s → 0} sE(s)
```
If ``D(s)`` is a step disturbance with Laplace transform ``D(s) = 1/s``, we get 
```math
e_{∞} = \lim_{s → 0} \dfrac{P}{I + PC}
```
which, if ``P(s)`` contains an integrator but ``C(s)`` does not, is ``1/C(0)``. Clearly, adding an integrator to ``C`` such that ``C(s) → ∞`` as $s \to 0$ will eliminate the steady-state error also in this case. If ``D(s)`` is a ramp disturbance with Laplace transform ``D(s) = 1/s^2``, we conclude using the same analysis that ``C(s)`` must contain two integrators in order to eliminate the steady-state error.


When using a PI(D) controller, we indeed achieve integral action by explicitly integrating the controller input. But what about if we are making use of some form of state-feedback controller, obtained through, e.g., pole placement or LQR design? In this case, we have three options
- Form the signal ``e = r - y`` and *augment the system model* with the dynamics ``\dot{x}_e = e``, where ``e`` is considered a new input and ``x_e`` a new "error integral state". When using feedback from the integral state ``x_e``, the controller contains explicit integral action.
- Augment the system model with an explicit integrator of the input. This will change the apparent input of the plant to be the derivative of the control input, and an explicit integration of the control input will have to be performed before the control signal is sent to the actual plant. This method can be viewed as adding the transfer function ``1/s`` to ``P(s)`` to form ``P_a = P\frac{1}{s}``. After the controller ``C(s)`` has been designed for the augmented plant ``P_a``, the integrator is *reassociated* with ``C``, such that we view the loop-transfer function as ``P(s) \cdot (\frac{1}{s}C(s))``. The operation ``\frac{1}{s}`` thus has to be performed explicitly on the output of ``C`` before it is sent to the plant, e.g., by multiplying the found controller `C` by `tf(1, [1, 0])`.
- Augment the system model with a disturbance model. If we model a low-frequency disturbance ``1/s`` acting on the plant input, get the following augmented system ``\dot{x} = Ax + Bu + B_d d``,  with `d` given by
```math
W : \; \begin{aligned}
\dot{x}_d &= w_d \\
d &= x_d
\end{aligned}
```
where ``w_d`` is a flat-spectrum disturbance with Laplace transform 1.
```
               │wd
               ▼
            ┌─────┐
            │  W  │
            └──┬──┘
               │d
      ┌─────┐  │  ┌─────┐
r   e │     │u ▼  │     │ y
──+──►│  C  ├──+─►│  P  ├─┬─►
  ▲   │     │     │     │ │
  │-  └─────┘     └─────┘ │
  │                       │
  └───────────────────────┘
```
This adds a pure integrator ``\dot{x}_d = w_d`` to the plant dynamics, and an observer defined for this augmented system will estimate the disturbance ``d``. The controller can then be designed to reject it by using feedback from the estimated disturbance state. See the following tutorials making use of this approach
- [Disturbance modeling and rejection with LQG controllers](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/lqg_disturbance/)
- [Disturbance modeling and rejection with MPC controllers](@ref)
- [Feedforward from known disturbances](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/measurable_disturbance/)

This approach can easily be generalized to allow for more complex models of the disturbance, i.e., models of the form
```math
\begin{aligned}
\dot{x}_d &= A_d x_d + B_d w_d \\
d &= C_d x_d
\end{aligned}
```
This approach can be used to, e.g., reject periodic disturbances by modeling them as white noise entering a resonant system. An example of this is provided in the tutorial [MPC with model estimated from data](@ref). Also see functions [`add_disturbance`](@ref), [`add_low_frequency_disturbance`](@ref), [`add_resonant_disturbance`](@ref) to help with the plant augmentation.

To learn more about this approach, consult, e.g., chapter 4 in "Computer Controlled Systems" by Åström and Wittenmark.[^CCS]

[^CCS]: Åström, Karl J., and Björn Wittenmark. Computer-controlled systems: theory and design.

## Software tools
DyadControlSystems contains a large number of tools to assist with the addition of integral action to a controller.

### Linear systems
Standard PID controllers may be created using the [`pid`](@ref) function, the [`placePI`](@ref), [`loopshapingPI`](@ref), [`loopshapingPID`](@ref) functions, or tuned using [PID Autotuning](@ref).

State-feedback controllers may be augmented with integral action using the [`add_low_frequency_disturbance`](@ref) function, see
- [Disturbance modeling and rejection with LQG controllers](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/lqg_disturbance/)

for an example. Integral action may also be added using loop-shaping, see the following examples:
- [Mixed-sensitivity ``H_\infty`` control design](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/hinf_DC/)
- [Glover McFarlane design](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/#Example:-Glover-McFarlane-design)

### MPC controllers
When designing an MPC controller, we may use more or less the same methods as for standard linear state-feedback controllers (indeed, the MPC controller is commonly a nonlinear state-feedback controller). Tutorials covering integral action for MPC controllers include
- [Disturbance modeling and rejection with MPC controllers](@ref)
- [Control design for a quadruple-tank system with JuliaSim Control](@ref)
- [Adaptive MPC](@ref)
- [Robust MPC tuning using the Glover McFarlane method](@ref)
- [Mixed-sensitivity $\mathcal{H}_2$ design for MPC controllers](@ref)
- [Disturbance modeling in ModelingToolkit](@ref)

In addition to the tutorials above, one may also add explicit input integration to the controller by passing the keyword argument `input_integrators` when creating a [`FunctionSystem`](@ref). Only the [`GenericMPCProblem`](@ref) supports this option. For this option to lead to a controller with integral action, one must not penalize the control signal ``u``, and instead penalize the *control signal difference* ``\Delta u = u_k - u_{k-1}`` using, e.g., the [`DifferenceCost`](@ref) objective. An example follows below.


#### Example: Integral action with input integration
In this example, we design an MPC controller for a simple first-order linear system and use a desired reference ``x_r ≠ 0``, without further consideration, this will cause a stationary error unless we add integral action to the controller. We add integral action by adding explicit input-integration to the controller, and penalize the control signal difference ``\Delta u`` instead of the control signal ``u`` itself. To illustrate the difference, we show both cases, penalty on ``u`` alone, and penalty on ``\Delta u`` alone. To indicate that we want explicit input integration, we pass `input_integrators=1:1` when we create the `FunctionSystem` (`1:1` is a range with only the element 1, we could also have passed the vector `[1]`).
```@example INPUT_INTEGRATION
using DyadControlSystems, DyadControlSystems.MPC, Plots, LinearAlgebra
gr(fmt=:png) # hide

function linsys(x, u, _, _)
    Ad = [0.3679;;] # A stable first-order system in discrete time
    B = [0.6321;;]
    Ad*x + B*u
end

Ts = 1          # Sample time
x0 = [10.0]     # Initial state
xr = [-4.0]     # Reference state
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x, u=:u, y=:y, input_integrators=1:1) # Specify that the first (and only) input is to be integrated
observer = StateFeedback(dynamics, x0)

running_cost = StageCost() do si, p, t
    e = (si.x-si.r)[] # Only penalize state errors here
    dot(e, e) + p.u_penalty*dot(si.u, si.u)
end

difference_cost = DifferenceCost() do Δu, p, t
    p.Δu_penalty*dot(Δu, Δu) # Penalize control signal differences here
end

terminal_cost = TerminalCost() do ti, p, t
    e = (ti.x-ti.r)[1]
    10dot(e, e)
end

objective = Objective(running_cost, terminal_cost, difference_cost)
N = 10 # MPC horizon

pu = (u_penalty = true, Δu_penalty = false) # Initially penalize u directly

prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    p=pu,
    xr,
    verbose = false,
);
histu = MPC.solve(prob; x0, T=20, verbose = false, p=pu)
plot(histu, plot_title="MPC with input integration", lab="Penalty on \$u\$", c=1)

pΔu = (u_penalty = false, Δu_penalty = true) # Change to instead penalize Δu
hist = MPC.solve(prob; x0, T=20, verbose = false, p=pΔu)
plot!(hist, lab="Penalty on \$Δu\$", c=2)
```

We can easily understand why we must not penalize the control signal directly, indeed, if we penalize ``u^2``, we prefer ``u`` to be zero, but the stable system requires a non-zero ``u`` to have a stationary point at the reference. The stationary point when ``u`` is penalized will thus be a trade-off between getting a small control error ``e = r - x`` and a small control input ``u``. No one likes trade offs, so we penalize ``\Delta u`` instead.

```@example INPUT_INTEGRATION
using Test
X,E,R,U,Y,UE = reduce(hcat, hist)
@test abs(E[1, end]) < 1e-3
@test abs(U[1, end]) ≈ -xr[] atol=1e-3
```