# MPC with quadratic cost functions
This section documents the quadratic-cost MPC formulations implemented by [`LQMPCProblem`](@ref) and [`QMPCProblem`](@ref). These problems are solved using quadratic programming in the linear case, and sequential quadratic programming (SQP) in the nonlinear case. These problems support linear/nonlinear dynamics, quadratic cost functions and bound constraints on control and state variables. For more advanced MPC formulations, see the section on [MPC with generic cost and constraints](@ref).



A video tutorial on quadratic MPC is available in the form of a webinar:
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/djQcM7KiB3M" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```


## Getting started: Nonlinear system
We begin the exposition of the MPC functionality with some examples, and provide details in the sections below. 

In this example, we will design an MPC controller for a pendulum on a cart. We start by defining the dynamics (in this case hand-written, see [`build_controlled_dynamics`](@ref) to generate a suitable function from a ModelingToolkit model). The states are
```math
(p, \theta, v, \omega)
```
corresponding to position, angle, velocity and angular velocity.
We also define some problem parameters.
```@example MPC
using DyadControlSystems
using DyadControlSystems.MPC, StaticArrays

# Dynamics function in continuous time (x,u,p,t) = (states, control inputs, parameters, time)
function cartpole(x, u, p, _)
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81

    q  = x[SA[1, 2]]
    qd = x[SA[3, 4]]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp * g * l * s]
    B = @SVector [1, 0]

    qdd = -H \ (C * qd + G - B * u[1])
    return [qd; qdd]
end

nu = 1    # number of control inputs
nx = 4    # number of states
ny = nx   # number of outputs (here we assume that all states are measurable)
Ts = 0.02 # sample time
N = 15    # MPC optimization horizon (control horizon is also equal to N)

x0 = zeros(nx) # Initial state
x0[1] = 3      # cart pos
x0[2] = pi*0.5 # pendulum angle

xr = zeros(nx) # reference state
nothing # hide
```
Next, we discretize the dynamics and specify the cost-function matrices
```@example MPC
discrete_dynamics0 = rk4(cartpole, Ts) # discretize using Runge-Kutta 4
measurement = (x,u,p,t)->x # Define the measurement function, in this case we measure the full state
state_names = [:p, :θ, :v, :ω]
discrete_dynamics = FunctionSystem(discrete_dynamics0, measurement, Ts, x=state_names, u=:u, y=state_names) # Add a 
Q1 = spdiagm(ones(nx))      # state cost matrix
Q2 = Ts * spdiagm(ones(nu)) # control cost matrix (penalty on control derivative is also supported)

# Control limits
umin = -5 * ones(nu)
umax = 5 * ones(nu)

# State limits (state constraints are soft by default if enabled)
xmin = nothing # no constraints on states
xmax = nothing
constraints = NonlinearMPCConstraints(; umin, umax, xmin, xmax)
```
```@example MPC
## Define problem structure
observer = MPC.StateFeedback(discrete_dynamics, x0)
solver = OSQPSolver(
    eps_rel  = 1e-3,
    max_iter = 500,        # in the QP solver
    check_termination = 5, # how often the QP solver checks termination criteria
    sqp_iters = 2,         # How many SQP iterations are taken per time step
)
nothing # hide
```
This example will use the [`QMPCProblem`](@ref) structure since our dynamics is nonlinear, but our cost is quadratic.
```@example MPC
prob = QMPCProblem(
    discrete_dynamics;
    observer, 
    Q1,
    Q2,
    constraints,
    N,
    xr,
    solver,
)
nothing # hide
```
With the problem defined, we could create a `DiscreteSystem` representing the MPC controller. This is currently work in progress and is waiting on the [these issues](https://github.com/SciML/ModelingToolkit.jl/issues?q=is%3Aopen+is%3Aissue+label%3Adiscrete-time) to be solved. Until then, we may run the MPC controller manually using
```@example MPC
history = MPC.solve(prob; x0, T = 500, verbose = false) # solve for T=500 time steps
nothing # hide
```
```@example MPC
using Plots
gr(fmt=:png) # hide
plot(history)
```

## Getting started: Linear system
Instead of passing a (generally) nonlinear function `dynamics` to the constructor [`QMPCProblem`](@ref), you may pass a linear system on statespace form to the constructor [`LQMPCProblem`](@ref). The LQMPC formulation acting on a linear system is equivalent to an LQG controller if no constraints are active, but allows for more principled handling of constraints than what using standard linear methods does. The MPC controller can also incorporate reference pre-view, achievable only by acausal pre-filtering of references for standard controllers.

This example illustrates the MPC controller controlling a simple discrete-time double integrator with sample time 1. We also illustrate the use of tight state constraints, here, the state reference goes from 0 to 1, at the same time, we constrain the system to have state components no greater than 1 to prevent overshoot etc. Being able to operate close to constraints is one of many appealing properties of an MPC controller. (In practice, noise makes having the state reference exactly equal to the constraint a somewhat questionable practice).

```@example MPC
using DyadControlSystems
using DyadControlSystems.MPC
using LinearAlgebra # For Diagonal
Ts = 1
sys = ss([1 Ts; 0 1], [0; 1;;], [1 0], 0, Ts) # Create a statespace object

(; nx,ny,nu) = sys
N = 20
x0 = [1.0, 0]
xr = zeros(nx, N+1) # reference trajectory (should have at least length N+1)
xr[1, 20:end] .= 1  # Change reference after 20 time steps
ur = zeros(nu, N)   # Control signal at operating point (should have at least length N)
yr = sys.C*xr

op = OperatingPoint() # Empty operating point implies x = u = y = 0

# Control limits
umin = -0.07 * ones(nu)
umax = 0.07 * ones(nu)

# State limits (state constraints are soft by default)
xmin = -0.2 * ones(nx)
xmax = 1 * ones(nx)

constraints = MPCConstraints(; umin, umax, xmin, xmax)

solver = OSQPSolver(
    verbose = false,
    eps_rel = 1e-6,
    max_iter = 1500,
    check_termination = 5,
    polish = true,
)

Q2 = spdiagm(ones(nu)) # control cost matrix
Q1 = Diagonal([10, 1]) # state cost matrix

T = 40 # Simulation length (time steps)
nothing # hide
```

The next step is to define a state observer and a prediction model, in this example, we will use a Kalman filter:
```@example MPC
R1 = 1.0I(nx)
R2 = 1.0I(ny)
kf = KalmanFilter(ssdata(sys)..., R1, R2)
named_sys = named_ss(sys, x=[:pos, :vel], u=:force, y=:pos) # give names to signals for nicer plot labels
predmodel = LinearMPCModel(named_sys, kf; constraints, op, x0)

prob = LQMPCProblem(predmodel; Q1, Q2, N, solver, r=xr)

hist = MPC.solve(prob; x0, T, verbose = false)
plot(hist); hline!(xmin[2:2], lab="Constraint", l=(:black, :dash), sp=1)
```
Notice how the initial step response never goes below -0.2, and the second step response has very little overshoot due to the state constraint. If you run the code, try increasing the state constraint and see how the step response changes

We also notice that before the change in reference at ``T \approx 20``, the response appears to make a slight move in the wrong direction. This behavior is actually expected, and arises due to the penalty and constraint on the control action. The system we are controlling is a simple double integrator, ``f = ma``, and the control signal thus corresponds to a force which is proportional to the acceleration. If it's expensive to perform large accelerations, it might be beneficial to accept a small state error (small errors are penalized very little under a quadratic cost) before the step in order to build momentum for the large step in reference and avoid larger errors after the step has occurred.


### Output references and constraints
To use references for controlled outputs rather than for states, make use of the keyword arguments `z` and `v` to [`LinearMPCModel`](@ref) to specify the controlled and constrained outputs.

See [MPC with model estimated from data](@ref) for an example where this functionality is used.


## Constraints
For linear MPC formulations, we provide the simple structure [`MPCConstraints`](@ref) that allows you to specify bounds on state variables and control inputs. We also provide the more general [`LinearMPCConstraints`](@ref) that allows you to define bounds on any linear combination of states and control inputs, such a linear combination is referred to as a *constrained output* $v$:
$v = C_v x + D_v u$. When the simple constructor is used, `xmin, xmax, umin, umax` are internally converted to `vmin, vmax`.

For nonlinear MPC problems, we have [`NonlinearMPCConstraints`](@ref) that accepts a function
$v = f(x,u,p,t)$ that computes the constrained outputs. This structure also requires you to manually specify which components of $v$ are soft constraints.

Constraints are by default
- Hard for control inputs.
- Soft for states.

Soft state constraints should be favored in practical applications and in simulations with disturbances and noise. They are, however, significantly more computationally expensive than hard constraints, and in some simulation scenarios it may be favorable to select hard state constraints for faster simulation. The parameters `qs` and `qs2` determine the weights on the slack variables for the soft constraints, `qs` weighs the absolute value of the slack and `qs2` weighs the square of the slack. Hard constraints are used if `qs = qs2 = 0`. In a practical application, using hard constraints is not recommended, a disturbance acting on the system might bring the state outside the feasible region from which there is no feasible trajectory back into the feasible region, leading to a failure of the optimizer.

By using only `qs2`, i.e. a quadratic penalty on the slack variables, a small violation is penalized lightly, and the problem remains easy to solve. Using only `qs` (with a sufficiently large value), constraint violation is kept zero if possible, violating constraints only if it is impossible to satisfy them due to, e.g., hard constraints on control inputs. A combination of `qs` and `qs2` can be used to tweak performance.

If `xmin, xmax` are not provided, state constraints are disabled.



## Internals of the quadratic MPC formulations
The internal representation of the MPC problem may be useful in order to interface to general-purpose QP/SQP solvers etc. This section provides useful details regarding the problem layout as well as the exact representation of soft constraints and other low level details.

The MPC problem is formulated as a QP problem with
```math
\operatorname{minimize}_{x,u} (z_{N+1} - r)^T Q_N (z_{N+1} - r) + 
\sum_{n=1}^N (z_n - r)^T Q_1 (z_n - r) + u_n^T Q_2 u_n + 
\Delta u_n^T Q_3 \Delta u_n
```
where $\Delta u_n = u_n - u_{n-1}$.
```math
\operatorname{subject~to}\\
x_{n+1} = Ax_n + Bu_n\\
v_{min} \leq v \leq v_{max}\\
v_n = C_v x + D_v u \\
z_n = C_z x\\
```
and $x_1$ is constrained to be equal to the MPC controller input and $Q_N$ is given by the solution to the discrete-time [algebraic Riccati equation](https://en.wikipedia.org/wiki/Algebraic_Riccati_equation).

In the formulation above, $r$ is constant, but $r$ is allowed to be of length $n_z \times N+1$ as well.

Hard state constraints is known to be problematic in MPC applications, since disturbances might cause the initial state of the problem to become infeasible. The solution used to solve this problem is to make the state constraints *soft* by means of slack variables. The soft inequality constraints are formulated as 

```math
v \leq v_{max} + s_u\\
v \geq v_{min} - s_l\\
s_u \geq 0 \\
s_l \geq 0 
```
where $s_u$ and $s_l$ are the upper and lower slack variables. To ensure that the slack variables remain zero unless necessary, the penalty term
$$q_s \left(||s_u||_1 + ||s_l||_1\right) + q_{s2} \left(||s_u||_2^2 + ||s_l||_2^2\right)$$
is added to the cost function. Since the slack variables are all non-negative, one-norm part of this term is linear, and equivalent to

$$q_s\mathbf{1} ^T s_u + q_s\mathbf{1} ^T s_l$$


### State constraints
Constraints on the state components (`xmin, xmax`) are by default "soft", in the sense that violations are penalized rather than enforced. The penalty weights can be controlled by the keyword arguments `qs, qs2` to [`LQMPCProblem`](@ref), where `qs` penalizes a linear term corresponding to the one-norm of the constraint violation, while `qs2` penalizes a quadratic term. By default, a combination of quadratic and linear penalty is used. Having a large linear penalty promotes tighter constraint enforcement, but makes the optimization problem less smooth and harder to solve.


### Output references
By default, references are expected for all state components. However, it is possible to use *output references* instead, where the outputs can be selected as any linear combination of the states $z = C_z x$.  The matrix $C_z$ can be chosen by either specifying the indices of the controlled variables `LinearMPCModel(...; z = state_indices)` or directly by `LinearMPCModel(...; z = Cz)`.

### Operating points
For linear MPC controllers, the type [`OperatingPoint`](@ref)`(x,u,y)` keeps track of state, input and output values around which a system is linearized. Nonlinear MPC formulations do not require this, but linear MPC controllers will generally perform better if the linearization point is provided to the constructors of [`LinearMPCProblem`](@ref) and [`RobustMPCProblem`](@ref).
Please note, if using nonlinear observers together with linear prediction models, an [`OperatingPointWrapper`](@ref) is required to ensure that the nonlinear observer operates in the original coordinates of the nonlinear system.

Given an `op::OperatingPoint`, `DyadControlSystems.linearize(system, op, p, t)` is shorthand to linearize the system around the operating point.

The [`GenericMPCProblem`](@ref) does not handle operating points due to the nonlinear applications of this problem type.

### Specialization for linear MPC
For linear MPC problems, further control over the constraints are allowed, the constraints are then posed on the *constrained output variable* $v$, defined as
```math
\operatorname{subject~to}\\
v_{min} - S s_l \leq C_v x + D_v u \leq v_{max} + S s_u\\
s_u \geq 0 \\
s_l \geq 0 
```
Where $S$ is used to be able to mix hard and soft constraints, i.e., $S$ is a matrix with the same row-size as $v$ and a column-size equal to the number of soft constraints. 

In the presence of output matrices $C_v, D_v$, the constraint equations change to
```math
C_v x + D_v u \leq v_{max} + S s_u\\
C_v x + D_v u \geq v_{min} - S s_l\\
```

Note, if loop-shaping weights $W_1$ and $W_2$ are in use, the constraint equations have been transformed from what the user originally provided.

### Problem layout
The variables are ordered
```math
\mathbf{x} = x_1,\; u_1,\; s^u_1, \; s^l_1 ...,\;  x_N,\; u_N,\; s^u_N, \; s^l_N, \; x_{N+1},
```

All equality constraints are encoded in the form (notation borrowed from the [solver documentation](https://osqp.org/docs/solver/index.html))
```math
l \leq \mathbf{Ax} \leq u
```
where $\mathbf{x}$ contains *all* variables and $\mathbf{A}$ is called the constraint matrix. Factorizing $\mathbf{A}$ is an expensive step of the solution to the MPC problem.

The cost function is encoded in the form
```math
\dfrac{1}{2} \mathbf{C_z x}^T P \mathbf{x} + q^T \mathbf{C_z x}
```


### Constraint updating
#### Linear MPC
For linear systems, the only constraint that must be updated each iteration is the constraint corresponding to the initial state, and if a $Q_3$ term is used, a similar constraint corresponding to $u_0$ (the previous control action).


#### Nonlinear MPC (NMPC)
When sequential quadratic programming (SQP) iterations are used, the nonlinear dynamics are linearized around a trajectory each SQP-iteration. This linearization leads to updated $A$ and $B$ matrices in the constraint $x_{n+1} = Ax_n + Bu_n$, and consequently an update and refactorization of the constraint matrix $\mathbf{A}$. This causes NMPC to be significantly more expensive than linear MPC. The number of SQP iterations is a user parameter and, due to the factorization dominating the computational time, often causes a linear increase in the computational time. Typically, a low number of SQP iterations is enough since the solution at the previous time step provides a very good initial guess for the optimization problem.
