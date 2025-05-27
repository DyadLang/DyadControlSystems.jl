# MPC with generic cost and constraints
Generic MPC, sometimes called *Economic MPC* or simply *Nonlinear MPC*, refers to Model-Predictive Control with an arbitrary cost function, i.e., not restricted to quadratic cost functions.

The MPC problem type [`GenericMPCProblem`](@ref) allows much more general cost functions and constraints than the quadratic-programming based MPC problem types, and instances of [`GenericMPCProblem`](@ref) are solved using Optimization.jl. The examples in this section will all use the IPOPT solver, a good general-purpose solver that supports large, sparse problems with nonlinear constraints.

The [`GenericMPCProblem`](@ref) also supports integer constraints, or *MINLP MPC* (mixed-integer nonlinear programming).

## Getting started
Below, we design an MPC controller for the nonlinear pendulum-on-a-cart system using the generic interface. Additional examples using this problem type are available in the tutorials
- [MPC control of a Continuously Stirred Tank Reactor (CSTR)](@ref)
- [Solving optimal-control problems](@ref optimal_control_example)
- [Solving optimal-control problems with MTK models](@ref optimal_control_mtk_example)
- [Model-Predictive Control for the Research Civil Aircraft system](@ref)
- [MPC with binary or integer variables (MINLP)](@ref)

The code for this example follows below, the code will be broken down in the following sections.
```@example MPC_generic
using DyadControlSystems, Plots
using DyadControlSystems.MPC
using DyadControlSystems.Symbolics
using StaticArrays
using LinearAlgebra
gr(fmt=:png) # hide

function cartpole(x, u, p, _=0)
    T = promote_type(eltype(x), eltype(u))
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81

    q  = x[SA[1, 2]]
    qd = x[SA[3, 4]]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp * g * l * s]
    B = @SVector [1, 0]
    if T <: Symbolics.Num
        qdd = Matrix(-H) \ Vector(C * qd + G - B * u[1])
        return [qd; qdd]
    else
        qdd = -H \ (C * qd + G - B * u[1])
        return [qd; qdd]
    end
end

nu = 1   # number of controls
nx = 4   # number of states
Ts = 0.1 # sample time
N  = 10  # MPC optimization horizon
x0 = ones(nx)  # Initial state
r  = zeros(nx) # Reference

discrete_dynamics = MPC.rk4(cartpole, Ts)   # Discretize the dynamics
measurement = (x,u,p,t) -> x                # The entire state is available for measurement
dynamics = FunctionSystem(discrete_dynamics, measurement, Ts; x=:x^nx, u=:u^nu, y=:y^nx)

# Create objective 
Q1 = Diagonal(@SVector ones(nx))    # state cost matrix
Q2 = 0.1Diagonal(@SVector ones(nu)) # control cost matrix
Q3 = Q2
QN, _ = MPC.calc_QN_AB(Q1, Q2, Q3, dynamics, r) # Compute terminal cost
QN = SMatrix{nx,nx}(QN)

p = (; Q1, Q2, Q3, QN) # Parameter vector

running_cost = StageCost() do si, p, t
    Q1, Q2 = p.Q1, p.Q2 # Access parameters from p
    e = (si.x)
    u = (si.u)
    dot(e, Q1, e) + dot(u, Q2, u)
end

difference_cost = DifferenceCost() do e, p, t
    dot(e, p.Q3, e)
end

terminal_cost = TerminalCost() do ti, p, t
    e = ti.x
    e'p.QN*e
end

objective = Objective(running_cost, terminal_cost, difference_cost)

# Create objective input
x = zeros(nx, N+1)
u = zeros(nu, N)
x, u = MPC.rollout(dynamics, x0, u, p, 0)
oi = ObjectiveInput(x, u, r)

# Create constraints
control_and_state_constraint = StageConstraint([-3, -4], [3, 4]) do si, p, t
    u = (si.u)[]
    x4 = (si.x)[4]
    SA[
        u
        x4
    ]
end

# Create observer, solver and problem
observer = StateFeedback(dynamics, x0)

solver = MPC.IpoptSolver(;
        verbose = false,
        tol = 1e-4,
        acceptable_tol = 1e-1, 
        max_iter = 100,
        max_cpu_time = 10.0,
        max_wall_time = 10.0,
        constr_viol_tol = 1e-4,
        acceptable_constr_viol_tol = 1e-1,
        acceptable_iter = 2,
    )

prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [control_and_state_constraint],
    p,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
);

# Run MPC controller
history = MPC.solve(prob; x0, T = 100, verbose = false)

# Extract matrices
X,E,R,U,Y = reduce(hcat, history)

plot(history)
```


## Specifying cost and constraints
The [`GenericMPCProblem`](@ref) requires the specification of an [`Objective`](@ref), which internally contains one or many cost functions, such as
- [`StageCost`](@ref)
- [`DifferenceCost`](@ref)
- [`TerminalCost`](@ref)
- [`TrajectoryCost`](@ref)

Each cost object takes a function as its first argument that computes the cost based on the relevant optimization variables. We illustrate with an example where we create a stage cost that computes $x(t)^T Q_1 x(t) + u(t)^T Q_2 u(t)$:
```julia
running_cost = StageCost() do si, p, t
    x = si.x
    u = si.u
    dot(x, Q1, x) + dot(u, Q2, u)
end
```

This uses the Julia `do`-syntax to create an anonymous function that takes the tuple `(si, p, t)`. `si` is of type [`MPC.StageInput`](@ref), a structure containing vectors `x, u, r`, all at the stage time `t`. A *stage* refers to a single instant in time in the optimization horizon. While most cost and constraint types passes a [`MPC.StageInput`](@ref) as the first argument to the cost/constraint function, the [`TrajectoryCost`](@ref) and [`TrajectoryConstraint`](@ref) are passed an [`MPC.ObjectiveInput`](@ref) which contains the entire trajectories $x \in \mathbb{R}^{n_x \times N+1}$ and $u \in \mathbb{R}^{n_u \times N}$.

One or many cost functions are finally packaged into an [`Objective`](@ref), illustrated in the comprehensive example below.


Constraints are similarly defined to take a function that computes the constrained output as first argument. We illustrate with an example that creates control and state constraints corresponding to
```math
\begin{aligned}
-3 &\leq u_1(t) \leq 3 \\
-4 &\leq x_2(t) + x_4(t) \leq 4
\end{aligned}
```

```julia
using StaticArrays
control_and_state_constraint = StageConstraint([-3, -4], [3, 4]) do si, p, t
    u = si.u[1]
    x2 = si.x[2]
    x4 = si.x[4]
    SA[
        u
        x2 + x4
    ]
end
```
The full signature of [`StageConstraint`](@ref) is
```julia
StageConstraint(fun, lb, ub)
```
where `fun` is a function from `(stage_input, parameters, time)` to constrained output and `lb, ub` are the lower and upper bounds of the constrained output. In this case, our constrained output is a static array (for high performance) containing `[u[1], x[2]+x[4]]`, which are the expressions we wanted to constrain, i.e., the *constrained output*.

If simple bounds on states and control inputs are desired, the function [`BoundsConstraint`](@ref) can be used instead.

The available constraint types are
- [`BoundsConstraint`](@ref)
- [`StageConstraint`](@ref)
- [`TerminalStateConstraint`](@ref)
- [`TrajectoryConstraint`](@ref)
- Integer and binary constraints are handled using the `int_x` and `int_u` keywords to [`GenericMPCProblem`](@ref), see [MPC with binary or integer variables (MINLP)](@ref) for an example.

## Specifying the discretization
The [`GenericMPCProblem`](@ref) allows the user to select the discretization method used in the transcription from continuous to discrete time. The available choices are
- [`MultipleShooting`](@ref) (the default if none is chosen)
- [`CollocationFinE`](@ref)
- [`Trapezoidal`](@ref)

More details on these choices are available under [Discretization](@ref).





