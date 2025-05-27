# [Solving optimal-control problems](@id optimal_control_example)

## Pendulum swing-up
In this example, we will solve an open-loop optimal-control problem (sometimes called trajectory optimization). The problem we will consider is to swing up a pendulum attached to a cart. A very similar tutorial that is using ModelingToolkit to build the model is available here [Optimal control using ModelingToolkit models](@ref optimal_control_mtk_example).

We start by defining the dynamics:
```@example OPTCONTROL
using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.Symbolics
using LinearAlgebra
using StaticArrays

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

nu = 1           # number of controls
nx = 4           # number of states
Ts = 0.04        # sample time
N = 80           # Optimization horizon (number of time steps)
x0 = zeros(nx)   # Initial state
r = [0, π, 0, 0] # Reference state is given by the pendulum upright (π rad)

discrete_dynamics = MPC.rk4(cartpole, Ts)   # Discretize the dynamics
measurement = (x,u,p,t) -> x                # The entire state is available for measurement
dynamics = FunctionSystem(discrete_dynamics, measurement, Ts; x=[:p, :α, :v, :ω], u=:u^nu, y=:y^nx)
nothing # hide
```

The next step is to define a cost function and constraints. We will use a quadratic [`StageCost`](@ref) and a [`TerminalStateConstraint`](@ref) to force the solution to end up in the desired terminal state: a stationary upright pendulum and a stationary cart in the origin. We also constraints the control input to have magnitude less than 10.
```@example OPTCONTROL
# Create objective 
Q1 = Diagonal(@SVector ones(nx))     # state cost matrix
Q2 = 0.01Diagonal(@SVector ones(nu)) # control cost matrix

p = (; Q1, Q2) # Parameter vector

running_cost = StageCost() do si, p, t
    Q1, Q2 = p.Q1, p.Q2
    e = si.x .- si.r
    u = si.u
    dot(e, Q1, e) + dot(u, Q2, u)
end

terminal_constraint = TerminalStateConstraint(r, r) do ti, p, t
    ti.x
end

objective = Objective(running_cost)

# Create objective input
x = zeros(nx, N+1)
u = zeros(nu, N)
oi = ObjectiveInput(x, u, r)

# Create constraints
control_constraint = StageConstraint([-10], [10]) do si, p, t
    u = (si.u)[]
    SA[
        u
    ]
end
nothing # hide
```
We will use the [`GenericMPCProblem`](@ref) structure to create the optimal-control problem. As the name implies, this technically defines an MPC problem, but an MPC problem is nothing more than a repeatedly solved optimal-control problem! When we create the problem, we pass `presolve = true` to have the problem solved immediately in the constructor.
```@example OPTCONTROL
# Create observer solver and problem
observer = StateFeedback(dynamics, x0)

solver = IpoptSolver(;
        verbose                    = false,
        tol                        = 1e-4,
        acceptable_tol             = 1e-2,
        max_iter                   = 2000,
        max_cpu_time               = 60.0,
        max_wall_time              = 60.0,
        constr_viol_tol            = 1e-4,
        acceptable_constr_viol_tol = 1e-2,
        acceptable_iter            = 100,
)

prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [control_constraint, terminal_constraint],
    p,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
);
nothing # hide
```
With the problem solved, we may extract the optimal trajectories an plot them
```@example OPTCONTROL
using Plots
gr(fmt=:png) # hide
x_sol, u_sol = get_xu(prob)
fig = plot(
    plot(x_sol', title="States", lab=permutedims(state_names(dynamics))),
    plot(u_sol', title="Control signal", lab=permutedims(input_names(dynamics))),
)
hline!([π], ls=:dash, c=2, sp=1, lab="α = π")
fig
```


## Rocket launch control
This example follows that of [Optimal rocket control with JuMP](https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/rocket_control/), and highlights the differences in interface between DyadControlSystems and JuMP.jl, which is a general-purpose modeling language for optimization.

The control problem to solve is to optimize a thrust trajectory for a rocket that aims at maximizing the achieved altitude. The model of the rocket is a simple three-state model where we have the height $h$, the velocity $v$ and the mass $m$ as states. Due to the burning of fuel, the mass decreases during launch.

We start by defining the dynamics, all constants are normalized to be unitless:
```@example ROCKET
using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.Symbolics
using LinearAlgebra
using StaticArrays

const h_0 = 1    # Initial height
const v_0 = 0    # Initial velocity
const m_0 = 1    # Initial mass
const g_0 = 1    # Gravity at the surface

const T_c = 3.5  # Used for thrust
const m_c = 0.6  # Fraction of initial mass left at end

const m_f = m_c * m_0              # Final mass
const T_max = T_c * g_0 * m_0      # Maximum thrust


function rocket(x, u, p, _=0)
    h_c = 500                    # Used for drag
    v_c = 620                    # Used for drag
    c = 0.5 * sqrt(g_0 * h_0)    # Thrust-to-fuel mass
    D_c = 0.5 * v_c * m_0 / g_0  # Drag scaling

    h, v, m = x
    T = u[]                      # Thrust (control signal)
    drag =  D_c * v^2 * exp(-h_c * (h - h_0) / h_0)
    grav = g_0 * (h_0 / h)^2
    SA[
        v
        (T - drag - m * grav) / m 
        -T/c
    ]
end

nu = 1            # number of control inputs
nx = 3            # number of states
N  = 200          # Optimization horizon (number of time steps)
Ts = 0.001        # sample time
x0 = Float64[h_0, v_0, m_0]   # Initial state
r = zeros(nx)

measurement = (x,u,p,t) -> x                # The entire state is available for measurement
dynamics = FunctionSystem(rocket, measurement; x=[:h, :v, :m], u=:T, y=:y^nx)
discrete_dynamics = MPC.rk4(dynamics, Ts; supersample=3)
nothing # hide
```

Next, we define constraints on the states and inputs. 
```@example ROCKET

lb = [h_0, v_0, m_f, 0]
ub = [Inf, Inf, m_0, T_max]

stage_constraint = StageConstraint(lb, ub) do si, p, t
    u = (si.u)[]
    h,v,m = si.x
    SA[h, v, m, u]
end

terminal_constraint = TerminalStateConstraint([m_f], [m_f]) do ti, p, t
    SA[ti.x[3]] # The final mass must be m_f
end

terminal_cost = TerminalCost() do ti, p, t
    h = ti.x[1]
    -h # Maximize the terminal altitude
end

objective = Objective(terminal_cost)
nothing # hide
```

We also define an initial guess, we create an input-signal trajectory that renders the initial state rollout feasible.
```@example ROCKET
using Plots
u = [0.7T_max * ones(nu, N÷5)  T_max / 5 * ones(nu, 4N÷5) ]

x, u = MPC.rollout(discrete_dynamics, x0, u, 0, 0)
oi = ObjectiveInput(x, u, r)
plot(x', layout=3)
```

We are now ready to create to optimal-control problem:
```@example ROCKET
observer = StateFeedback(discrete_dynamics, x0)

solver = IpoptSolver(;
        verbose                    = true,
        tol                        = 1e-8,
        acceptable_tol             = 1e-5,
        constr_viol_tol            = 1e-8,
        acceptable_constr_viol_tol = 1e-5,
        acceptable_iter            = 10,
)

prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    Ts,
    objective,
    solver,
    constraints     = [stage_constraint, terminal_constraint],
    objective_input = oi,
    xr              = r,
    presolve        = true,
    verbose         = false,
    disc  = Trapezoidal(; dyn=dynamics),
)
nothing # hide
```
When the problem is solved, we may plot the optimal trajectory
```@example ROCKET
using Plots
x_sol, u_sol = get_xu(prob)
plot(
    plot(x_sol', title="States",         lab=permutedims(state_names(dynamics)), layout=(nx,1)),
    plot(u_sol', title="Control signal", lab=permutedims(input_names(dynamics))),
)
```

```@example ROCKET
using Test
@test x_sol[1, end] > 1.012 # Test that the rocket reached high enough
```