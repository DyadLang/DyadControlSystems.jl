using DyadControlSystems, Plots
using DyadControlSystems.MPC
using StaticArrays
using LinearAlgebra

huber(x, a=0.1) = ifelse(abs(x) < a, x^2 / 2a, (abs(x) - a / 2))

function dynamics(x, u, p, t)
    Ts = 0.1
    A = SA[1 Ts; 0 1]
    B = SA[0 0; Ts Ts]
    return A * x + B * u
end

nu = 2 # number of controls
nx = 2 # number of states
Ts = 0.1 # sample time
N  = 10 # MPC optimization horizon
x0 = ones(nx) # Initial state
r  = zeros(nx)

measurement = (x,u,p,t) -> x                # The entire state is available for measurement
discrete_dynamics = FunctionSystem(dynamics, measurement, Ts; x=:x^nx, u=:u^nu, y=:y^nx)
# discrete_dynamics = MPC.rk4(cont_dynamics, Ts)   # Discretize the dynamics

# Create objective
Q = Diagonal(@SVector ones(nx))    # state cost matrix
p = (; q1 = 0.1, q2 = 1.0)

running_cost = StageCost() do si, p, t
    q1, q2 = p
    e = (si.x)
    u = (si.u)
    dot(e, Q, e) + 
    q1*u[1]^2 + q1*u[2]^2 + 
    q2*huber(u[2], 0.001)
end

terminal_cost = TerminalCost() do ti, p, t
    e = ti.x
    dot(e, 10Q, e)
end

objective = Objective(running_cost, terminal_cost)

# Create objective input
x = zeros(nx, N+1)
u = zeros(nu, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p, 0)
oi = ObjectiveInput(x, u, r)

# Create constraints
bounds_constraint = BoundsConstraint(umin = [-1, -1], umax = [1, 1], xmin = [-Inf, -Inf], xmax = [Inf, Inf])
observer = StateFeedback(discrete_dynamics, x0)

solver = MPC.IpoptSolver(;
        verbose = false,
        tol = 1e-4,
        acceptable_tol = 1e-3,
        max_iter = 200,
        max_cpu_time = 10.0,
        max_wall_time = 10.0,
        constr_viol_tol = 1e-4,
        acceptable_constr_viol_tol = 1e-3,
        acceptable_iter = 2,
        # exact_hessian = false,
    )

prob = GenericMPCProblem(
    discrete_dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
    verbose = true,
);

x0 = [10.0, 0.0]
# Run MPC controller
history = MPC.solve(prob; x0, T = 100, verbose = true)

# Extract matrices
X,E,R,U,Y = reduce(hcat, history)

plot(history, seriestype=:steppre) |> display


##
p = (; q1 = 0.1, q2 = 0.1) # Choose a smaller q2

history = MPC.solve(prob; x0, T = 100, verbose = true, p)
plot(history, seriestype=:steppre)