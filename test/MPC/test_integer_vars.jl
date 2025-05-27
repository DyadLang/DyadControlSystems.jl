using DyadControlSystems, Plots
using DyadControlSystems.MPC
using StaticArrays
using LinearAlgebra

function simple_dynamics(x, u, p, t)
    Ts = 0.1
    A = SA[1 Ts; 0 1]
    B = SA[0; Ts;;]
    return A * x + B * u
end

nu = 1              # number of controls
nx = 2              # number of states
Ts = 0.1            # sample time
N  = 10             # MPC optimization horizon
x0 = [10.0, 0.0]    # Initial state
r  = zeros(nx)

measurement = (x,u,p,t) -> x                # The entire state is available for measurement
discrete_dynamics = FunctionSystem(simple_dynamics, measurement, Ts; x=[:p, :v], u=:u, y=:y^nx)


Q1 = Diagonal(@SVector ones(nx))    # state cost matrix
p = (; Q1, q2 = 0.1)

running_cost = StageCost() do si, p, t
    Q1, q2 = p
    e = (si.x)
    u = (si.u)
    dot(e, Q1, e) +
    q2*u[1]^2
end

terminal_cost = TerminalCost() do ti, p, t
    e = ti.x
    dot(e, 10p.Q1, e)
end

objective = Objective(running_cost, terminal_cost)

x = zeros(nx, N+1)
u = zeros(nu, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p, 0)
oi = ObjectiveInput(x, u, r)


bounds_constraint = BoundsConstraint(umin = [-1], umax = [1], xmin = [-Inf, -Inf], xmax = [Inf, Inf])
observer = StateFeedback(discrete_dynamics, x0)

inner_solver = ()->MPC.IpoptSolver(;
        verbose = false,
        tol = 1e-4,
        acceptable_tol = 1e-3,
        max_iter = 200,
        max_cpu_time = 10.0,
        max_wall_time = 10.0,
        constr_viol_tol = 1e-4,
        acceptable_constr_viol_tol = 1e-3,
        acceptable_iter = 2,
)

using Juniper, OptimizationMOI
const MOI = OptimizationMOI.MOI
solver = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer, "nl_solver" => inner_solver, "allow_almost_solved"=>true, "atol"=>1e-3, "mip_gap" => 1e-3, "log_levels"=>[])

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
    verbose = isinteractive(),
    int_u = [true],
    Nint = 2,
);

# Run MPC controller
@time history = MPC.solve(prob; x0, T = 100)
isinteractive() && plot(history, seriestype=:steppre)

u = reduce(hcat, history.U)[:]
@test maximum(minimum(abs.(u .- [-1, 0, 1]'), dims=2)) < 1e-3 # Test that control input is integer
@test norm(history.X[end]) < 0.05