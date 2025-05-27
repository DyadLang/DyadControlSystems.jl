using HardwareAbstractions
import HardwareAbstractions as hw
using QuanserInterface
using DyadControlSystems
using DyadControlSystems.MPC
using LinearAlgebra
import DyadControlSystems.ControlDemoSystems as demo
using StaticArrays
using Plots
cd(@__DIR__)


nu  = 1 # number of controls
nx  = 4 # number of states

function plotD(D, th=0.2)
    tvec = D[1, :]
    y = D[2:3, :]'
    # y[:, 2] .-= pi
    # y[:, 2] .*= -1
    xh = D[4:7, :]'
    u = D[8, :]
    plot(tvec, xh, layout=6, lab=permutedims(state_names(dynamics)), framestyle=:zerolines)
    plot!(tvec, y, sp=[1 2], lab = permutedims(output_names(dynamics)) .* "_meas", framestyle=:zerolines)
    hline!([-pi pi], lab="", sp=2)
    hline!([-pi-th -pi+th pi-th pi+th], lab="", l=(:black, :dash), sp=2)
    plot!(tvec, centraldiff(y) ./ median(diff(tvec)), sp=[3 4], lab="central diff")
    plot!(tvec, u, sp=5, lab = permutedims(input_names(dynamics)), framestyle=:zerolines)
    plot!(diff(D[1,:]), sp=6, lab="Δt"); hline!([process.Ts], sp=6, framestyle=:zerolines, lab="Ts")
end

using DelimitedFiles
function saveD(filename, D)
    writedlm(filename, [["t" "arm_angle" "pend_angle" "x1h" "x2h" "x3h" "x4h" "control_input"]; D'], ',')
end

# ==============================================================================
## Model validation
# ==============================================================================
# sys1 = QuanserInterface.linearized_pendulum()
# psim = QubeServoPendulumSimulator()

# xlin = Float64[0, π, 0, 0] # π stable, 0 unstable
# Afu, Bfu = DyadControlSystems.linearize(psim.dynamics, xlin, [0], psim.p, 0)
# Cfu, Dfu = DyadControlSystems.linearize(psim.measurement, xlin, [0], psim.p, 0)
# sys2 = ss(Afu, Bfu, Cfu, Dfu)

# w = exp10.(LinRange(-2, 3, 300))
# bodeplot(sys1, w, lab="Quanser lin", hz=true)
# bodeplot!(sys2, w, lab="Our lin", hz=true)


# ==============================================================================
## Collect identification data
# ==============================================================================
# using DelimitedFiles
# cd(@__DIR__)
# p = QubeServoPendulum(; Ts = 0.01)
# y = HardwareAbstractions.collect_data(p; Tf=14)
# Y = reduce(hcat, y)'
# writedlm("pendulum_freefall.csv", [["t" "arm_angle" "pendulum_angle"]; Y], ',')

# ==============================================================================
## Load identification data
# ==============================================================================
using DelimitedFiles
cd(@__DIR__)
Y, head = readdlm("pendulum_freefall.csv", ',', header=true)
tvec = Y[:,1]
y = [y[2:end] .+ [0, 0] for y in eachrow(Y)]
u = fill([0], length(tvec))

# ==============================================================================
## Observer filtering
# ==============================================================================
using LowLevelParticleFilters
import LowLevelParticleFilters as llpf
using Distributions
Ts = median(diff(tvec))
psim = QubeServoPendulumSimulator(; Ts)
ny, nu = 2, 1 
R1 = kron(LowLevelParticleFilters.double_integrator_covariance(Ts, 1000), diagm([10, 1])) + 1e-9I
R2 = 2pi/2048 * I(2)
x0 = SA[0.0, 0.99pi, 0, 0.0]
psim.x = x0
R10 = copy(10R1)
R10[1,1] *= 1000000 # Not sure about the initial arm angle
d0 = MvNormal(x0, R10)
# ekf = ExtendedKalmanFilter(psim.ddyn, psim.measurement, R1, R2, d0; nu, psim.p) # EKF suffers from exploding covariance, not sure why. Could be that sampled covariance matrix is wrong due to the discretization of the covariance matrix is wrong if we're far from the linearizaiton point? UKF works well
kf = UnscentedKalmanFilter(psim.ddyn, psim.measurement, R1, R2, d0; ny, nu, psim.p)
#
Ndata = length(y)
fsol = forward_trajectory(kf, u[1:Ndata], y[1:Ndata], llpf.parameters(kf))
# plot(fsol, plotu=false, ploty=false)
# plot!(Y[:, 2:end], sp=[1 2])
# plot!(centraldiff(Y[:, 2:3]) ./ median(diff(Y[:, 1])), sp=[3 4])
# This plot looks bad unless we reduce damping by 10x, but if we do, then we don't match their linearization, for which it appears easy to make a good controller
# ==============================================================================
## MPC
##
# ==============================================================================
Ts  = 0.01 # sample time MPC
Ts_fast = 0.01
psim = QubeServoPendulumSimulator(; Ts)
x0 = SA[0, 0.01, 0.0, 0]
psim = QubeServoPendulumSimulator(; Ts=Ts_fast) # The discrete dynamics here is used by the UKF
ny, nu = 2, 1 
R1 = kron(LowLevelParticleFilters.double_integrator_covariance(Ts_fast, 1000), I(2)) + 1e-9I
R2 = 2pi/2048/10 * I(2)
R10 = copy(10R1)
R10[1,1] *= 1000000 # Not sure about the initial arm angle
d0 = MvNormal(x0, R10)
x0 = SA[0.0, 0, 0, 0.0]
d0 = MvNormal(x0, R10)

kf = UnscentedKalmanFilter(psim.ddyn, psim.measurement, R1, R2, d0; ny, nu, psim.p)
x_names = [:ϕ, :θ, :ϕ̇, :θ̇]
u_names = [:u]
y_names = [:ϕ, :θ]
# QuanserInterface.go_home(process, r = 0)
dynamics = FunctionSystem(psim.dynamics, psim.measurement; x=x_names, u=u_names, y=y_names)
N   = 100 # MPC prediction horizon
discrete_dynamics = MPC.rk4(dynamics, Ts_fast; supersample=2)

r = [0, pi, 0, 0]

xmin = [-deg2rad(80), -Inf, -Inf, -Inf]
xmax = [deg2rad(80), Inf, Inf, Inf]
umin = [-8]
umax = [8]
terminal_set_u = [deg2rad(30), Inf, 1, 1] # Pend angle handled by TerminalStateConstraint
terminal_set_l = [-deg2rad(30), -Inf, -1, -1]
bounds_constraints = BoundsConstraint(; xmin, xmax, umin, umax, xNmin = terminal_set_l, xNmax = terminal_set_u)

scale_x = [pi, pi, 10, 20]
scale_u = [7.0]

const Q1_ = Diagonal([1, 1, 1, 0.1])
const Q2_ = 1I(1)
QN, A0, B0 = MPC.calc_QN_AB(Q1, Q2, nothing, discrete_dynamics, r, [0], psim.p)

const QN_ = SMatrix{4,4}(QN)

running_cost = StageCost() do si, p, t
    e = si.x - si.r
    dot(e, Q1_, e) + dot(si.u, Q2_, si.u)
end

# TODO: consider adding large terminal cost with LQR cost to go

# difference_cost = DifferenceCost() do e, p, t
#     10abs2(e[])
# end

terminal_cost = TerminalCost() do ti, p, t
    e = ti.x - ti.r
    dot(e, QN_, e)
end

# This is used only for the pendulum angle, the other state variables are bounded using BoundsConstraint
terminal_state_constraint = TerminalStateConstraint([-Inf], [cos(pi+deg2rad(5))]) do ti, p, t
    cos(ti.x[2])
end

objective = Objective(running_cost, terminal_cost)

x_init = zeros(nx, N+1) .+ x0
u_init = 0.01randn(nu, N) # lb[5:6] .* ones(1, N)
x_init, u_init = MPC.rollout(discrete_dynamics, x0, u_init, psim.p, 0)
oi = ObjectiveInput(x_init, u_init, r)


solver = MPC.IpoptSolver(;
        verbose                     = true,
        printerval                  = 10,
        tol                         = 1e-4,
        acceptable_tol              = 1e-1,
        max_iter                    = 2500,
        max_cpu_time                = 50.0,
        max_wall_time               = 50.0,
        constr_viol_tol             = 1e-4,
        acceptable_constr_viol_tol  = 0.02,
        compl_inf_tol               = 1e-3,
        acceptable_iter             = 10,
        exact_hessian               = true,
        # linear_inequality_constraints = true,
        # mu_strategy                 = "adaptive",
        # mu_init                     = 1e-6,
        acceptable_obj_change_tol = 0.1,

)

# disc = Trapezoidal(dyn=dynamics)
disc = CollocationFinE(dynamics, n_colloc=2, hold_order=0)
# obs = StateFeedback(discrete_dynamics, Vector(x0))

prob = GenericMPCProblem(
    dynamics;
    N,
    observer = kf,
    objective,
    constraints = [bounds_constraints, terminal_state_constraint],
    psim.p,
    Ts,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
    scale_x,
    scale_u,
    disc,
    verbose = true,
);

x_presolve,u_presolve = get_xu(prob)

# plot(
#     plot(x_presolve', layout=4, title=permutedims(state_names(dynamics))),
#     plot(u_presolve', title="u"),
# ) |> display

# @time history = MPC.solve(prob; x0, T = 100, verbose = true, noise=false, dyn_actual = discrete_dynamics, callback = (args...)->sleep(0.001));

# plot(history) |> display