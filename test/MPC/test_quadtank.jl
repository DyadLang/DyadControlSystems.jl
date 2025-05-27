using DyadControlSystems
using DyadControlSystems.MPC

function sqp_callback(x,u,xopt,t,sqp_iter,prob)
    nx, nu = size(x,1), size(u,1)
    c = MPC.lqr_cost(x,u,prob)
    plot(x[:, 1:end-1]', layout=2, sp=1, c=(1:nx)', label="x nonlinear", title="Time: $t SQP iteration: $sqp_iter")
    plot!(xopt[:, 1:end-1]', sp=1, c=(1:nx)', l=:dash, label="x opt")
    plot!(u', sp=2, c=(1:nu)', label = "u", title="Cost: $(round(c, digits=5))") |> display
    sleep(0.001)
end
## Nonlinear quadtank
kc = 0.5
function quadtank(h,u,p=nothing,t=nothing)
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a1, a3, a2, a4= 0.03, 0.03, 0.03, 0.03
    γ1, γ2 = 0.3, 0.3

    ssqrt(x) = √(max(x, zero(x)) + 1e-3)
    # ssqrt(x) = sqrt(x)
    xd = SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end

nu = 2 # number of controls
nx = 4 # number of states
ny = 2 # number of outputs
Ts = 2 # sample time

discrete_dynamics0 = rk4(quadtank, Ts)
x_names = :h^4
measurement = (x,u,p,t) -> kc*x[1:2]
discrete_dynamics = FunctionSystem(discrete_dynamics0, measurement, Ts, x=x_names, u=:u^2, y=:h^2)


# Control limits
umin = 0 * ones(nu)
umax = 1 * ones(nu)

# State limits (state constraints are soft by default)
xmin = 0 * ones(nx)
xmax = Float64[12, 12, 8, 8]
constraints = NonlinearMPCConstraints(; umin, umax, xmin, xmax, soft=true)

x0 = [2, 1, 8, 3]       # Initial state
xr = [10, 10, 4.9, 4.9] # reference state
ur = [0.26, 0.26]

N = 10
Q1 = 10I(nx)
Q2 = I(nu)
qs = 100
qs2 = 100000

R1 = 1e-5*I(nx)
R2 = I(ny)

solver = OSQPSolver(
    eps_rel = 1e-6,
    eps_abs = 1e-6,
    max_iter = 50000,
    check_termination = 5,
    sqp_iters = 1, 
    dynamics_interval = 1,
    verbose=false,
    polish=false, 
)

kf = let
    A,B = DyadControlSystems.linearize(discrete_dynamics, xr, ur, 0, 0)
    C,D = DyadControlSystems.linearize(measurement, xr, ur, 0, 0)
    KalmanFilter(A, B, C, D, R1, R2)
end
ekf = ExtendedKalmanFilter(
    kf,
    discrete_dynamics.dynamics,
    discrete_dynamics.measurement,
)
prob = QMPCProblem(discrete_dynamics; observer=ekf,Q1,Q2,qs,qs2,constraints,N,xr,ur,solver)

@time hist = MPC.solve(prob; x0, T = 1000÷Ts, verbose = false, noise=0)#, sqp_callback)
plot(hist, plot_title="Nonlinear MPC")

# Test convergence
@test MPC.lqr_cost(hist) < 1.1*48320
@test hist.X[end] ≈ xr rtol=1e-1
U = reduce(hcat, hist.U)
@test all(maximum(U, dims=2) .< umax .+ 1e-4)
@test all(minimum(U, dims=2) .> umin .- 1e-4)


## SQP iterations
solver = OSQPSolver(
    eps_rel = 1e-6,
    eps_abs = 1e-3,
    max_iter = 50000,
    check_termination = 2,
    sqp_iters = 5, 
    dynamics_interval = 1,
    verbose=false,
    polish=true, 
)

qs = 0
qs2 = 0

ekf = ExtendedKalmanFilter(
    kf,
    discrete_dynamics.dynamics,
    discrete_dynamics.measurement,
)
prob = QMPCProblem(discrete_dynamics; observer=ekf,Q1,Q2,qs,qs2,constraints,N,xr,ur,solver)
@time hist = MPC.solve(prob; x0, T = 1000÷Ts, verbose = false)#, sqp_callback)

plot(hist, plot_title="Nonlinear MPC with SQP iterations")

# Test convergence
@test MPC.lqr_cost(hist) < 1.1*48148
@test hist.X[end] ≈ xr atol=1e-1
U = reduce(hcat, hist.U)
@test all(maximum(U, dims=2) .< umax .+ 1e-3)
@test all(minimum(U, dims=2) .> umin .- 1e-3)


# ==============================================================================
## Linear MPC
# ==============================================================================
k1, k2, g = 1.6, 1.6, 9.81
A1 = A3 = A2 = A4 = 4.9
a1, a3, a2, a4= 0.03, 0.03, 0.03, 0.03
h01, h02, h03, h04 = xr
T1, T2 = (A1/a1)sqrt(2*h01/g), (A2/a2)sqrt(2*h02/g)
T3, T4 = (A3/a3)sqrt(2*h03/g), (A4/a4)sqrt(2*h04/g)
c1, c2 = (T1*k1*kc/A1), (T2*k2*kc/A2)
γ1, γ2 = 0.3, 0.3

# Define the process dynamics
Ac = [-1/T1     0 A3/(A1*T3)          0
     0     -1/T2          0 A4/(A2*T4)
     0         0      -1/T3          0
     0         0          0      -1/T4]
Bc = [γ1*k1/A1     0
     0                γ2*k2/A2
     0                (1-γ2)k2/A3
     (1-γ1)k1/A4 0              ]

Ac0,Bc0 = DyadControlSystems.linearize(quadtank, xr, ur, 0, 0)
Ad0,Bd0 = DyadControlSystems.linearize(discrete_dynamics, xr, ur, 0, 0)

@test norm(Ac0-Ac) < 1e-5
@test norm(Bc0-Bc) < 1e-5

Cc = kc*[I(2) 0*I(2)] # Measure the first two tank levels
# Cc = kc*I(nx)
Dc = 0
Gc = ss(Ac,Bc,Cc,Dc)

disc = (x) -> c2d(ss(x), Ts)
G = disc(Gc)

@test norm(Ad0-G.A) < 1e-5
@test norm(Bd0-G.B) < 1e-5

kf = let (A,B,C,D) = ssdata(G)
    KalmanFilter(A, B, C, D, R1, R2)
end

qs = 100
qs2 = 100000

solver = OSQPSolver(
    eps_rel = 1e-6,
    eps_abs = 1e-6,
    max_iter = 50000,
    check_termination = 5,
    verbose=false,
    polish=true, 
)

op = OperatingPoint(xr, ur, Cc*xr)
constraints = MPCConstraints(; umin, umax, xmin, xmax, soft=true)
pm = LinearMPCModel(G, kf; constraints, op, x0, strictly_proper=false)
@test pm.observer.x == x0 - op.x
@test pm.vmin == [constraints.xmin; constraints.umin] .- [op.x; op.u]
@test pm.vmax == [constraints.xmax; constraints.umax] .- [op.x; op.u]


r = xr
prob_lin = LQMPCProblem(pm; Q1, Q2, qs, qs2,N,r,solver)

@time hist_lin = MPC.solve(prob_lin; x0, T = 1000÷Ts, verbose = false, noise=0, dyn_actual=discrete_dynamics)
# (36.06 k allocations: 7.850 MiB)
# 75.852 ms (34560 allocations: 7.62 MiB) fix allocations in symmetrize



cost_lin = MPC.lqr_cost(hist_lin)

plot(hist_lin, plot_title="Linear MPC") |> display

# Test convergence (the linear controller without integral action ends up with a very large steady-state error)
U = reduce(hcat, hist_lin.U)
X = reduce(hcat, hist_lin.X)
@test hist_lin.X[end] ≈ r atol=1e-2 norm=maximum
@test all(maximum(U, dims=2) .< umax .+ 2e-2)
@test all(minimum(U, dims=2) .> umin .- 2e-2)

@test all(maximum(X, dims=2) .< xmax .+ 0.5)
@test all(minimum(X, dims=2) .> xmin .- 0.5)

## Test OperatingPointWrapper

ekf = ExtendedKalmanFilter(
    kf,
    discrete_dynamics.dynamics,
    discrete_dynamics.measurement,
)

pm_ekf = LinearMPCModel(G, DyadControlSystems.OperatingPointWrapper(ekf, op); constraints, op, x0, strictly_proper=false)
prob_ekf = LQMPCProblem(pm_ekf; Q1, Q2, qs, qs2, N, r, solver)

@time hist_ekf = MPC.solve(prob_ekf; x0, T = 1000÷Ts, verbose = false, noise=0, dyn_actual=discrete_dynamics)
cost_ekf = MPC.lqr_cost(hist_ekf)

isinteractive() && plot(hist_ekf) |> display

@test cost_ekf <= cost_lin


##

solver = OSQPSolver(
    eps_rel = 1e-6,
    eps_abs = 1e-6,
    max_iter = 50000,
    check_termination = 5,
    sqp_iters = 1, 
    dynamics_interval = 1,
    verbose=false,
    polish=false, 
)

ekf = ExtendedKalmanFilter(
    kf,
    discrete_dynamics.dynamics,
    discrete_dynamics.measurement,
)

# When we use a nonlinear observer, we do not want to adjust the constraints to the operating point

pm_ekf = LinearMPCModel(G, DyadControlSystems.OperatingPointWrapper(ekf, op); constraints, op, x0, strictly_proper=false)

prob_ekf = LQMPCProblem(pm_ekf; Q1,Q2=Q2,qs,qs2,N,r,solver)

@time hist_ekf = MPC.solve(prob_ekf; x0, T = 1000÷Ts, verbose = false, noise=0, dyn_actual=discrete_dynamics)
plot(hist_ekf, plot_title="Linear MPC with nonlinear observer") |> display

# 253.127 ms (84070 allocations: 13.42 MiB)
# 287.515 ms (68070 allocations: 12.03 MiB) views in copy_x
# 276.004 ms (63066 allocations: 10.33 MiB) help type inference in getproperty
# 283.472 ms (57561 allocations: 9.60 MiB) more getproperty help in kalman filter


# Test convergence
U = reduce(hcat, hist_ekf.U)
X = reduce(hcat, hist_ekf.X)
@test hist_ekf.X[end] ≈ xr atol=1e-1
@test all(maximum(U, dims=2) .< umax .+ 2e-4)
@test all(minimum(U, dims=2) .> umin .- 2e-4)

@test all(maximum(X, dims=2) .< xmax .+ 0.5)
@test all(minimum(X, dims=2) .> xmin .- 0.5)

# ==============================================================================
## Robust design with output references
# ==============================================================================
constraints = MPCConstraints(; umin, umax, xmin, xmax)

W1 = tf(0.001*[100, 1],[1,1e-6]) |> disc # "Shape" the plant with a PI controller
W1 = W1*I(G.nu)
W2 = I(G.ny)

pm = RobustMPCModel(G; W1, W2, constraints, x0, op, K=kf)
@test size(pm.K) == (G.nx, G.ny)

solver = OSQPSolver(
    eps_rel = 2e-5,
    eps_abs = 2e-5,
    max_iter = 7000,
    check_termination = 5,
    sqp_iters = 1,
    dynamics_interval = 1,
    verbose=false,
    polish=false, # to get high accuracy
)
prob_roby = LQMPCProblem(
    pm;
    qs,
    qs2,
    N = 15,
    r = op.y,
    solver,
)

@time hist_roby = MPC.solve(prob_roby; x0, T = 1000÷Ts, verbose = false, noise=0, dyn_actual=discrete_dynamics, Cz_actual=G.C)
plot(hist_roby, plot_title="Linear MPC with robust loop shaping and output reference")

##

# The soft state-constraints are not fully respected; since the observer uses a linearized model, it fails to estimate the true value of the state and the actual value might thus be violating the constraints slightly. THe figure above shows state trajectories, we may plot also the output trajectories to verify that the output reference was met without steady-state tracking error
plot(hist_roby, ploty = true, plot_title="Linear MPC with robust loop shaping and output reference")




# Test convergence
U = reduce(hcat, hist_roby.U)
X = reduce(hcat, hist_roby.X)
@test hist_roby.X[end] ≈ xr atol=1e-1
@test all(maximum(U, dims=2) .< umax .+ 1e-4)
@test all(minimum(U, dims=2) .> umin .- 1e-4)

@test all(maximum(X, dims=2) .< xmax .+ 0.5)
@test all(minimum(X, dims=2) .> xmin .- 0.5)

## Test with input disturbance

# If we do not incorporate a disturbance observer, the W! integrator design leads to a steady-state error, this is indicated also by Bortoff 2019, without constraints, there is not stady-state error (and the simulation is about 20x faster)
constraints = MPCConstraints(; umin, umax)
pm = RobustMPCModel(G; W1, W2, constraints, x0, op, K=kf)
prob_roby = LQMPCProblem(
    pm;
    qs=10,
    qs2,
    N = 15,
    r = op.y,
    solver,
)

@time hist_roby = MPC.solve(prob_roby; x0, T = 7000÷Ts, verbose = false, noise=0.0, dyn_actual=discrete_dynamics, Cz_actual=G.C, disturbance = (u,t)-> [-0.1*(t > 2000), 0])
plot(hist_roby, plot_title="Linear MPC with robust loop shaping and output reference") |> display

U = reduce(hcat, hist_roby.U)
@test hist_roby.X[end] ≈ xr atol=1e-1
@test all(maximum(U, dims=2) .< umax .+ 1e-4)
@test all(minimum(U, dims=2) .> umin .- 1e-4)

# # ==============================================================================
# ## W2 integrator design (not yet supported)
# # ==============================================================================
# op = OperatingPoint(zeros(nx), zeros(nu), zeros(ny))
# constraints = MPCConstraints(G; umin=-nu*ones(nu), umax=nu*ones(nu), op)
# W2 = tf(0.001*[100, 1],[1,1e-6]) .* I(G.ny) |> disc 
# W1 = I(G.nu)
# @test DyadControlSystems.hasintegrator(W2)
# pm = RobustMPCModel(G; W1, W2, constraints, x0, op, state_reference=false, K = kf)

# @test isstable(pm.gmf[1])
# # w = exp10.(LinRange(-4, log10(pi/Ts), 200))
# # RobustAndOptimalControl.gangoffourplot2(G, pm.gmf[1], w)

# @test pm.nx == 4+2+4
# @test pm.nxw1 == 0
# @test pm.nxw2 == W2.nx
# @test pm.w2sinds == 1:W2.nx
# @test pm.xqinds == (1:G.nx) .+ (4+2)

# xr = 0*[10, 10, 4.9, 4.9]
# # x0 = xr
# aa = 0.1
# x0 = aa*[2, 1, 8, 3] + (1-aa)*xr
# # x0 = [2, 1, 8, 3];
# pm = RobustMPCModel(G; W1, W2, constraints, x0, op, state_reference=false, K = kf)
# solver = OSQPSolver(
#     eps_rel = 1e-5,
#     eps_abs = 1e-5,
#     max_iter = 7000,
#     check_termination = 5,
#     sqp_iters = 1,
#     dynamics_interval = 1,
#     verbose=false,
#     polish=false, # to get high accuracy
# )
# prob_roby = LQMPCProblem(;
#     dynamics = pm,
#     observer = pm, 
#     pm.Q1,
#     pm.Q2,
#     qs=10,
#     qs2,
#     N = 50,
#     xr = op.y,
#     solver,
# )


# # NOTE: I changed dyn_actual to G
# @time hist_roby = MPC.solve(prob_roby; x0, T = 1000÷Ts, verbose = false, noise=0, dyn_actual=G, Cz_actual=G.C)#, disturbance = (u,t)-> [-0.1*(t > 3000), 0])
# plot(hist_roby, plot_title="Linear MPC with robust loop shaping and output reference")
