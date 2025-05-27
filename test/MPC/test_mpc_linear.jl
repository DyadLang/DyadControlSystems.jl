using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.MPC: modelfit
using StaticArrays
using ControlSystems: Discrete
using Statistics, LinearAlgebra
using Test
using Plots


function XUcost(hist)
    X,E,R,U,Y = reduce(hcat, hist)
    X, U, MPC.lqr_cost(hist)
end

## Test behavior on linear system

Ts = 1
sys = ss([1 Ts; 0 1], [0; 1;;], [1 0], 0, Ts)


nx = sys.nx
nu = sys.nu
N = 10
x0 = [1.0, 0]
r = zeros(nx) # reference state
# Control limits
umin = -100 * ones(nu)
umax = 100 * ones(nu)
# State limits (state constraints are soft by default)
xmin = -100 * ones(nx)
xmax = 100 * ones(nx)

constraints = MPCConstraints(; umin, umax, xmin, xmax, soft=false)

solver = OSQPSolver(
    verbose = false,
    eps_rel = 1e-10, # solve to high accuracy to compare to known solution
    max_iter = 1500,
    check_termination = 5,
    sqp_iters = 1,
    dynamics_interval = 10,
    polish = true,
)
T = 20
Q2 = spdiagm(ones(nu)) # control cost matrix
Q1 = spdiagm(ones(nx)) # state cost matrix

observer = StateFeedback(sys, x0, nu, sys.ny)
model = LinearMPCModel(sys, observer; constraints, v=1:nx, z=1:nx, x0)
prob = LQMPCProblem(model; Q1,Q2,N,r,solver)







@testset "updating of problem matrices" begin
    @info "Testing updating of problem matrices"

    N = 4
    for soft in [false, true]
        @show soft
        constraints = MPCConstraints(; umin, umax, xmin, xmax, soft)
        model = LinearMPCModel(sys, observer; constraints, v=1:nx, z=1:nx, x0)
        ns = soft ? 2nx : 0
        ns1 = soft ? nx : 0
        r = randn(nx)
        prob = LQMPCProblem(model; Q1,Q2,N,r,solver)
        QN = prob.QN

        u = randn(nu, N)
        x = zeros(nx, N + 1)
        x[:, 1] .= x0
        xo, uo = MPC.rollout!(prob.dynamics, x, copy(u))
        @test x[:, 2] == prob.dynamics(x[:, 1], u[:, 1], 0, 0)
        @test uo == u
        @test xo == x

        Ad, Bd = MPC.linearize(prob.dynamics, x[:, 1], u[:, 1], 0, 0)

        A = deepcopy(prob.A)
        MPC.update_constraints!(prob, x, u, x0, update_dynamics=true)
        @test A.colptr == prob.A.colptr
        @test A.rowval == prob.A.rowval

        @test prob.A[(1:nx) .+ 0, (1:nx) .+ 0] == -I
        @test prob.A[(1:nx) .+ nx, (1:nx) .+ 0] == Ad
        @test prob.A[(1:nx) .+ nx, (1:nu) .+ nx] == Bd
        @test prob.q[(1:nx)] == -Q1 * r

        @test prob.A[(1:nx) .+ 2nx, (1:nx) .+ (nx+nu+ns)] == Ad
        @test prob.A[(1:nx) .+ 2nx, (1:nu) .+ (2nx+nu+ns)] == Bd
        @test prob.q[(1:nx) .+ (nx+nu+ns)] == -Q1 * r

        @test prob.q[end-nx+1:end] ≈ -QN * r
        uinds = [repeat([falses(nx); trues(nu); falses(ns)], N); falses(nx)]
        @test all(iszero, prob.q[uinds])

        sinds = [repeat([falses(nx); falses(nu); trues(ns)], N); falses(nx)]
        @test all(==(prob.qs), prob.q[sinds])

        Aineq = A[(N+1)*nx+1:end, :]
        @test size(Aineq, 1) == N*((1+soft)*(nx+nu)+ns) # currently no inequality for xN
        if soft
            @test Aineq[1:nx, 1:(nx+nu+ns)] == [I(nx) zeros(nx, nu) -I(ns1) 0I(ns1)]
            @test Aineq[(1:nx) .+ (nx+nu), 1:(nx+nu+ns)] == [I(nx) zeros(nx, nu) 0I(ns1) I(ns1)]
        else
            @test Aineq[1:nx, 1:(nx+nu+ns)] == [I(nx) zeros(nx, nu)]
        end

        @test Aineq[(1:nu) .+ nx, 1:(nx+nu+ns)] == [zeros(nu, nx) ones(nu, nu) zeros(nu, ns)]
    end
end


observer = StateFeedback(sys, x0, nu, sys.ny)
model = LinearMPCModel(sys, observer; constraints, v=1:nx, z=1:nx, x0)
prob = LQMPCProblem(model; Q1,Q2,N,r,solver)

@time hist = MPC.solve(prob; x0, T, verbose = false)#, callback=plot_callback)
X, U, cost = XUcost(hist)
plot(hist) |> display

@test X ≈ MPC.rollout(sys, x0, U)[1]

QN, Ad, Bd = MPC.calc_QN_AB(Q1, Q2, 0Q2, sys)
L = lqr(Discrete, Matrix(Ad), Matrix(Bd), Matrix(Q1), Matrix(Q2))

@test X[:, 1] == x0
@test U[:, 1] ≈ -L*x0 rtol=1e-5
@test X[:, 2] == Ad*x0 + Bd*U[:, 1]
@test X[:, 3] == Ad*X[:,2] + Bd*U[:, 2]
@test U[:, 2] ≈ -L*X[:, 2] rtol=1e-3
@test U[:, 3] ≈ -L*X[:, 3] rtol=1e-3
@test dot(x0, QN, x0) ≈ cost rtol=1e-7



# Test with soft constraints
observer = StateFeedback(sys, x0, nu, sys.ny)
constraints = MPCConstraints(; umin, umax, xmin, xmax, soft=true)
model = LinearMPCModel(sys, observer; constraints, v=1:nx, z=1:nx, x0)
prob = LQMPCProblem(model; Q1,Q2,N,r,solver)

@time hist = MPC.solve(prob; x0, T, verbose = false)#, callback=plot_callback)
X, U, cost = XUcost(hist)
plot(hist) |> display

@test X ≈ MPC.rollout(sys, x0, U)[1]

QN, Ad, Bd = MPC.calc_QN_AB(Q1, Q2, 0Q2, sys)
L = lqr(Discrete, Matrix(Ad), Matrix(Bd), Matrix(Q1), Matrix(Q2))

@test X[:, 1] == x0
@test U[:, 1] ≈ -L*x0 rtol=1e-5
@test X[:, 2] == Ad*x0 + Bd*U[:, 1]
@test X[:, 3] == Ad*X[:,2] + Bd*U[:, 2]
@test U[:, 2] ≈ -L*X[:, 2] rtol=1e-3
@test U[:, 3] ≈ -L*X[:, 3] rtol=1e-3
@test dot(x0, QN, x0) ≈ cost rtol=1e-7








## Benchmark large linear system
# Ts = 1
# sys = ssrand(10, 10, 15, proper=true, Ts = Ts)


# nx = sys.nx
# nu = sys.nu
# N = 10
# x0 = randn(nx)
# xr = zeros(nx) # reference state
# # Control limits
# umin = -100 * ones(nu)
# umax = 100 * ones(nu)
# # State limits (state constraints are soft by default)
# xmin = -100 * ones(nx)
# xmax = 100 * ones(nx)
# solver = OSQPSolver(
#     verbose = false,
#     eps_rel = 1e-10, # solve to high accuracy to compare to known solution
#     max_iter = 1500,
#     check_termination = 5,
#     polish = true,
# )
# T = 10
# Q2 = spdiagm(ones(nu)) # control cost matrix
# Q1 = spdiagm(ones(nx)) # state cost matrix


# kf = KalmanFilter(sys.A, sys.B, sys.C, 0, Matrix(Q1), Matrix(Q2))

# prob = LQMPCProblem(sys; observer=kf, Q1,Q2,umin,umax,xmin,xmax,N,xr,solver)

# @time hist = MPC.solve(prob; x0, T, verbose = false, noise=true)#, callback=plot_callback)
# # plot(hist) |> display

# @btime MPC.solve($prob; x0, T, verbose = false, noise = true);
# 6.496 ms (1811 allocations: 629.02 KiB) for 10 iterations



## Glover McFarlane tuning
# Preliminary design
Ts = 0.01
disc = G -> c2d(balreal(G)[1], Ts)
G = tf(200, [10, 1])*tf(1, [0.05, 1])^2     |> ss |> disc
Gd = tf(100, [10, 1])                       |> ss |> disc
W1 = tf([1, 2], [1, 1e-2])                  |> ss |> disc
gmf = K, γ, info = glover_mcfarlane(G, 1.1; W1)

sys = info.Gs
method = GMF(gmf)
Q1,Q2 = inverse_lqr(method)
Q2 = Symmetric(Matrix(1.0*Q2))

nx = sys.nx
nu = sys.nu
N = 10
x0 = [1.0,2,3,4]
# x0 = [1.0,0,0,0]
r = zeros(nx) # reference state    
# Control limits
umin = -100 * ones(nu)
umax = 100 * ones(nu)
# State limits (state constraints are soft by default)
xmin = -1000 * ones(nx)
xmax = 1000 * ones(nx)
constraints = MPCConstraints(; umin, umax, xmin, xmax)
solver = OSQPSolver(
    verbose = false,
    eps_rel = 1e-10, # solve to high accuracy to compare to known solution
    max_iter = 1500,
    check_termination = 5,
    polish = true,
)
T = 200


## Verify that the MPC controller acts as expected under ideal conditions
## No observer

observer = StateFeedback(sys, x0, nu, sys.ny)
model = LinearMPCModel(sys, observer; constraints, v=1:nx, z=1:nx, x0)

prob = LQMPCProblem(model; Q1,Q2,N,r,solver, qs=0, qs2=0)
@time hist = MPC.solve(prob; x0, T, verbose = false, noise=false)#, callback=plot_callback)
X,E,R,U,Y = reduce(hcat, hist)
res_mpc = lsim(sys, U, x0=x0)
mpc_y = sys.C*X

sys_ideal = let
    A,B,C,D = ssdata(sys)
    ss(A, [B B], [C; I(sys.nx)], 0, sys.timeevol)
end
Ks_ideal = ss(info.F, sys.timeevol)
Gcl2 = lft(sys_ideal, Ks_ideal)
x0cl = x0
res_ideal = lsim(Gcl2, zeros(1,T+1), x0=x0cl)
# plot(res_mpc, lab="MPC")
# plot!(res_ideal, sp=1, lab="Ideal") |> display
@test mpc_y[:,1:end-1] ≈ res_mpc.y
@test res_mpc.y ≈ res_ideal.y[:,1:end-1]


##



## With observer
H0 = kalman(sys, I(sys.nx), I(sys.ny))
H = -info.Hkf 
@test sign.(H) == sign.(H0)
# Since there is no error in the dynamics, we have to introduce it somewhere, we perturb the initial condition of the observer
kf = DyadControlSystems.FixedGainObserver(sys, 0.8*x0, H)

model = LinearMPCModel(sys, kf; constraints, v=1:nx, z=1:nx, x0=0.8*x0)

prob = LQMPCProblem(model; Q1,Q2,N,r,solver, qs=0, qs2=0)
@time hist = MPC.solve(prob; x0=x0, T, verbose = false, noise=false, reset_observer=false)#, callback=plot_callback)
X,E,R,U,Y = reduce(hcat, hist)
res_mpc = lsim(sys, U, x0=x0)
mpc_y = sys.C*X
@test mpc_y[:,1:end-1] ≈ res_mpc.y

Ks = observer_controller(info)
Gcl2 = feedback(sys*Ks)
x0cl = [x0; 0.8*x0]
res_ideal = lsim(Gcl2, zeros(1,T+1), x0=x0cl)
if isinteractive()
    plot(res_mpc, lab="MPC")
    plot!(res_ideal, sp=1, lab="Ideal")
    plot!(res_mpc.t, Y', lab="MPC Y")    |> display
end
@test norm(res_mpc.y - res_ideal.y[:,1:end-1])/ norm(res_ideal.y[:,1:end-1]) < 0.02



## Testing LinearPredictionModel
using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.MPC: modelfit
using Statistics, LinearAlgebra
using Test
using Plots


@testset "Quad tank" begin


## Quad tank process

k1, k2, kc, g = 3.33, 3.35, 0.5, 981
A1, A3, A2, A4 = 28, 28, 32, 32
a1, a3, a2, a4= 0.071, 0.071, 0.057, 0.057
h01, h02, h03, h04 = 12.4, 12.7, 1.8, 1.4
T1, T2 = (A1/a1)sqrt(2*h01/g), (A2/a2)sqrt(2*h02/g)
T3, T4 = (A3/a3)sqrt(2*h03/g), (A4/a4)sqrt(2*h04/g)
c1, c2 = (T1*k1*kc/A1), (T2*k2*kc/A2)
γ1, γ2 = 0.7, 0.6

# Define the process dynamics
A = [-1/T1     0 A3/(A1*T3)          0;
     0     -1/T2          0 A4/(A2*T4);
     0         0      -1/T3          0;
     0         0          0      -1/T4];
B = [γ1*k1/A1     0;
     0                γ2*k2/A2;
     0                (1-γ2)k2/A3;
     (1-γ1)k1/A4 0              ];
# B = B[:,1]
C = [kc 0 0 0;
     0 kc 0 0];
D = 0
Gc = ss(A,B,C,D)
Ts = 2 # sample time

disc = (x) -> c2d(ss(x), Ts)

G = disc(Gc)
W1 = tf(1,[1, 1e-6]) |> disc # "Shape" the plant with an integrator
W1 = W1*I(G.nu)
A,B,C,D = ssdata(G)



##
nu = G.nu  # number of control inputs
nx = G.nx  # number of states
ny = G.ny # number of outputs (here we assume that all states are measurable)
N = 30 # MPC optimization horizon
x0 = zeros(nx) # Initial state
# TODO:  we previously had r = zeros(G.ny) below but I hacked in 6 to keep moving. Make sure we figure out if we should have G.ny
r = zeros(G.ny) # x_ref # reference state
x0 = 1ones(G.nx)

# Control limits
umin = -100 * ones(nu)
umax = 100 * ones(nu)
# State limits (state constraints are soft by default)
xmin = -100ones(G.nx)
xmax = 100ones(G.nx)

constraints = MPCConstraints(; umin, umax, xmin, xmax)

Cv, Dv, vmin, vmax, soft_indices = MPC.setup_output_constraints(G.nx, G.nu, constraints, OperatingPoint(), I(G.nx))
@test soft_indices == 1:G.nx
@test Cv == [I(G.nx); zeros(G.nu, G.nx)]
@test Dv == [zeros(G.nx, G.nu); I(G.nu)]

constraints2 = BoundsConstraint(; umin, umax, xmin, xmax)
Cv2, Dv2, vmin2, vmax2, soft_indices2 = MPC.setup_output_constraints(G.nx, G.nu, constraints2, OperatingPoint(), I(G.nx))
@test Cv2 == Cv
@test Dv2 == Dv
@test vmin2 == vmin
@test vmax2 == vmax
@test soft_indices2 == soft_indices



K = ControlSystemsBase.kalman(ControlSystemsBase.Discrete, A, C, Matrix(1e-6*I(nx)), Matrix(I(ny)))
pm = RobustMPCModel(G; W1, W2=I(G.ny), constraints, x0=0*x0, K)



solver = OSQPSolver(
    eps_rel = 1e-7,
    max_iter = 50000,        # in the QP solver
    check_termination = 5, # how often the QP solver checks termination criteria
    sqp_iters = 1,
    dynamics_interval = 10000, # The linearized dynamics is updated with this interval
    polish = true,
)
prob = LQMPCProblem(pm;
    qs = 10000,
    qs2 = 10000,
    N,
    r,
    solver,
)

history = MPC.solve(prob; x0, T = 70, verbose=false, dyn_actual=G, Cz_actual=G.C)

X,E,R,U,Y = reduce(hcat, history)

K = pm.gmf[1]

K,gam,info = glover_mcfarlane(G; W1, strictly_proper=false)

T = extended_gangoffour(G, K)
res_gmf = lsim(T, repeat([r; 0*r],1,length(history)), x0=[x0; zeros(K.nx)])
res_gmf2 = lsim(-feedback(G*K), repeat(r,1,length(history)), x0=[x0; zeros(K.nx)])
@test res_gmf.y[1:2, :] ≈ res_gmf2.y

res_rec = lsim(G, U, x0=x0)


# plot(history)
# plot(res_gmf, layout=2, sp = [1 1 2 2], lab="gmf", seriestype=:path)
# plot!(res_rec, plotu=true, sp = [1 1 2 2], lab="rec", seriestype=:path)


# I believe the MPC result with the above tuning should be almost identical to if the closed-loop with the GMF controller is simulated by lsim, but there is a clear difference in the dynamics. To be revisited
@test_broken norm(res_rec.y - res_gmf.y[1:2, :]) / norm(res_rec.y) < 0.01


# sys = info.Gs
# res_mpc = lsim(G, U, x0=x0)
# mpc_y = sys.C*X
# sys_ideal = let
#     A,B,C,D = ssdata(sys)
#     ss(A, [B B], [C; I(sys.nx)], 0, sys.timeevol)
# end
# Ks_ideal = ss(info.F, sys.timeevol)
# Gcl2 = lft(sys_ideal, Ks_ideal)
# x0cl = [x0; zeros(W1.nx)]
# res_ideal = lsim(Gcl2, zeros(Gcl2.nu,length(history)), x0=x0cl)
# plot(res_mpc, lab="MPC")
# plot!(res_ideal, sp=1, lab="Ideal") |> display
# @test mpc_y[:,1:end-1] ≈ res_mpc.y
# @test res_mpc.y ≈ res_ideal.y[:,1:end-1]



## Test with a more interesting reference

# Find steady state
y_ref = [5,5] # Desired output
# x_ref, u_ref = linear_trim(info.Gs, y_ref)# x = xr)#, xw=0.000000001ones(Gs.nx))
xu_ref = ([A-I B; C D]) \ [zeros(G.nx); y_ref]
x_ref = xu_ref[1:end-G.nu]
u_ref = xu_ref[end-G.nu+1:end]

@assert norm(A*x_ref + B*u_ref - x_ref) < 10eps()
@assert norm(C*x_ref - y_ref) < 10eps()

##
nu = G.nu  # number of control inputs
nx = G.nx  # number of states
ny = G.ny # number of outputs (here we assume that all states are measurable)
N = 30 # MPC optimization horizon
x0 = zeros(nx) # Initial state
r = y_ref # x_ref # reference state
ur = zeros(nu)
x0 = 1ones(G.nx)

# Control limits
umin = 0 * ones(nu)
umax = 3 * ones(nu)
# State limits (state constraints are soft by default)
xmin = 0*ones(G.nx)
xmax = Float64[11, 11, 2, 2]#11ones(G.nx)

constraints = MPCConstraints(; umin, umax, xmin, xmax)

K = ControlSystemsBase.kalman(ControlSystemsBase.Discrete, A,C, Matrix(1e-6*I(nx)), Matrix(I(ny)))
pm = RobustMPCModel(G; W1, W2=I(G.ny), constraints, x0, K)

solver = OSQPSolver(
    eps_rel = 1e-7,
    max_iter = 50000,        # in the QP solver
    check_termination = 5, # how often the QP solver checks termination criteria
    sqp_iters = 1,
    dynamics_interval = 10000, # The linearized dynamics is updated with this interval
    polish = true,
)
prob = LQMPCProblem(pm;
    qs = 10000,
    qs2 = 10000,
    N,
    r,
    solver,
)



# TODO: Cz is related to the Glover-McFarlane controller which in essence determines the "controlled variables" z. This is distinct from measurements which in this approach are divided into controlled outputs and other measurements. References are only provided for controlled measurements.

# Test that the output when using u_actual to simulate G converges to the reference
@time history = MPC.solve(prob; x0, T = 200, verbose=false, dyn_actual=G, Cz_actual = G.C)
U = reduce(hcat, history.U)
res = lsim(G, U)
@test res.y[:, end] ≈ r rtol=1e-2

@test all(maximum(U, dims=2) .< umax .+ 1e-4)
@test all(minimum(U, dims=2) .> umin .- 1e-4)



if isinteractive()
    plot(history)
    plot!(res, sp=1, lab="reconstruct")
    display(current())
end



x1 = [x[1] for x in history.X]
@test maximum(x1) < vmax[1]+0.6 # Some slack for soft constraints





@testset "output reference for LinearMPCModel" begin
    @info "Testing output reference for LinearMPCModel"
    Ts = 1
    sys = ss([1 Ts; 0 1], [0; 1;;], [1 0], 0, Ts)
    nx = sys.nx
    nu = sys.nu
    ny = sys.ny
    N = 10
    x0 = [1.0, 0]
    umin = -100 * ones(nu)
    umax = 100 * ones(nu)
    constraints = MPCConstraints(; umin, umax)

    solver = OSQPSolver(
        verbose = false,
        eps_rel = 1e-10, # solve to high accuracy to compare to known solution
        max_iter = 1500,
        check_termination = 5,
        sqp_iters = 1,
        dynamics_interval = 10,
        polish = true,
    )
    T = 20
    Q2 = spdiagm(ones(nu)) # control cost matrix
    Q1 = spdiagm(ones(ny)) # state cost matrix

    observer = StateFeedback(sys, x0, nu, sys.ny)
    model = LinearMPCModel(sys, observer; constraints, v=1:nx, z=sys.C, x0)

    # Set r to zero
    r = zeros(ny)
    prob = LQMPCProblem(model; Q1,Q2,N,r,solver)
    hist = MPC.solve(prob; x0, T, verbose = false)
    X, U, cost = XUcost(hist)
    # isinteractive() && plot(hist) |> display
    @test sys.C*X[:, end] ≈ r atol=1e-5

    # Change r in an existing problem
    r = [-0.9]
    MPC.update_xr!(prob, r, [0])
    hist = MPC.solve(prob; x0, T, verbose = false)
    X, U, cost = XUcost(hist)
    # isinteractive() && plot(hist) |> display
    @test sys.C*X[:, end] ≈ r atol=1e-5   

    # Create a new problem with a different r
    prob = LQMPCProblem(model; Q1,Q2,N,r,solver)
    hist = MPC.solve(prob; x0, T, verbose = false)
    X, U, cost = XUcost(hist)
    # isinteractive() && plot(hist) |> display
    @test sys.C*X[:, end] ≈ r atol=1e-5


end




@testset "sysid example" begin
    @info "Testing sysid example"
    
    Ts = 0.08
    sys = let
        tempA = [0.0 0.5 0.0 0.0; 0.0 0.0 0.5 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 -0.40665868538055294 1.2887298646631016]
        tempB = [0.0; 0.0; 0.0; 0.5;;]
        tempC = [0.3506080016815353 0.26207183738027917 0.0 0.0]
        tempD = [0.0;;]
        ss(tempA, tempB, tempC, tempD, 0.08)
    end
    x0 = [5.2998070886966975, 10.599614177393395, 21.19922835478679, 21.19922835478679]
    u0 = [4.999999999999928]
    y0 = [4.636015135657129]
    op = OperatingPoint(x0, u0, y0)

    xr = copy(x0)
    ur = copy(u0)[:]
    yr = sys.C * xr

    umin = [3.41]
    umax = [6.41]

    constraints = MPCConstraints(; umin, umax)

    solver = OSQPSolver(
        verbose = false,
        eps_rel = 1e-6,
        max_iter = 1500,
        check_termination = 5,
        polish = true,
    )

    Q1 = Diagonal(ones(sys.ny))   # output cost matrix
    Q2 = 0.01 * spdiagm(ones(sys.nu)) # control cost matrix

    R1 = I(sys.nx)
    R2 = I(sys.ny)
    kf = KalmanFilter(ssdata(sys)..., R1, R2, MvNormal(x0, R1))
    # kf = StateFeedback(sys, x0, sys.nu, sys.ny) # Should also pass tests


    N = 10
    Cz = sys.C
    predmodel = LinearMPCModel(sys, kf; constraints, op, x0, z = Cz)
    prob = LQMPCProblem(predmodel; Q1, Q2, N, solver, r = yr)

    T = 80 # Simulation length (time steps)
    x0sim = x0
    hist = MPC.solve(prob; x0 = x0sim, T)
    # plot(hist, ploty = true); hline!([umin[] umax[]], sp = 2, l = (:black, :dash), primary = false)

    @test Cz*hist.X[end] ≈ yr atol=1e-5

    # Change r to something else

    r = yr .+ 0.5
    MPC.update_xr!(prob, r, [0])
    hist = MPC.solve(prob; x0 = x0sim, T)
    # plot(hist, ploty = true); hline!([umin[] umax[]], sp = 2, l = (:black, :dash), primary = false)
    @test Cz*hist.X[end] ≈ r rtol=1e-2 # Small stationary error expected in this case


end

end
