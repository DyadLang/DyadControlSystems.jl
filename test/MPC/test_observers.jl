using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.MPC: modelfit
using StaticArrays
using Statistics, LinearAlgebra
using Test
using Plots
using LowLevelParticleFilters
import Distributions
using Distributions: Normal, MvNormal


## Test behavior on linear system
@testset "Kalman filter" begin
    @info "Testing Kalman filter"
    eye(n) = Matrix{Float64}(I(n))

    function lindyn(x, u, _, t) 
        A*x + B*u
    end

    
    
    nx = 2
    nu = 1
    ny = 2
    N = 10
    x0 = [1.0, 0]
    h = 1
    Ts = h
    linsys = FunctionSystem(lindyn, (x,u,p,t) -> x, Ts, x=:x^2, u=[:u], y = :y^2)

    A = [1 h; 0 1]
    B = [0; 1;;]
    C = eye(nx)

    Q1 = spdiagm(ones(nx)) # state cost matrix
    Q2 = spdiagm(ones(nu)) # control cost matrix

    R1 = LowLevelParticleFilters.double_integrator_covariance(h, 1) # state covariance matrix
    R2 = eye(ny) # measurement covariance matrix

    Rinf = Symmetric(ControlSystemsBase.are(ControlSystemsBase.Discrete, A', C', R1, R2))
    QN = ControlSystemsBase.are(ControlSystemsBase.Discrete, A, B, Matrix(Q1), Matrix(Q2))

    kf1     = KalmanFilter(A, B, C, 0, R1, R2, MvNormal(x0, Rinf))
    # ukf    = UnscentedKalmanFilter(dynamics, measurement,  R1, R2, MvNormal(R1))

    sys = ss(A,B,C,0,h)
    K = kalman(sys, R1, R2)
    kf2 = DyadControlSystems.FixedGainObserver(sys, x0, K)
    filters = [kf1, kf2]
    
    # Control limits
    umin = -100 * ones(nu)
    umax = 100 * ones(nu)
    # State limits (state constraints are soft by default)
    xmin = -100 * ones(nx)
    xmax = 100 * ones(nx)
    constraints = NonlinearMPCConstraints(; umin, umax, xmin, xmax)
    solver = OSQPSolver(
        verbose = false,
        max_iter = 1500,
        eps_rel = 1e-10, # solve to high accuracy to compare to known solution
        check_termination = 5,
        sqp_iters = 1,
        dynamics_interval = 10,
    )

    kf = kf1
    # kf = kf2
    for kf in filters
        x0 .= [1,0]
        xr = zeros(nx) # reference state

        prob = QMPCProblem(linsys; observer=kf, Q1,Q2,constraints,N,xr,solver)

        T = 20
        @time hist = MPC.solve(prob; x0, T, verbose = false)#, callback=plot_callback)
        plot(hist) |> display

        cost = MPC.lqr_cost(hist)

        @test dot(x0, QN, x0) ≈ cost rtol=1e-7 # For a linear system with quadratic cost function, the total infinite-horizon cost is known from the solution to the Riccati equation
        if kf isa KalmanFilter
            # @test A*kf.R*A' + R1 ≈ Rinf rtol = 1e-3 # For a linear system with gaussian noise, the infinite-horizon covariance is known from the solution to the Riccati equation the extra lyap step is required since the kf operates in stages and hasn't applied the last update yet
            @test kf.R ≈ Rinf rtol = 1e-3 # This version is more appropriate if the call to correct! comes before optimize
        end
    end

    # With some noise

    kf1     = KalmanFilter(A, B, C, 0, R1, 0.001R2, MvNormal(x0, Rinf))
    filters = [kf1, kf2]

    for kf in filters
        x0 .= [1,0]

        xr = zeros(nx) # reference state
        prob = QMPCProblem(linsys; observer=kf, Q1,Q2,constraints,N,xr,solver)

        T = 20
        @time hist = MPC.solve(prob; x0, T, verbose = false, noise=0.01)#, callback=plot_callback)
        plot(hist) |> display

        cost = MPC.lqr_cost(hist)
        @test cost < 10
        @test norm(hist.X[end]) < 0.1
    end


end


