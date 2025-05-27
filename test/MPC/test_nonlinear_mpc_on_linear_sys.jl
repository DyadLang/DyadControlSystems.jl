
using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.MPC: modelfit
using StaticArrays
using ControlSystems: Discrete
using Statistics, LinearAlgebra
using Test
using Plots


## Test behavior on linear system
function propagate_linearized(f, x0, u; halfway=true)
    N = size(u, 2)
    x = similar(u, size(x0,1), N+1)
    x[:,1] = x0[:, 1]
    for i = 1:N
        xi = x[:, i]
        A,B = MPC.linearize(f, xi, u[:, i], nothing, i)
        x[:,i+1] = A*x[:, i] + B*u[:, i]
        if halfway
            xi = 0.5*x[:, i] + 0.5*x[:, i+1]
            A,B = MPC.linearize(f, xi, u[:, i], nothing, i)
            x[:,i+1] = A*x[:, i] + B*u[:, i]
        end
    end
    x
end
function linsys(x, u, _, _)
    h = 1
    # A = [0 0; 0 1]
    Ad = [1 h; 0 1]
    B = [0; 1;;]
    # C = [1 0]
    # @show size.((Ad,B,x,u))
    Ad*x + B*u
end
function XUcost(hist)
    X,E,R,U,Y = reduce(hcat, hist)
    X, U, MPC.lqr_cost(hist)
end

@testset "NMPC on Linear system" begin
    @info "Testing NMPC on Linear system"

    nx = 2
    nu = 1
    Ts = 1
    N = 10
    x0 = [1.0, 0]
    xr = zeros(nx) # reference state
    # Control limits
    vmin = -100 * ones(nu)
    vmax = 100 * ones(nu)
    # constrained_outputs = (x,u,p,t)->u
    # constraints = NonlinearMPCConstraints(constrained_outputs, vmin, vmax, Int[])

    constraints = NonlinearMPCConstraints(umin=vmin, umax=vmax)

    dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x^2, u=[:u], y=:y^2)
    solver = OSQPSolver(
        verbose = false,
        eps_rel = 1e-10, # solve to high accuracy to compare to known solution
        max_iter = 1500,
        check_termination = 5,
        sqp_iters = 1,
        dynamics_interval = 1,
        polish = true,
    )
    T = 20
    Q2 = spdiagm(ones(nu)) # control cost matrix
    qfactor = 1
    for qfactor in [0.1, 1, 10]
        Q1 = qfactor*spdiagm(ones(nx)) # state cost matrix

        prob = QMPCProblem(dynamics;Q1,Q2,constraints,N,xr,solver)

        @time hist = MPC.solve(prob; x0, T, verbose = false)#, callback=plot_callback)
        X, U, cost = XUcost(hist)
        plot(hist) |> display

        @test X ≈ MPC.rollout(dynamics, x0, U)[1]

        QN, Ad, Bd = MPC.calc_QN_AB(Q1, Q2, 0Q2, dynamics, xr)
        L = ControlSystemsBase.lqr(Discrete, Matrix(Ad), Matrix(Bd), Matrix(Q1), Matrix(Q2))

        @test X[:, 1] == x0
        @test U[:, 1] ≈ -L*x0 rtol=1e-5
        @test X[:, 2] == Ad*x0 + Bd*U[:, 1]
        @test X[:, 3] == Ad*X[:,2] + Bd*U[:, 2]
        @test U[:, 2] ≈ -L*X[:, 2] rtol=1e-3
        @test U[:, 3] ≈ -L*X[:, 3] rtol=1e-3


        @test dot(x0, QN, x0) ≈ cost rtol=1e-7 # For a linear system with quadratic cost function, the total infinite-horizon cost is known from the solution to the Riccati equation

        # Test the LQR rollout function
        xlqr, ulqr = MPC.lqr_rollout!(prob, x0)
        if isinteractive()
            plot(xlqr', lab="LQR", layout=2)
            plot!(X', lab="MPC")
            display(current())
        end
        @test xlqr ≈ X[:, 1:size(xlqr, 2)]

        # test that the optimizer outputs state variables matching the full linear propagation
        prob.u .= randn.()
        co = MPC.optimize!(prob, x0, 0, 1, verbose=false); xopt, u = co.x, co.u
        x,u = MPC.rollout!(prob.dynamics, prob.x, u)
        xl = propagate_linearized(prob.dynamics, x, u)
        @test xopt ≈ xl
    end
    ## Test with derivative penalty
    Q1 = spdiagm(ones(nx)) # state cost matrix
    prob = QMPCProblem(dynamics; Q1 = 100Q1 ,Q2 = 1*Q2, Q3=10000Q2,constraints,N,xr,solver)
    hist = MPC.solve(prob; x0, T, verbose = false)#, callback=plot_callback)
    X, U, cost = XUcost(hist)
    ##
    P3 = ss([1 1; 0 1], [0;1;;], I(2), 0, 1)
    QN3 = dare3(P3, Matrix(prob.Q1),Matrix(prob.Q2),Matrix(prob.Q3))
    # QN3 = [308.25689415814384 270.98311690105027; 270.98311690105027 531.43435572401]
    xN = X[:, end]
    @show dot(xN, QN3, xN)
    acost = cost + dot(xN, QN3, xN)
    @show abs(dot(x0, QN3, x0) - acost)/acost
    @test dot(x0, QN3, x0) ≈ acost rtol = 1e-4
end


@testset "Linear system with SQP iterations" begin
    @info "Testing Linear system with SQP iterations"

    nx = 2
    nu = 1
    Ts = 1
    N = 10
    x0 = [1.0, 0]
    xr = zeros(nx) # reference state
    # Control limits
    vmin = -100 * ones(nu)
    vmax = 100 * ones(nu)
    constrained_outputs = (x,u,p,t)->u
    constraints = NonlinearMPCConstraints(constrained_outputs, vmin, vmax, Int[])
    dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x^2, u=[:u], y=:y^2)

    solver = OSQPSolver(
        verbose = false,
        eps_rel = 1e-10, # solve to high accuracy to compare to known solution
        max_iter = 1500,
        check_termination = 5,
        sqp_iters = 50,
        dynamics_interval = 1,
        polish = true,
    )
    T = 20
    Q2 = spdiagm(ones(nu)) # control cost matrix
    qfactor = 1
    for qfactor in [0.1, 1, 10]
        Q1 = qfactor*spdiagm(ones(nx)) # state cost matrix

        prob = QMPCProblem(dynamics;Q1,Q2,constraints,N,xr,solver)

        @time hist = MPC.solve(prob; x0, T, verbose = false)#, callback=plot_callback)
        X, U, cost = XUcost(hist)
        plot(hist) |> display

        @test X ≈ MPC.rollout(dynamics, x0, U)[1]

        QN, Ad, Bd = MPC.calc_QN_AB(Q1, Q2, 0Q2, dynamics, xr)
        L = ControlSystemsBase.lqr(Discrete, Matrix(Ad), Matrix(Bd), Matrix(Q1), Matrix(Q2))

        @test X[:, 1] == x0
        @test U[:, 1] ≈ -L*x0 rtol=1e-5
        @test X[:, 2] == Ad*x0 + Bd*U[:, 1]
        @test X[:, 3] == Ad*X[:,2] + Bd*U[:, 2]
        @test U[:, 2] ≈ -L*X[:, 2] rtol=1e-3
        @test U[:, 3] ≈ -L*X[:, 3] rtol=1e-3


        @test dot(x0, QN, x0) ≈ cost rtol=1e-7 # For a linear system with quadratic cost function, the total infinite-horizon cost is known from the solution to the Riccati equation


        # Broken tests from above with less accuracy
        @test U[:, 1] ≈ -L*x0 rtol=5e-1
        @test U[:, 2] ≈ -L*X[:, 2] rtol=3e-1
        @test U[:, 3] ≈ -L*X[:, 3] rtol=3e-1
        @test dot(x0, QN, x0) ≈ cost rtol=3e-1 # For a linear system with quadratic cost 



        # Test the LQR rollout function
        xlqr, ulqr = MPC.lqr_rollout!(prob, x0)
        if isinteractive()
            plot(xlqr', lab="LQR", layout=2)
            plot!(X', lab="MPC")
            display(current())
        end
        @test xlqr ≈ X[:, 1:size(xlqr, 2)] atol=0.1

        # test that the optimizer outputs state variables matching the full linear propagation
        prob.u .= randn.()
        co = MPC.optimize!(prob, x0, 0, 1, verbose=false); xopt, u = co.x, co.u
        x,u = MPC.rollout!(prob.dynamics, prob.x, u)
        xl = propagate_linearized(prob.dynamics, x, u)
        @test norm(xopt) < 1e-8
    end
    ## Test with derivative penalty
    @test_skip begin
        Q1 = spdiagm(ones(nx)) # state cost matrix
        prob = QMPCProblem(dynamics; Q1 = 100Q1 ,Q2 = 1*Q2, Q3=10000Q2,constraints,N,xr,solver)
        hist = MPC.solve(prob; x0, T, verbose = false)#, callback=plot_callback)
        X, U, cost = XUcost(hist)
        ##
        P3 = ss([1 1; 0 1], [0;1;;], I(2), 0, 1)
        QN3 = dare3(P3, Matrix(prob.Q1),Matrix(prob.Q2),Matrix(prob.Q3))
        # QN3 = [308.25689415814384 270.98311690105027; 270.98311690105027 531.43435572401]
        xN = X[:, end]
        @show dot(xN, QN3, xN)
        acost = cost + dot(xN, QN3, xN)
        @show abs(dot(x0, QN3, x0) - acost)/acost
        @test dot(x0, QN3, x0) ≈ acost rtol = 1e-4
    end
end



