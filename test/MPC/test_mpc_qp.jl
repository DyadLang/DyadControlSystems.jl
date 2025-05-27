using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.MPC: modelfit
using StaticArrays
using Statistics, LinearAlgebra
using Test
using Plots


function propagate_linearized(f, x0, u; halfway=true)
    N = size(u, 2)
    x = similar(u, size(x0,1), N+1)
    x[:,1] = x0[:, 1]
    for i = 1:N
        xi = x[:, i]
        ui = u[:, i]
        f0 = f(xi, ui, 0, i)
        A,B = MPC.linearize(f, xi, ui, nothing, i)
        x[:,i+1] = A*xi + B*ui #+ f0-xi
        if halfway
            xi = 0.5*xi + 0.5*x[:, i+1]
            A,B = MPC.linearize(f, xi, ui, nothing, i)
            x[:,i+1] = A*x[:, i] + B*ui
        end
    end
    x
end

function plot_callback(x,u,xopt,X,U)
    X,U = reduce(hcat, X), reduce(hcat, U)
    nx, nu = size(x,1), size(u,1)
    past = 1:size(X, 2)
    future = (0:size(u, 2)-1) .+ size(X, 2)
    plot(past, X', layout=2, sp=1, c=(1:nx)', label=["x" "ϕ" "ẋ" "ϕ̇"])
    plot!(past, U', sp=2, c=(1:nu)')
    vline!([last(past)], l=(:dash, :black), primary=false, sp=1)
    vline!([last(past)], l=(:dash, :black), primary=false, sp=2)
    plot!(future, x[:, 1:end-1]', sp=1, c=(1:nx)', l=:dash, label="")
    plot!(future, u', sp=2, c=(1:nu)', l=:dash, label="") #|> display
    if isdefined(Main, :anim)
        frame(anim)
    else
        display(current())
    end
    # sleep(0.2)
end

function sqp_callback(x,u,xopt,t,sqp_iter,prob)
    nx, nu = size(x,1), size(u,1)
    c = MPC.lqr_cost(x,u,prob)
    plot(x[:, 1:end-1]', layout=2, sp=1, c=(1:nx)', label="x nonlinear", title="Time: $t SQP iteration: $sqp_iter")
    plot!(xopt[:, 1:end-1]', sp=1, c=(1:nx)', l=:dash, label="x opt")
    plot!(u', sp=2, c=(1:nu)', label = "u", title="Cost: $(round(c, digits=5))") |> display
end

function XUcost(hist)
    X,E,R,U,Y = reduce(hcat, hist)
    X, U, MPC.lqr_cost(hist)
end


## Nonlinear system

function cartpole(x, u, p, t)
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

nu = 1 # number of controls
nx = 4 # number of states
Ts = 0.02 # sample time
N = 150
x0 = zeros(nx) # Initial state
x0[1] = 3 # cart pos
x0[2] = pi*0.5 # pendulum angle
xr = randn(nx) # reference state
ur = randn(nu)

discrete_dynamics0 = rk4(cartpole, Ts)
x_names = [:p, :θ, :v, :ω]
discrete_dynamics = DyadControlSystems.FunctionSystem(discrete_dynamics0, (x,u,p,t)->x,Ts, x=x_names, u=[:u], y=x_names)
@test discrete_dynamics.z == 1:nx


Q1 = spdiagm(Float64[1, 1, 1, 1]) # state cost matrix
# Q1 = spdiagm([0.1, 1, 0.1, 1]) # state cost matrix
# Q1 = spdiagm([1, 0.1, 1, 0.1]) # state cost matrix
Q2 = Ts * spdiagm(ones(nu)) # control cost matrix

# Control limits
umin = -10 * ones(nu)
umax = 10 * ones(nu)
# State limits (state constraints are soft by default)
xmin = -50 * ones(nx)
xmax = 50 * ones(nx)

vmin = [xmin; umin]
vmax = [xmax; umax]

constrained_outputs = (x,u,p,t)->[x;u]
constraints = NonlinearMPCConstraints(constrained_outputs, vmin, vmax, 1:nx)


# function lqr_sensitivity(discrete_dynamics, xr, Ts, Q1, Q2)
#     A,B = MPC.linearize(discrete_dynamics, xr, [0], 0,0)
#     C = I(4)
#     sys = ss(A,B,C,0, Ts)
#     L = lqr(sys, Matrix(Q1), Matrix(Q2))
#     controller = ss(L, sys.timeevol)
#     loopgain = DyadControlSystems.minreal(controller*sys)
#     sens = DyadControlSystems.minreal(1 / (1 + loopgain))
#     CS = sens * controller
#     if !isstable(sens)
#         @show poles(sens)
#         error("Unstable")
#     end
#     controller, sys, sens, CS
# end
# controller, sys, sens, CS = lqr_sensitivity(discrete_dynamics, xr, Ts, Q1, Q2)
# plot(
#     bodeplot(sens, plotphase=false),
#     bodeplot(CS, plotphase=false),
#     layout=(2, 1),
# )

##

## =============================================================================
# The tests below have not been updated to the new variable layout
## =============================================================================

# @testset "updating of problem matrices" begin
#     @info "Testing updating of problem matrices"
#     N = 4
#     Ts = 1
#     solver = OSQPSolver(
#         eps_rel = 1e-6,
#         max_iter = 15000,
#         check_termination = 5,
#         sqp_iters = 1,
#         dynamics_interval = 1,
#     )
#     prob = QMPCProblem( discrete_dynamics;Q1,Q2,constraints,N,xr,solver)
#     QN = prob.QN

#     u = randn(nu, N)
#     x = zeros(nx, N + 1)
#     x[:, 1] .= x0
#     xo, uo = MPC.rollout!(prob.dynamics, x, copy(u))
#     @test x[:, 2] == discrete_dynamics(x[:, 1], u[:, 1], 0, 0)
#     @test uo == u
#     @test xo == x


#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, 1], u[:, 1], 0, 0)

#     @test length(prob.Ad_pattern) == count(!iszero, Ad)
#     @test length(prob.Bd_pattern) == count(!iszero, Bd)

#     A = deepcopy(prob.A)
#     MPC.update_constraints!(prob, x, u, x0, update_dynamics=true)
#     @test A.colptr == prob.A.colptr
#     @test A.rowval == prob.A.rowval

#     @test prob.A[(5:8) .+ 0, (1:4) .+ 0] == Ad
#     @test prob.A[(5:8) .+ 0, (N+1)*4 .+ (1:1) .+ 0] == Bd
#     @test prob.q[5:8] == -Q1 * xr

#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, 2], u[:, 2], 0, 0)
#     @test prob.A[(5:8) .+ 4, (1:4) .+ 4] == Ad
#     @test prob.A[(5:8) .+ 4, (N+1)*4 .+ (1:1) .+ 1] == Bd

#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, N], u[:, N], 0, 0)
#     @test prob.A[(5:8) .+ (N-1)*nx, (1:4) .+ (N-1)*nx] == Ad
#     @test prob.A[(5:8) .+ (N-1)*nx, (N+1)*4 .+ (1:1) .+ (N-1)] == Bd



#     @test prob.q[(1:nx) .+ N*nx] == -QN * xr
#     uinds = (1:N*nu) .+ (N+1)*nx
#     @test all(iszero, prob.q[uinds])


#     # test that update dynamics updates the matrices correctly
#     u .= randn.()
#     MPC.rollout!(prob.dynamics, x, copy(u))
#     MPC.update_constraints!(prob, x, u, x0, update_dynamics=true)
#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, 1], u[:, 1], 0, 0)

#     @test prob.A[(5:8) .+ 0, (1:4) .+ 0] == Ad
#     @test prob.A[(5:8) .+ 0, (N+1)*4 .+ (1:1) .+ 0] == Bd
#     @test prob.q[5:8] == -Q1 * xr

#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, 2], u[:, 2], 0, 0)
#     @test prob.A[(5:8) .+ 4, (1:4) .+ 4] == Ad
#     @test prob.A[(5:8) .+ 4, (N+1)*4 .+ (1:1) .+ 1] == Bd

#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, N], u[:, N], 0, 0)
#     @test prob.A[(5:8) .+ (N-1)*nx, (1:4) .+ (N-1)*nx] == Ad
#     @test prob.A[(5:8) .+ (N-1)*nx, (N+1)*4 .+ (1:1) .+ (N-1)] == Bd

#     @test prob.q[(1:nx) .+ N*nx] == -QN * xr
#     @test all(iszero, prob.q[uinds])

# end


# @testset "updating of problem matrices SQP" begin
#     @info "Testing updating of problem matrices SQP"
#     N = 4
#     Ts = 1
#     solver = OSQPSolver(
#         eps_rel = 1e-6,
#         max_iter = 15000,
#         check_termination = 5,
#         sqp_iters = 2,
#         dynamics_interval = 1,
#     )
#     prob = LQMPCProblem( discrete_dynamics;Q1,Q2,constraints,N,xr,ur,solver, qs=0, qs2=0)
#     QN = prob.QN

#     u = randn(nu, N)
#     x = zeros(nx, N + 1)
#     x[:, 1] .= x0
#     xo, uo = MPC.rollout!(prob.dynamics, x, copy(u))
#     @test x[:, 2] == discrete_dynamics(x[:, 1], u[:, 1], 0, 0)
#     @test uo == u
#     @test xo == x


#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, 1], u[:, 1], 0, 0)

#     @test length(prob.Ad_pattern) == count(!iszero, Ad)
#     @test length(prob.Bd_pattern) == count(!iszero, Bd)

#     A = deepcopy(prob.A)
#     MPC.update_constraints_sqp!(prob, x, u, x0, update_dynamics=true)
#     MPC.update_xr_sqp!(prob, xr, ur, copy(x), copy(u))
#     @test A.colptr == prob.A.colptr
#     @test A.rowval == prob.A.rowval

#     @test prob.A[(5:8) .+ 0, (1:4) .+ 0] == Ad
#     @test prob.A[(5:8) .+ 0, (N+1)*4 .+ (1:1) .+ 0] == Bd
#     @test prob.q[5:8] ≈ Q1 * (x[:,2] - xr)

#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, 2], u[:, 2], 0, 0)
#     @test prob.A[(5:8) .+ 4, (1:4) .+ 4] == Ad
#     @test prob.A[(5:8) .+ 4, (N+1)*4 .+ (1:1) .+ 1] == Bd

#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, N], u[:, N], 0, 0)
#     @test prob.A[(5:8) .+ (N-1)*nx, (1:4) .+ (N-1)*nx] == Ad
#     @test prob.A[(5:8) .+ (N-1)*nx, (N+1)*4 .+ (1:1) .+ (N-1)] == Bd



#     @test prob.q[(1:nx) .+ N*nx] == QN * (x[:,end] - xr)
#     uinds = (1:N*nu) .+ (N+1)*nx
#     @test prob.q[uinds] ≈ (Q2*(u .- ur))'


#     # test that update dynamics updates the matrices correctly
#     u .= randn.()
#     MPC.rollout!(prob.dynamics, x, copy(u))
#     MPC.update_constraints_sqp!(prob, x, u, x0, update_dynamics=true)
#     MPC.update_xr_sqp!(prob, xr, ur, copy(x), copy(u))
#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, 1], u[:, 1], 0, 0)

#     @test prob.A[(5:8) .+ 0, (1:4) .+ 0] == Ad
#     @test prob.A[(5:8) .+ 0, (N+1)*4 .+ (1:1) .+ 0] == Bd
#     @test prob.q[5:8] ≈ Q1 * (x[:,2] - xr)

#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, 2], u[:, 2], 0, 0)
#     @test prob.A[(5:8) .+ 4, (1:4) .+ 4] == Ad
#     @test prob.A[(5:8) .+ 4, (N+1)*4 .+ (1:1) .+ 1] == Bd

#     Ad, Bd = MPC.linearize(discrete_dynamics, x[:, N], u[:, N], 0, 0)
#     @test prob.A[(5:8) .+ (N-1)*nx, (1:4) .+ (N-1)*nx] == Ad
#     @test prob.A[(5:8) .+ (N-1)*nx, (N+1)*4 .+ (1:1) .+ (N-1)] == Bd

#     @test prob.q[(1:nx) .+ N*nx] ≈ QN * (x[:, end] - xr)
#     @test prob.q[uinds] ≈ (Q2*(u .- ur))'

# end


##

solver = OSQPSolver(
    eps_rel = 1e-10,
    eps_abs = 1e-10,
    max_iter = 25000,
    check_termination = 5,
    sqp_iters = 25, # test with a huge number of iters to make sure the optimized trajectory is close to the nonlinear trajectory
    dynamics_interval = 1,
    verbose=false,
    polish=true, # to get high accuracy
)

## x0 = 0 everything should be 0
@testset "everything zero" begin
    prob = QMPCProblem( discrete_dynamics;Q1,Q2,constraints,N=5,xr=0*xr,solver,qs=0, qs2=0)
    @info "Testing everything zero"
    @time hist = MPC.solve(prob; x0=0*x0, T = 50, verbose = false)
    X, U, cost = XUcost(hist)
    @test all(iszero, X)
    @test all(iszero, U)
    @test cost == 0
end

## x0 = xr != 0
@testset "x0 = xr" begin
    @info "Testing x0 = xr"
    N = 10
    xr2 = Float64[1,0,0,0]
    ur = zeros(nu)
    prob2 = QMPCProblem( discrete_dynamics;Q1,Q2,constraints,N,xr=repeat(xr2, 1, N+1),ur=repeat(ur, 1, N),solver,qs=0, qs2=0)
    # MPC.update_xr!(prob2, xr2)
    
    prob2.x[:, 1] .= xr2
    MPC.rollout!(prob2.dynamics, prob2.x, prob2.u)
    @test prob2.x[:, 1] == xr2
    co = MPC.optimize!(prob2, prob2.x[:, 1], 0, 1, verbose=false)
    xopt, u = co.x, co.u
    @test mean(abs2, u) < 1e-10
    x,u = MPC.rollout!(prob2.dynamics, prob2.x, u)
    @test x[:, 1] ≈ xr2
    xl = propagate_linearized(prob2.dynamics, x, u)
    @test minimum(modelfit(x, xl)) > 90
    @test norm(xopt) < 1e-10 # xopt is the Δ

    # if isinteractive()
    #     plot(xopt', layout=4, lab="Opt")
    #     plot!(xl', lab="Linear")
    #     plot!(x', lab="Nonlinear")
    #     display(current())
    # end


    ## still x0 = xr != 0
    prob = QMPCProblem( discrete_dynamics;Q1,Q2,constraints,N,xr=xr2,ur,solver,qs=0, qs2=0)
    @time hist = MPC.solve(prob; x0=xr2, T = 50, verbose = false)
    X, U, cost = XUcost(hist)
    @test all(abs.(X .- xr2) .< 1e-3)
    @test norm(u) < 1e-10
    @test cost < 1e-16
end



# If we give the natural evolution of the dynamics as the reference for the controller, it should output close to zero control action and follow the reference well.
# NOTE: these tests are not passing atm. it might be related to how the reference trajecotry is forwarded, it adds the last point over and over again from the back. The last point will in general not be a point at which the pendulum system can stay.
@testset "xr = unforced dynamics" begin
    @info "Testing xr = unforced dynamics"
    Ts = 0.01 # sample time
    Q1 = spdiagm(Float64[1e-6,1000,1e-6,1000])
    discrete_dynamics = FunctionSystem(rk4(cartpole, Ts), (x,u,p,t)->x, Ts, x=x_names, u=:u, y=x_names)
    solver = OSQPSolver(
        eps_rel = 1e-8,
        eps_abs = 1e-8,
        max_iter = 5000,
        check_termination = 5,
        sqp_iters = 2, 
        dynamics_interval = 1,
        verbose=false,
        polish=true, # to get high accuracy
    )
    N = 50
    x0 = zeros(nx) # Initial state
    x0[1] = 3 # cart pos
    x0[2] = pi/6 # pendulum angle 

    u = zeros(nu, 10N)
    xr,ur = MPC.rollout(discrete_dynamics, x0, u)
    xl = propagate_linearized(discrete_dynamics, x0, u, halfway=false)
    prob = QMPCProblem(discrete_dynamics; Q1,Q2,constraints,N,xr,ur,solver,qs=0, qs2=0)
    
    T = 50
    @time hist = MPC.solve(prob; x0, T, verbose = false)
    X, U, cost = XUcost(hist)
    @test cost < 1e-6

    if isinteractive()
        f1 = plot(X', layout=4, lab="Opt")
        plot!(xl', lab="Linear")
        plot!(xr', lab="Ref")
        plot(f1, plot(U', title="Control. Cost: $cost"))
        display(current())
    end

    @test sum(abs2, U) < 0.1
    @test minimum(modelfit(xr[:, 1:T+1], X)) > 95

    # less strict versions of the two tests above
    @test sum(abs2, U) < 2
    @test minimum(modelfit(xr[:, 1:T+1], X)) > 85
    # @test minimum(modelfit(X, xopt)) > 95 # the trajectory in the optimizer should be almost exactly the same as xl, subject to constraint tolerances

    @show merr = maximum(abs.(X[:, 1:T] .- xr[:, 1:T]))
    @test merr < 0.05
    # @test all(abs.(U[:, 1:T] .- u[:, 1:T]) .< 1e-3)
    # @test cost < 1e-16
end


@testset "SQP iterations" begin
    @info "Testing SQP iterations"

    N = 20
    x0 = zeros(nx) # Initial state
    x0[1] = 3 # cart pos
    x0[2] = pi*0.5 # pendulum angle (if this is set a bit higher, SQP iterations fail to improve the cost
    xr = zeros(nx)

    ## test normal operation with moderate accuracy settings
    solver = OSQPSolver(
        eps_rel = 1e-6,
        max_iter = 25000,
        check_termination = 5,
        sqp_iters = 1,
        dynamics_interval = 2, # this should be > 1 to compare with SQP iters below
        verbose=false,
        polish=true,
    )

    prob = QMPCProblem( discrete_dynamics;Q1,Q2,constraints,N,xr,solver,qs=0,qs2=0)
    @time hist = MPC.solve(prob; x0, T = 250, verbose = false, lqr_init=false)#, sqp_callback)
    X, U, cost0 = XUcost(hist)
    isinteractive() && plot(hist) |> display
    @test X[:, end] ≈ xr atol = 1e-1
    @test cost0 ≈ 1058.6 rtol = 1e-2

    ## Test SQP iterations, they should improve the cost
    # sanity check, set dynamics_interval=1 above and see that it produces the same results as for sqp_iters = 1 here (with the same α). A higher number of SQP iterations should improve the cost.
    sqp_iters = 10
    sqp_iters = [1, 2, 3]
    costs = map(sqp_iters) do sqp_iters
        solver = OSQPSolver(;
            eps_rel = 1e-6,
            eps_abs = 1e-6,
            max_iter = 25000,
            check_termination = 5,
            sqp_iters,
            dynamics_interval = 1,
            verbose=false,
            polish=true,
        )
        prob = QMPCProblem( discrete_dynamics;Q1,Q2,constraints,N,xr,solver,qs=0,qs2=0)
        @time hist = MPC.solve(prob; x0, T = 250, verbose = false, lqr_init=false)#, sqp_callback)#, callback=plot_callback)
        cost = MPC.lqr_cost(hist)
        cost
    end
    if isinteractive()
        plot(sqp_iters, costs)
        hline!([cost0], label="Cost with slow update of dynamics")
        display(current())
    end
    @test all(diff(costs) .< 0) # cost should be decreasing with the number of SQP iterations
    @test all(costs .<= cost0) # cost should be lower when dynamics is updated every iteration
end



## test updating xr to new goal point
@testset "update xr" begin
    @info "Testing update xr"

    N = 150
    xr = zeros(nx)
    ur = zeros(nu)
    solver = OSQPSolver()
    vmin = [xmin; umin]
    vmax = [xmax; umax]

    constrained_outputs = (x,u,p,t)->[x;u]
    constraints = NonlinearMPCConstraints(constrained_outputs, vmin, vmax, 1:nx)

    prob = QMPCProblem( discrete_dynamics;Q1,Q2,constraints,N,xr,solver,qs=0,qs2=0)
    q = deepcopy(prob.q)
    xr = [1, 0, 0, 0]
    MPC.update_xr!(prob, xr, ur, prob.u[:,1])
    @test prob.q != q
    @time hist = MPC.solve(prob; x0, T = 500, verbose = false)
    X, U, cost = XUcost(hist)
    isinteractive() && plot(hist) |> display
    @test X[:, end] ≈ xr atol=1e-1

    ## test xr reference trajectory
    N = 400
    xr = zeros(nx, N+1)
    ur = zeros(nu, N)
    xr[1, 1:390] .= 0
    xr[1, 391:end] .= 1
    solver = OSQPSolver(
        eps_rel = 1e-6,
        max_iter = 15000,
        check_termination = 5,
        sqp_iters = 1,
        dynamics_interval = 1,
        polish=true,
    )

    vmin = [xmin; 2umin]
    vmax = [xmax; 2umax]

    constrained_outputs = (x,u,p,t)->[x;u]
    constraints = NonlinearMPCConstraints(constrained_outputs, vmin, vmax, 1:nx)

    prob = QMPCProblem(discrete_dynamics;
        Q1 = spdiagm([10.0, 1, 1, 1]),
        Q2 = 0.001Q2,
        # allow more force to limit smoothing of traj at reference step
        constraints,
        N,    xr=copy(xr), ur=copy(ur),
        solver,qs=0,qs2=0
    )

    @time hist = MPC.solve(prob; x0, T = 800, verbose = false)#, callback=plot_callback)
    X, U, cost = XUcost(hist)
    isinteractive() && plot(hist) |> display
    ##

    @test X[:, end-1] ≈ xr[:, end-1] atol=1e-1
    @test X[:, 300] ≈ xr[:, 300] atol=1e-1

end

## Make it difficult by setting the initial state to a more difficult position
@testset "difficult starting point" begin
    @info "Testing difficult starting point"

    N = 60
    x0 = zeros(nx)
    x0[1] = 3 # cart pos
    x0[2] = 0.9pi # pendulum angle
    xr = zeros(nx) # reference state
    ur = zeros(nu)

    solver = OSQPSolver(
        verbose = false,
        eps_rel = 1e-8,
        eps_abs = 1e-8,
        max_iter = 150000,
        check_termination = 5,
        sqp_iters = 3,
        dynamics_interval = 1,
        polish=true,
    )
    vmin = umin
    vmax = umax

    constrained_outputs = (x,u,p,t)->u
    constraints = NonlinearMPCConstraints(constrained_outputs, vmin, vmax, 1:0)
    prob = QMPCProblem( discrete_dynamics;Q1,Q2=Q2,constraints,N,xr,ur,solver,qs=0,qs2=0)


    @time hist = MPC.solve(prob; x0, T = 500, verbose = false)#, callback=plot_callback)
    X, U, cost = XUcost(hist)
    isinteractive() && plot(plot(X', title="difficult starting point"), plot(U', title="Cost = $(round(cost, digits=3))")) |> display
    @test X[:, end] ≈ xr atol = 1e-2
    @test cost ≈ 1928 rtol=1e-2 

end

@testset "control derivative penalty" begin
    @info "Testing control derivative penalty"

    N = 100
    ## Add control derivative penalty
    x0 = zeros(nx) # Initial state
    x0[1] = 3 # cart pos
    x0[2] = pi*0.5 # pendulum angle
    xr = zeros(nx) # reference state
    solver = OSQPSolver(
        eps_rel = 1e-3,
        max_iter = 500,
        check_termination = 5,
        sqp_iters = 1,
        dynamics_interval = 2,
    )
    prob = QMPCProblem( discrete_dynamics;Q1,Q2,constraints,N,xr,solver)
    @time hist = MPC.solve(prob; x0, T = 500, verbose = false)
    X, U, cost = XUcost(hist)

    prob3 = QMPCProblem(discrete_dynamics;
        Q1,    Q2,
        Q3 = 10Q2,
        constraints,
        N,    xr,
        solver
    )
    @time hist3 = MPC.solve(prob3; x0, T = 500, verbose = false)
    X3, U3, cost3 = XUcost(hist3)
    # @test X[:, end] ≈ xr[:,end] atol = 2e-2
    # @test cost ≈ 1.3773325799667657 atol = 1e-2
    if isinteractive()
        plot(X', sp=1, layout=(1,2))
        plot!(U', label="Cost = $(round(cost, digits=3))", sp=2)
        plot!(X3', sp=1, c=(1:4)', l=:dash)
        plot!(U3', label="Cost = $(round(cost3, digits=3))", sp=2)
        display(current())
    end

    @test sum(abs2, diff(U3[:])) < sum(abs2, diff(U[:])) # test that Q3 indeed lowered the variance in the control derivative

    @test cost3 < cost + dot(U,prob3.Q3,U) # this is the added cost due to control derivative

end

@testset "Short horizon" begin
    @info "Testing Short horizon"
    N = 25
    x0[1] = 3 # cart pos
    x0[2] = 0.9pi # pendulum angle
    xr = zeros(nx) # reference state

    solver = OSQPSolver(
        verbose = false,
        eps_rel = 1e-3,
        max_iter = 5000,
        check_termination = 5,
        sqp_iters = 20,
        dynamics_interval = 1,
    )
    prob = QMPCProblem( discrete_dynamics;Q1,Q2,constraints,N,xr,solver,qs=0,qs2=0)

    @time hist = MPC.solve(prob; x0, T = 300, verbose = false)
    X, U, cost = XUcost(hist)
    isinteractive() && plot(hist) |> display
    @test X[:, end] ≈ xr atol = 5e-2
    @show isapprox(cost, 1995, rtol=1e-1)
    @test_skip cost ≈ 1995 rtol=1e-1 # This test should pass and used to do https://github.com/JuliaComputing/DyadControlSystems.jl/issues/156
end


## test copyto!
@testset "copyto" begin
    @info "Testing copyto"
    # test that copyto with aliased src and dst works as expected. It's not documented and if it changes we must be aware of it since thsi is used a lot in the MPC toolbox.
    N = 10
    nx = 2
    x0 = randn(nx, N+1)
    x1 = copy(x0)
    copyto!(x1, 1, x1, nx+1, N*nx)
    x2 = copy(x0)
    @views x2[:, 1:end-1] .= x2[:, 2:end]
    @test x1[:,1] == x0[:,2]
    @test x1 == x2
end

## test if the prediction by the solver is close to the true evolution. We evaluate in a narrow regime around the downward equilibrium in such a way that the system is approximately linear
@testset "solver dynamics linear" begin
    @info "Testing solver dynamics linear"

    solver = OSQPSolver(;
        eps_rel = 1e-10,
        eps_abs = 1e-10,
        max_iter = 5000,
        check_termination = 5,
        sqp_iters = 1,
        dynamics_interval = 1,
        polish=true,
    )
    prob = QMPCProblem( discrete_dynamics;Q1,Q2,constraints,N,xr=0*xr,solver)

    x0 = [1,0,0,0]
    prob.x[:, 1] .= x0
    MPC.rollout!(prob)

    errors = Float64[]
    costs = Float64[]
    i = 1
    for i = 1:5
        u = prob.u
        x = prob.x
        Ad, Bd = MPC.linearize(discrete_dynamics, x0, u[:, 1], 1, 0)
        co = MPC.optimize!(prob, x, 0, 1)
        xopt, u = co.x, co.u
        MPC.rollout!(prob.dynamics, x, u)
        @test minimum(modelfit(x, xopt)) > 97
        
        @test xopt[:, 1] ≈ x0 rtol=1e-3
        @test xopt[:, 2] ≈ Ad*xopt[:, 1] + Bd*u[:,1] rtol=1e-3

        err = sum(abs2, x-xopt) / sum(abs2, x)
        push!(errors, err)
        push!(costs, MPC.lqr_cost(x,u,prob))
        plot(x'); plot!(xopt', c=(1:4)', l=:dash, title=err) |> display
    end
    isinteractive() && plot([errors costs], layout=2, title=["Traj error" "LQR cost"]) |> display

    @test all(errors .< 1e-4)
    @test all(diff(costs) .< 1e-4)
end


## Testing soft constraints
# The test limits the velocity of the cart (x[3]) to be greater than -2

@testset "Soft constraints" begin
    @info "Testing Soft constraints"

    N = 20
    x0 = zeros(nx)
    x0[1] = 3 # cart pos
    x0[2] = 0.5pi # pendulum angle
    xr = zeros(nx) # reference state

    solver = OSQPSolver(
        verbose = false,
        eps_rel = 1e-7,
        eps_abs = 1e-7,
        max_iter = 1500000,
        check_termination = 5,
        sqp_iters = 1,
        dynamics_interval = 1,
        polish=true,
    )

    xmax = fill(10, nx)
    xmin = [-10, -10, -2, -10]
    vmin = [xmin; umin]
    vmax = [xmax; umax]

    constrained_outputs = (x,u,p,t)->[x;u]
    constraints = NonlinearMPCConstraints(constrained_outputs, vmin, vmax, 1:nx)
    qs = 100
    qs2 = 1000
    ##
    prob = QMPCProblem( discrete_dynamics;Q1,Q2,qs,qs2,constraints,N,xr,solver)

    @test prob.qs == qs
    @test prob.qs2 == qs2

    slackuinds = [repeat([falses(nx+nu); trues(nx); falses(nx)], N); falses(nx)]
    slacklinds = [repeat([falses(nx+nu); falses(nx); trues(nx)], N); falses(nx)]
    nS = N*(2nx)
    # @test all(prob.ub[slackuinds] .== Inf)
    # @test all(prob.lb[slacklinds] .== 0)
    @test all(prob.q[slackuinds .| slacklinds] .== qs)
    @test prob.P[slackuinds .| slacklinds, slackuinds .| slacklinds] == qs2*I
    @test prob.A[end-nS+1:end, slackuinds .| slacklinds] == I


    # Tests below deactivated after variable layout change
    # Aieq = prob.A[(N+1)*nx+1:end, :]
    # nX = (N+1)*nx
    # nU = (N)*nu
    # nS = 2nX

    # @test size(Aieq) == (2nX + nU + nS, nX + nU + nS)

    # @test Aieq[1:nX, 1:nX] == I
    # @test Aieq[(1:nX) .+ nX, 1:nX] == I

    # suinds = (1:nX) .+ (nX+nU)
    # @test Aieq[1:nX, suinds] == -I
    # slinds = (1:nX) .+ (2nX+nU)
    # @test Aieq[(1:nX) .+ nX, slinds] == I

    # uinds = (1:nU) .+ nX
    # @test Aieq[2nX .+ uinds, nX .+ uinds] == I




    @time hist = MPC.solve(prob; x0, T = 100, verbose = false)#, callback=plot_callback)
    isinteractive() && plot(hist)
    ##
    x3 = [x[3] for x in hist.X]
    # plot(x3)
    @test minimum(x3 .+ 2) > -0.03 # The constraint is soft, so we have a high tolerance

    ## No state constraints
    vmin = umin
    vmax = umax

    constrained_outputs = (x,u,p,t)->u
    constraints = NonlinearMPCConstraints(constrained_outputs, vmin, vmax, Int[])
    prob = QMPCProblem( discrete_dynamics;Q1,Q2,qs,qs2,constraints,N,xr,solver)

    @test prob.qs == 0
    @test prob.qs2 == 0

    @time hist = MPC.solve(prob; x0, T = 100, verbose = false)#, callback=plot_callback)
    isinteractive() && plot(hist)
    @test MPC.lqr_cost(hist) ≈ 1058.3770 rtol=1e-2


    ## Output selection by z
    
end

# z = [1]
# xr = [0]
# qs = 100
# qs2 = 1000
# Q1 = Diagonal([1])
# prob = QMPCProblem( discrete_dynamics;z,Q1,Q2,qs,qs2,constraints,N,xr,solver)
# @time hist = MPC.solve(prob; x0, T = 100, verbose = true)#, callback=plot_callback)
# isinteractive() && plot(hist)

# @error("
# TODO:
# - xr används för att linearisera, om xr är en referens för z så vet vi inte var vi ska linearisera
# ")

##

## Make animation
# anim = Animation()
# N = 400
# xr = zeros(nx, N+1)
# xr[1, 1:390] .= 0
# xr[1, 391:end] .= 1
# solver = OSQPSolver(
#     eps_rel = 1e-3,
#     max_iter = 15000,
#     check_termination = 5,
#     sqp_iters = 1,
#     dynamics_interval = 1,
#     polish=false,
# )
# prob = QMPCProblem(;
#     dynamics = discrete_dynamics,
#     Q1 = spdiagm([10.0, 1, 1, 5]),
#     Q2 = 0.01Q2,
#     Q3 = Q2,
#     umin = 0.5umin, # allow more force to limit smoothing of traj at reference step
#     umax = 0.5umax,
#     xmin,    xmax,
#     N,    xr,
#     solver,
# )

# @time X, U, cost = MPC.solve(prob; x0, T = 600, verbose = false, callback=plot_callback)
# gif(anim, joinpath(@__DIR__(), "MPC_animation3.gif"), fps=60)
# @time MPC.solve(prob; x0, T = 600, verbose = false)
# @time X, U, cost = MPC.solve(prob; x0, T = 600, verbose = false)
# plot(plot(X', title="States"), plot(U', title="Control input. Cost = $(round(cost, digits=3))"))
