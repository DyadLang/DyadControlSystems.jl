# ENV["JULIA_DEBUG"] = ""
#=
This example implements an MPC controller for the cartpole system. The continuous dynamics are discretized with RK4 and a quadratic cost function is optimized using Optim.LBFGS
This is a very naive approach, but gives surprisingly okay performance. 
There are no contraints on control signals or states.
=#
using DyadControlSystems
using DyadControlSystems.MPC
using LinearAlgebra, Plots
using StaticArrays
using Test

function linsys(x, u, _, _)
    h = 1
    # A = [0 0; 0 1]
    Ad = SA[1 h; 0 1]
    B = SA[0; 1;;]
    # C = [1 0]
    # @show size.((Ad,B,x,u))
    Ad*x + B*u
end
function XUcost(hist,Q1,Q2,Q3)
    X,E,R,U,Y = reduce(hcat, hist)
    X, U, MPC.lqr_cost(X,U,Q1,Q2,Q3)
end



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

dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x^2, u=[:u], y=:y^2)
ny = dynamics.ny
observer = StateFeedback(dynamics, x0, nu, ny)
T = 20
Q2 = spdiagm(ones(nu)) # control cost matrix
Q3 = 10Q2
qfactor = 1
t = 1
for qfactor in [0.1, 1, 10]
    global running_cost, terminal_cost, QN, oi
    Q1 = qfactor*Diagonal(ones(nx)) # state cost matrix
    QN, Ad, Bd = MPC.calc_QN_AB(Q1, Q2, 0Q2, dynamics, xr)

    p = (; Q1, Q2, Q3, QN)

    running_cost = StageCost() do si, p, t
        Q1, Q2 = p.Q1, p.Q2
        e = (si.x) #.- value(si.r)
        u = (si.u)
        dot(e, Q1, e) + dot(u, Q2, u)
    end



    terminal_cost = TerminalCost() do ti, p, t
        e = (ti.x) #.- value(ti.r)
        dot(e, p.QN, e) 
    end

    objective = Objective(running_cost, terminal_cost)

    x = zeros(nx, N+1)
    u = zeros(nu, N)

    x, u = MPC.rollout(dynamics, x0, u, p, t)

    oi = ObjectiveInput(x, u, xr)


    prob = GenericMPCProblem(
        dynamics;
        N,
        observer,
        objective,
        p,
        objective_input = oi,
        xr,
        # jacobian_method = :symbolics,
        # gradient_method = :symbolics,
        presolve = true,
    );

    x_, u_ = get_xu(prob.vars)



    @time hist = MPC.solve(prob; x0, T = 100, verbose = false); # solve for T=500 time steps
    # 0.38
    X, U, cost = XUcost(hist,Q1,Q2,0Q3)
    oi2 = MPC.remake(oi, x=X, u=U, discretization=prob.constraints.constraints[2])
    @test MPC.evaluate(objective, oi2, p, 0) ≈ cost
    plot(hist) |> display

    @test X ≈ MPC.rollout(dynamics, x0, U)[1]

    
    L = ControlSystemsBase.lqr(Discrete, Matrix(Ad), Matrix(Bd), Matrix(Q1), Matrix(Q2))

    @test X[:, 1] == x0
    @test U[:, 1] ≈ -L*x0 rtol=1e-4
    @test X[:, 2] == Ad*x0 + Bd*U[:, 1]
    @test X[:, 3] == Ad*X[:,2] + Bd*U[:, 2]
    @test U[:, 2] ≈ -L*X[:, 2] rtol=1e-3
    @test U[:, 3] ≈ -L*X[:, 3] rtol=1e-3


    @test dot(x0, QN, x0) ≈ cost rtol=1e-6 # For a linear system with quadratic cost function, the total infinite-horizon cost is known from the solution to the Riccati equation

end
## Test with derivative penalty
Q1 = spdiagm(ones(nx)) # state cost matrix
p = (; Q1, Q2, Q3, QN)
difference_cost = DifferenceCost((si,p,t)->si.u) do e, p, t
    dot(e, p.Q3, e)
end
objective = Objective(running_cost, terminal_cost, difference_cost)
prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    p,
    objective_input = oi,
    xr,
    presolve = true,
);
hist = MPC.solve(prob; x0, T, verbose = false)
X, U, cost = XUcost(hist,Q1,Q2,Q3)
##
P3 = ss([1 1; 0 1], [0;1;;], LinearAlgebra.I(2), 0, 1)
QN3 = dare3(P3, Matrix(Q1),Matrix(Q2),Matrix(Q3))
xN = X[:, end]
@show dot(xN, QN3, xN)
acost = cost + dot(xN, QN3, xN)
@show abs(dot(x0, QN3, x0) - acost)/acost
cost_u0 = dot(U[:,1], Q3, U[:,1])
@test dot(x0, QN3, x0) ≈ acost rtol = 1e-6
plot(hist)

## Test with input integrators

@testset "input integration" begin
    let
        nu = 4
        u = randn(nu, 10)
        u0 = randn(nu)
        o = similar(u)
        MPC.integrate_inputs!(o, u, u0, [1, 3])
        
        @test o[1, :] ≈ cumsum(u[1, :]) .+ u0[1]
        @test o[2, :] ≈ u[2, :]
        @test o[3, :] ≈ cumsum(u[3, :]) .+ u0[3]
        @test o[4, :] ≈ u[4, :]
    end
end
    



dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x^2, u=[:u], y=:y^2)
dynamics = DyadControlSystems.add_input_integrators(dynamics)
(; nx, nu) = dynamics

x0 = [1.0; 0; 0]
x = zeros(nx, N+1)
u = zeros(nu, N)
x, u = MPC.rollout(dynamics, x0, u, p, t)
xr = zeros(nx)
oi = ObjectiveInput(x, u, xr)

Q1 = cat(spdiagm(ones(2)), Q2, dims=(1,2)) # state cost matrix
QN, Ad, Bd = MPC.calc_QN_AB(Q1, Q3, 0Q2, dynamics, xr)
p = (; Q1, Q2=Q3, Q3, QN) # our input is now the control difference
objective = Objective(running_cost, terminal_cost)


observer = StateFeedback(dynamics, x0)

prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    p,
    objective_input = oi,
    xr,
    presolve = true,
    verbose = false,
);
histii = MPC.solve(prob; x0, T, verbose = false)
X, U, cost = XUcost(histii,Q1,Q2,Q3)
plot(histii)
plot!(0:length(U)-1, cumsum(U[:]), sp=2)


@test cumsum(U[:]) ≈ first.(hist.U) atol=1e-3
@test X[3, 2:end-1] ≈ cumsum(U[1,1:end-1])

# plot([X[3, 2:end-1] cumsum(U[1,1:end-1])])


## Test continuous integrators by using rk4
function linsys(x, u, _, _)
    Ad = [0 1; 0 0]
    B = [0; 1;;]
    Ad*x + B*u
end
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, x=:x^2, u=[:u], y=:y^2)
dynamics = DyadControlSystems.add_input_integrators(dynamics)
dynamics = MPC.rk4(dynamics, 1)
observer = StateFeedback(dynamics, x0)


prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    p,
    objective_input = oi,
    xr,
    presolve = true,
    verbose = true,
);
histii2 = MPC.solve(prob; x0, T, verbose = false)
X2, U2, cost2 = XUcost(histii2,Q1,Q2,Q3)
plot!(histii2, l=:dash)
# plot!(0:length(U2)-1, cumsum(U2[:]), sp=2)

@test X2[[1,3], :] ≈ X[[1,3], :] rtol=3e-2 # The velocity is a bit more off
@test cumsum(U2[:]) ≈ first.(hist.U) atol=1e-2
@test X2[3, 2:end-1] ≈ cumsum(U2[1,1:end-1])


# ==============================================================================
## Test manual stepping while adding references
# ==============================================================================
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, x=:x^2, u=[:u], y=:y^2)
dynamics = MPC.rk4(dynamics, 1)
nx = 2
N = 10
xr = zeros(nx, N+1) # initial reference state
x0 = Float64[1,0]
Q1 = 10Diagonal(ones(nx)) # state cost matrix
Q2 = Diagonal(ones(nu)) # control cost matrix
QN, Ad, Bd = MPC.calc_QN_AB(Q1, Q2, 0Q2, dynamics, xr)

running_cost = StageCost() do si, p, t
    Q1, Q2 = p.Q1, p.Q2
    e = (si.x) .- (si.r)
    u = (si.u)
    dot(e, Q1, e) + dot(u, Q2, u)
end

terminal_cost = TerminalCost() do ti, p, t
    e = (ti.x) .- (ti.r)
    dot(e, p.QN, e) 
end

objective = Objective(running_cost, terminal_cost)

x = zeros(nx, N+1)
u = zeros(nu, N)
x, u = MPC.rollout(dynamics, x0, u, p, t)
oi = ObjectiveInput(x, u, xr)
p = (; Q1, Q2, QN)
observer = StateFeedback(dynamics, x0)
prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    p,
    objective_input = oi,
    xr,
    presolve = true,
);

i = 1
X = []
U = []
R = Float64[]
u0 = [0.0]
for i = 0:30
    global u0
    r_new = Float64[i % 15, 0]
    controllerinput = ControllerInput(MPC.state(prob.observer), r_new, [], u0)
    controlleroutput = optimize!(prob, controllerinput, p, 1);
    u0 = MPC.get_first(controlleroutput.u)
    push!(X, copy(MPC.state(prob.observer)))
    push!(U, copy(MPC.get_first(oi.u)))
    push!(R, prob.xr[1])
    MPC.mpc_observer_predict!(prob.observer, u0, r_new, [], t)
end
X = reduce(hcat, X)
U = reduce(hcat, U)
# plot([X; U; R']', layout=3)
@test norm(X[1,:] - R) < 8

# ==============================================================================
## Test integral action property throuh add_input_integrators for a system without integrators
# ==============================================================================
function linsys(x, u, _, _)
    Ad = [0.3679;;]
    B = [0.6321;;]
    Ad*x + B*u
end

Ts = 1
x0 = [10.0; 0] # Extra state due to explicit input integrator
xr = [-4.0; 0]
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x, u=:u, y=:y)
dynamics = DyadControlSystems.add_input_integrators(dynamics)
observer = StateFeedback(dynamics, x0)

running_cost = StageCost() do si, p, t
    e = (si.x-si.r)[1]
    dot(e, e) + dot(si.u, si.u)
end
terminal_cost = TerminalCost() do ti, p, t
    e = (ti.x-ti.r)[1]
    10dot(e, e)
end

objective = Objective(running_cost, terminal_cost)
p = nothing
N = 10
prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    p,
    xr,
    presolve = true,
    verbose = isinteractive(),
);
histii3 = MPC.solve(prob; x0, T=100, verbose = isinteractive())
plot(histii3)
X,E,R,U,Y,UE = reduce(hcat, histii3)
@test abs(E[1, end]) < 1e-4
@test abs(U[1, end]) < 1e-4


# ==============================================================================
## Same as above but with integration in the solver
# ==============================================================================
function linsys(x, u, _, _)
    Ad = [0.3679;;]
    B = [0.6321;;]
    Ad*x + B*u
end

Ts = 1
x0 = [10.0]
xr = [-4.0]
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x, u=:u, y=:y, input_integrators=1:1)
observer = StateFeedback(dynamics, x0)

running_cost = StageCost() do si, p, t
    e = (si.x-si.r)[1]
    dot(e, e)
end
difference_cost = DifferenceCost((si, p, t)->si.u) do e, p, t
    dot(e, e)
end
terminal_cost = TerminalCost() do ti, p, t
    e = (ti.x-ti.r)[1]
    10dot(e, e)
end

objective = Objective(running_cost, terminal_cost, difference_cost)
p = nothing
N = 10
prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    p,
    xr,
    presolve = true,
    verbose = isinteractive(),
);
histii4 = MPC.solve(prob; x0, T=100, verbose = isinteractive())
plot(histii4)
X2,E2,R2,U2,Y2,UE2 = reduce(hcat, histii4)
@test abs(E2[1, end]) < 1e-4
@test abs(U2[1, end]) ≈ -xr[] atol=1e-4
@test X[2, 2:end] ≈ U2[1, :] atol=1e-3 # Test that the method with solver integration is the same as the method with explicit integration
# plot([X[2, 2:end] U2[1, :]])



## With disturbance
# kf = UnscentedKalmanFilter(dynamics, [1;;], [1;;])
# prob5 = GenericMPCProblem(
#     dynamics;
#     N,
#     observer=kf,
#     objective,
#     p,
#     xr,
#     presolve = true,
#     verbose = isinteractive(),
# );
# disturbance = (u,t)-> t > 25
# histii5 = MPC.solve(prob5; x0, T=100, verbose = isinteractive(), disturbance)
# plot(histii5)