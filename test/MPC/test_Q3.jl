using DyadControlSystems
using DyadControlSystems.MPC
using ControlSystems: Discrete
using Statistics, LinearAlgebra
using Test
using Plots

Ts = 0.5
Ad = [0.99 Ts; 0 0.95]
B = [0.1; 1;;]

function linsys(x, u, _, _)
    # A = [0 0; 0 1]
    # C = [1 0]
    # @show size.((Ad,B,x,u))
    Ad*x + B*u
end
function XUcost(hist)
    X,E,R,U,Y = reduce(hcat, hist)
    X, U, MPC.lqr_cost(hist)
end
function sqp_callback(x,u,xopt,t,sqp_iter,prob)
    nx, nu = size(x,1), size(u,1)
    c = MPC.lqr_cost(x,u,prob)
    plot(x[:, 1:end-1]', layout=2, sp=1, c=(1:nx)', label="x nonlinear", title="Time: $t SQP iteration: $sqp_iter")
    plot!(xopt[:, 1:end-1]', sp=1, c=(1:nx)', l=:dash, label="x opt")
    plot!(u', sp=2, c=(1:nu)', label = "u", title="Cost: $(round(c, digits=5))") |> display
end

##
nx = 2
nu = 1
N = 6 # must have at least N=6 to expose all problems
x0 = [1.0, 0.0]
xr = zeros(nx) # reference state
# Control limits
umin = -100 * ones(nu)
umax = 100 * ones(nu)
# State limits (state constraints are soft by default)
xmin = -100 * ones(nx)
xmax = 100 * ones(nx)
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x^2, u=[:u], y=:y^2)
constraints = NonlinearMPCConstraints(; umin, umax)
solver = OSQPSolver(
    verbose = false,
    eps_rel = 1e-10, # solve to high accuracy to compare to known solution
    max_iter = 1500,
    check_termination = 5,
    sqp_iters = 1,
    dynamics_interval = 1,
    polish = true,
)
T = 200
Q2 = spdiagm(ones(nu)) # control cost matrix
qfactor = 1
Q1 = spdiagm(ones(nx)) # state cost matrix
##
prob = QMPCProblem(dynamics; Q1 = 100Q1, Q2 = 10000Q2, Q3=10000Q2,constraints,N,xr,solver, qs=0, qs2=0)

@test all(>(0), eigvals(Matrix(prob.P)))

hist = MPC.solve(prob; x0, T, verbose = false)#, sqp_callback)#, callback=plot_callback)
plot(hist)
##
X, U_, cost = XUcost(hist)

#
P3 = ss(Ad, B, I(2), 0, 1)
QN3 = dare3(P3, Matrix(prob.Q1),Matrix(prob.Q2),Matrix(prob.Q3))
@test isposdef(QN3)
@show eigvals(QN3)
# xN = [X[:, end]; U_[:, end]]
# x00 = [x0; 0U_[:, 1]]
xN = X[:, end]
x00 = x0
@show dot(xN, QN3, xN)
acost = cost + dot(xN, QN3, xN)
@show abs(dot(x00, QN3, x00) - acost)/acost
@test dot(x00, QN3, x00) ≈ acost rtol = 1e-4

#

## =============================================================================
# Symbolic tests deactivated since variable-layout change
## =============================================================================

## Build symbolic representation of optimal control problem.
# using Symbolics
# function vvariable(name, length)
#     un = Symbol(name)
#     u = @variables $un[1:length]
#     collect(u[])
#     # u[]
# end
# function loss(x, u, ulast, n)
#     c = 0.5*(dot(x, prob.Q1, x) + dot(u, prob.Q2, u))
#     # if n > 1
#         du = u - ulast
#         c +=  0.5dot(du, prob.Q3, du)
#     # end
#     c
# end
# function final_cost(x)
#     0.5x'QN3*x # TODO: replace by Riccati solution
# end
# X   = Num[] # variables
# U   = Num[] # variables
# lbX = Num[] # lower bound on w
# ubX = Num[] # upper bound on w
# lbU = Num[] # lower bound on w
# ubU = Num[] # upper bound on w
# g = Num[]   # equality constraints

# L = 0
# x = vvariable("x1", nx) # initial value variable


# append!(g, x0 - x)
# append!(X, x)
# append!(lbX, x) # Initial state is fixed
# append!(ubX, x)


# u0 = [U_[end]]
# u = u0#vvariable("u0", nu)
# n = 1
# for n = 1:N # for whole time horizon N
#     global x, L, u
#     ulast = u
#     u = vvariable("u$n", nu)
#     append!(U, u)
#     append!(lbU, umin)
#     append!(ubU, umax)
#     xp = dynamics(x, u, 0, 0)
#     L += loss(x, u, ulast, n) # discrete integration of loss instead of rk4 integration, this makes the hessian constant
#     append!(lbX, xmin)
#     append!(ubX, xmax)
    
#     x = vvariable("x$(n+1)", nx) # x in next time point
#     append!(X, x)
#     append!(g, xp - x) # propagated x is x in next time point
# end
# L += final_cost(x)

# # append!(g, lbX - X)
# append!(g, U - lbU)

# w = [X;U]

# H = Symbolics.hessian(L, w) .|> Symbolics.value .|> Float64
# q = Symbolics.gradient(L, w) - H*w .|> Symbolics.value .|> Float64

# @test all(>(0), eigvals(H))


# nH = size(H,1)
# @test prob.P[1:nH, 1:nH] ≈ H rtol=1e-10
# @test prob.q[1:nH] ≈ q atol=1e-10

# ##
# A = Symbolics.jacobian(g, w) .|> Symbolics.value .|> Float64
# nA = size(A, 1)
# Matrix(prob.A)
# @test Matrix(prob.A) ≈ A rtol=1e-10
