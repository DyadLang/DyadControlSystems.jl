using DyadControlSystems
using DyadControlSystems.MPC
using ControlSystems: Discrete
using Statistics, LinearAlgebra
using Test
using Plots
using Optim

include("../../src/MPC/symbolic_mpc.jl")




## Basic almost linear case
Ts = 1

Ad = [0.9 0.5*Ts; 0 0.9]
B = 0.01*[0.1; 1;;]
function linsys(x, u, _, _)
    Ad*x - 0*0.05*x.^3 + B*u
end

nx = 2
nu = 1
xr0 = [1, 1] # reference state
xr, ur = quick_trim(linsys, xr0, zeros(nu))

#
N = 3 # must have at least N=6 to expose all problems
x0 = [1.0, 0.0]
# Control limits
umin = -100 * ones(nu)
umax = 100 * ones(nu)
# State limits (state constraints are soft by default)
xmin = nothing#-100 * ones(nx)
xmax = nothing#100 * ones(nx)
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x^2, u=[:u], y=:y^2)
constraints = NonlinearMPCConstraints(; umin, umax)
solver = OSQPSolver(
    verbose = false,
    eps_rel = 1e-10, # solve to high accuracy to compare to known solution
    max_iter = 1500,
    check_termination = 5,
    sqp_iters = 15,
    dynamics_interval = 1,
    polish = true,
)

Q2 = 0.01spdiagm(ones(nu)) # control cost matrix
Q1 = 10spdiagm(ones(nx)) # state cost matrix

prob = QMPCProblem(dynamics; Q1 = Q1, Q2 = Q2,constraints,N,xr,ur,solver, qs=0, qs2=0)

hist = solve(prob; x0, T=30)#, sqp_callback)

cost0 = MPC.lqr_cost(hist)
plot(hist)


prob.QN .= MPC.calc_QN_AB(prob, xr, ur)[1]
L = lqr(Discrete, Ad, B, Matrix(Q1), Matrix(Q2))
u0 = -L*(x0-xr)
##
@time funs, syms = build_symbolic_functions(prob, constraints)
X,U = symbolic_solve(prob, funs, x0, repeat(prob.xr, 1, prob.N+1), repeat(ur, 1, prob.N); T = length(hist))
cost_sqp = MPC.lqr_cost(X .- xr, U .- ur, prob.Q1, prob.Q2)
cost1 = MPC.lqr_cost(reduce(hcat, hist.X) .- xr, reduce(hcat, hist.U) .- ur, prob.Q1, prob.Q2)
@test cost0 ≈ cost1
@test cost0 ≈ dot(x0-xr, prob.QN, x0-xr) rtol=1e-5
@test_broken hist.U[1] ≈ u0 # Similar tests pass when xr = 0 so there is still some problem in the handling of xr
@test_broken U[1] ≈ u0[]

#=
Om man manuellt sätter initialgissningen för u i symbolc MPC får man en bra lösning, men oavsett vad man sätter den till flyttar optimeringen inte på den mer än ytterst lite.

Ett annat problem är att symbolic MPC använder konstant QN, medan LQMPC räknar om QN baserat på xg

I symbolic mpc verkar det som att vi tolkar wk både som optimeringsvariablen ,dvs. Δx, och faktiskt x. Ingen av funktionerna verkar bero på wk ändå, förutom g och där sätter vi wk till 0

Om antalet SQP-iters görs extremt högt (50000) så blir lösningen för symbolic MPC identisk med LQMPC
=#


plot(hist)
plot!(range(0, length=size(X,2), step=Ts), X', lab="SQP", sp=1)
plot!(range(0, length=size(X,2), step=Ts), U', lab="SQP $cost_sqp", sp=2)
scatter!([0], u0, sp=2, lab="lqr u0")
display(current())

##


@show cost_sqp - cost0
@show cost_sqp - cost1

@test cost_sqp ≈ cost0 rtol=1e-7
@test norm(X[:,end]-xr) < 1e-4
@test norm(U[:,end]-ur) < 1e-4

##
XR = repeat(prob.xr, 1, prob.N+1)
XR[:,1] .= x0
UR = repeat(ur, 1, prob.N)
w0 = [vec(XR); vec(UR)]

##
# using Optim, Optim.LineSearches
# res = Optim.optimize(
#     w->syms.lossfun(w, x0, XR,UR,XR,UR) + 10*sum(abs, funs.g(w, x0, XR, UR)),
#     w0,
#     NelderMead(),
#     Optim.Options(
#         store_trace       = true,
#         show_trace        = true,
#         show_every        = 10,
#         iterations        = 10000,
#         allow_f_increases = true,
#         time_limit        = 100,
#         x_tol             = 0,
#         f_abstol          = 0,
#         g_tol             = 1e-8,
#         f_calls_limit     = 0,
#         g_calls_limit     = 0,
#     ),
# )

# res.minimizer


prob = LQMPCProblem(dynamics; Q1 = Q1, Q2 = Q2,constraints,N,xr,ur,solver, qs=0, qs2=0)

XR[:,1] .= x0
MPC.rollout!(prob.dynamics, XR, 0*UR, 0, 1)


wk = [vec(XR); vec(UR)]
xg = copy(XR)
ug = copy(UR)
x_current = copy(x0)
fq = funs.q(wk, xr, ur, XR, UR)
fg = funs.g(0*wk, x_current, xg, ug)

MPC.update_constraints_sqp!(prob, XR, UR, x0; update_dynamics=true)
MPC.update_xr_sqp!(prob, xr, ur, XR, 0*UR; force=false)

@test prob.q ≈ [7.54498376401307, -0.48137782034558513, 6.54498376401307, -0.48137782034558513, 5.6449837640130704, -0.48137782034558513, 14.855308785570347, 8.81271127797157, -0.004813751638520473, -0.004813751638520473, -0.004813751638520473]

res = OSQP.solve!(prob.solver.model)





xopt, u = MPC.optimize!(prob, x0, 0, 1, verbose=false)


# prob.q = [7.54498376401307, -0.48137782034558513, 6.54498376401307, -0.48137782034558513, 5.6449837640130704, -0.48137782034558513, 14.855308785570347, 8.81271127797157, -0.004813751638520473, -0.004813751638520473, -0.004813751638520473]