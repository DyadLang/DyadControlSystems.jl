# ENV["JULIA_DEBUG"] = ""
#=
This example implements an MPC controller for the pendulum DAE system. The continuous dynamics are discretized with RK4 and a quadratic cost function is optimized using Optim.LBFGS
This is a very naive approach, but gives surprisingly okay performance. 
There are no contraints on control signals or states.
=#
using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.Symbolics
using StaticArrays
using LinearAlgebra
using Plots
# using Optimization

"""
    dx = pend(x, u)

Continuous-time dynamics for the pendulum-mass system with state `x` and control input `u`.
"""
function pend(x, u, p=0, t=0)
    x,y,ux,v,λ = x
    g = 9.82
    SA[
        ux
        v
        -λ*x + u[1]
        -λ*y - g + u[2]
        #x^2 + y^2 - 1
        #x*ux + y*v
        ux^2 + v^2 − λ*(x^2 + y^2) − g*y + x*u[1] + y*u[2]
    ]
end


##
nu = 2 # number of controls
ns = 4 # number of ODE states
na = 1 # number of DAE states
nx = ns+na
Ts = 0.1 # sample time
N = 10
# Mass matrix for DAE 
M = Matrix(1I, 5, 5)
M[5,5] = 0
x0 = [ones(ns);-3.91] # Initial state
r = zeros(nx)
Q1 = Diagonal(@SVector ones(nx)) # state cost matrix
Q2 = 0.1Diagonal(@SVector ones(nu)) # control cost matrix
Q3 = 1Q2

e = zeros(nx)

measurement = (x,u,p,t) -> x

pend_f = ODEFunction(pend,mass_matrix=M)
dynamics = FunctionSystem(pend_f , measurement; x=:x^nx, u=:u^nu, y=:y^nx)
discrete_dynamics_pend    = MPC.MPCIntegrator(dynamics, ODEProblem, Rodas4(); Ts, nx, nu, dt=Ts, adaptive=false)
discrete_dynamics = FunctionSystem(discrete_dynamics_pend, measurement, Ts; x=:x^nx, u=:u^nu, y=:y^nx)

QN, _ = MPC.calc_QN_AB(Q1, Q2, Q3, discrete_dynamics, r)
QN = Matrix(QN)

t = 1

p = (; Q1, Q2, Q3, QN, e)

running_cost = StageCost() do si, p, t
    Q1, Q2 = p.Q1, p.Q2
    e = (si.x) #.- value(si.r)
    u = (si.u)
    dot(e, Q1, e) + dot(u, Q2, u)
end

difference_cost = DifferenceCost((si,p,t)->SVector(si.u[1])) do e, p, t
    dot(e, p.Q3, e)
end

terminal_cost = TerminalCost() do ti, p, t
    e = ti.x #.- value(ti.r)
    # @show size(e)
    dot(e, p.QN, e) 
end

objective = Objective(running_cost, terminal_cost, difference_cost)

control_constraint = StageConstraint([-3, -4], [3, 4]) do si, p, t
    u = (si.u)[1]
    x4 = (si.x)[4]
    SA[
        u
        x4
    ]
end

#n_colloc = 5
x = zeros(nx, N+1)
u = zeros(nu, N)
x, u = MPC.rollout(discrete_dynamics_pend, x0, u, p, t)
oi = ObjectiveInput(x, u, r)
ny = nx
observer = StateFeedback(discrete_dynamics, x0)

solver = MPC.IpoptSolver(;
        verbose = true,
        tol = 1e-4,
        acceptable_tol = 1e-1, 
        max_iter = 300,
        max_cpu_time = 20.0,
        max_wall_time = 20.0,
        constr_viol_tol = 1e-5,
        acceptable_constr_viol_tol = 1e-3,
        acceptable_iter = 10,
)
discr = CollocationFinE(dynamics, false,roots_c = "Radau")
prob = GenericMPCProblem(
    dynamics;
    N,
    Ts,
    observer,
    objective,
    constraints = [control_constraint],
    p,
    objective_input = oi,
    solver,
    xr = r,
    disc = discr,
    presolve = true,
    verbose = isinteractive(),
    # gradient_method = :forwarddiff,
    # jacobian_method = :symbolics,
);


x_, u_ = get_xu(prob.vars)

plot(
    plot(x_[:, 1:5:end]'),
    plot(u_'),
) |> display


@time history = MPC.solve(prob; x0, T = 50, verbose = false, dyn_actual=discrete_dynamics) # solve for T=500 time steps

X,E,R,U,Y = reduce(hcat, history)

plot(
    plot(X'),
    plot(U'),
)


## Bech hess of lagrangian
# hess = prob.optprob.f.cons_h
# mu = zeros(length(constraints))
# H = copy(prob.optprob.f.cons_hess_prototype[])
# x__ = randn(length(prob.vars))
# hess(H, x__, mu)

# @code_warntype hess(H, x, mu)