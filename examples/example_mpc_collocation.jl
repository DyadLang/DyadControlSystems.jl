# ENV["JULIA_DEBUG"] = ""
#=
This example implements an MPC controller for the cartpole system. The continuous dynamics are discretized with RK4 and a quadratic cost function is optimized using Optim.LBFGS
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
    dx = cartpole(x, u)

Continuous-time dynamics for the cart-pole system with state `x` and control input `u`.
"""
function cartpole(x, u, p, _=0)
    T = promote_type(eltype(x), eltype(u))
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81

    q  = x[SA[1, 2]]
    qd = x[SA[3, 4]]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp * g * l * s]
    B = @SVector [1, 0]
    if T <: Symbolics.Num
        qdd = Matrix(-H) \ Vector(C * qd + G - B * u[1])
        return [qd; qdd]
    else
        qdd = -H \ (C * qd + G - B * u[1])
        return [qd; qdd]::SVector{4, T}
    end
end


##
nu = 1 # number of controls
nx = 4 # number of states
Ts = 0.1 # sample time
N = 20
x0 = ones(nx) # Initial state
r = zeros(nx)
Q1 = Diagonal(@SVector ones(nx)) # state cost matrix
Q2 = 0.1Diagonal(@SVector ones(nu)) # control cost matrix
Q3 = 1Q2

e = zeros(nx)

measurement = (x,u,p,t) -> x

dynamics = FunctionSystem(cartpole , measurement; x=:x^nx, u=:u^nu, y=:y^nx)
discrete_dynamics = MPC.rk4(dynamics, Ts)

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

difference_cost = DifferenceCost((si,p,t)->SVector(si.u[])) do e, p, t
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

x = zeros(nx, N+1)
u = zeros(nu, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p, t)
oi = ObjectiveInput(x, u, r)
ny = nx
observer = StateFeedback(discrete_dynamics, x0)

solver = MPC.IpoptSolver(;
        verbose = false,
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
    verbose = false,
    # gradient_method = :forwarddiff,
    # jacobian_method = :symbolics,
);

# 12  1.1953815e+02 4.72e-12 7.64e-06  -5.0 5.28e-05    -  1.00e+00 1.00e+00h  1 N = 20
# 8  7.8815742e+01 7.48e-14 4.17e-06  -5.0 7.34e-06    -  1.00e+00 1.00e+00h  1  N = 2
# 10  1.0655019e+02 4.71e-08 1.76e-06  -5.0 8.23e-03    -  1.00e+00 1.00e+00h  1 N = 20 symbolic jacobian

# c = randn(length(prob.constraints))
# oinp = DyadControlSystems.remake(prob.objective_input, x=xn)
# MPC.evaluate!(c, prob.constraints, oinp, p, t)

x_, u_ = get_xu(prob.vars)

plot(
    plot(x_[:, 1:5:end]'),
    plot(u_'),
) |> display


@time history = MPC.solve(prob; x0, T = 50, verbose = false, dyn_actual=discrete_dynamics) # solve for T=500 time steps
# 3.848386 seconds (1.88 M allocations: 85.015 MiB) T = 50
# 3.754208 seconds (1.88 M allocations: 84.984 MiB) less views into c
# 0.555517 seconds (1.88 M allocations: 81.781 MiB) with symbolic lagrangian hessian

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