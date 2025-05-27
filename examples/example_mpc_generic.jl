# ENV["JULIA_DEBUG"] = ""
#=
This example implements an MPC controller for the cartpole system. The continuous dynamics are discretized with RK4 and a quadratic cost function is optimized using Optim.LBFGS
This is a very naive approach, but gives surprisingly okay performance. 
There are no contraints on control signals or states.
=#
using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.Symbolics
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
N = 10
x0 = ones(nx) # Initial state
r = zeros(nx)
Q1 = Diagonal(@SVector ones(nx)) # state cost matrix
Q2 = 0.1Diagonal(@SVector ones(nu)) # control cost matrix
Q3 = 1Q2

e = zeros(nx)

discrete_dynamics = MPC.rk4(cartpole, Ts)
measurement = (x,u,p,t) -> x

dynamics = FunctionSystem(discrete_dynamics, measurement, Ts; x=:x^nx, u=:u^nu, y=:y^nx)

QN, _ = MPC.calc_QN_AB(Q1, Q2, Q3, dynamics, r)
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
    u = (si.u)[]
    x4 = (si.x)[4]
    SA[
        u
        x4
    ]
end

x = zeros(nx, N+1)
u = zeros(nu, N)

x, u = MPC.rollout(dynamics, x0, u, p, t)

oi = ObjectiveInput(x, u, r)

# MPC.evaluate(objective, oi, p, t)
# MPC.evaluate(difference_cost, oi, p, t)


# @test obj(oi, p, t) == 0



# ms = MPC.MultipleShooting(dynamics, N, false)
# c = randn(length(ms))
# MPC.evaluate!(c, ms, oi, p, t)
# @test c ≈ 0*c atol=1e-6

# initial_state_constraint = InitialStateConstraint(copy(x0))
# constraints = CompositeMPCConstraint(initial_state_constraint, ms, control_constraint)

# length(constraints)

# c = randn(length(constraints))

# evaluate!(c, constraints, oi, p, t)
# constraints(c, oi, p, t)


## Using interface
ny = nx
observer = StateFeedback(dynamics, x0)
solver = MPC.IpoptSolver(;
        verbose = false,
        tol = 1e-4,
        acceptable_tol = 1e-1, 
        max_iter = 100,
        max_cpu_time = 20.0,
        max_wall_time = 20.0,
        constr_viol_tol = 1e-4,
        acceptable_constr_viol_tol = 1e-1,
        acceptable_iter = 2,
)

using MadNLP
solver = MadNLP.Optimizer(print_level=MadNLP.WARN, blas_num_threads=4, tol = 1e-4,
                            acceptable_tol = 1e-1, 
                            max_iter = 100,
                            max_wall_time = 20.0,
                            # constr_viol_tol = 1e-4,
                            # acceptable_constr_viol_tol = 1e-1,
                            acceptable_iter = 2,)

prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [control_constraint],
    p,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
    verbose = false,
);

# c = randn(length(prob.constraints))
# MPC.evaluate!(c, prob.constraints, oi, p, t)

x_, u_ = get_xu(prob.vars)

plot(
    plot(x_'),
    plot(u_'),
) |> display


# TODO: update x0 and xr stuff in problem between solves
# rewrite get_bounds so that each constraint can tell us their bounds, this helps the user specify stuff more easily
@time history = MPC.solve(prob; x0, T = 100, verbose = false) # solve for T=500 time steps
# 0.38
# 500.697 ms (1776198 allocations: 118.18 MiB)
# 465.147 ms (1720368 allocations: 114.65 MiB) advance vars for better warm start
# 373.771 ms (886161 allocations: 78.84 MiB) fix type instability of findnz in eval hessian lagrangian
# 347.253 ms (605615 allocations: 65.13 MiB) function barrier in SparseDiffTools ForwardColorJacCache
# 340.673 ms (463239 allocations: 48.83 MiB) StageCost{F} where F to force specialization on function
# 341.764 ms (488125 allocations: 42.00 MiB) # generated function over costs
# 341.023 ms (412969 allocations: 36.25 MiB) sum(f, sci) -> for loop
# 334.638 ms (393043 allocations: 26.01 MiB) # reuse ForwardColorJacCache
# 325.648 ms (330891 allocations: 22.90 MiB) type asserts in eval_hessian_lagrangian
# 260.198 ms (324591 allocations: 21.53 MiB) tweak Ipopt tolerances
# 190.719 ms (627482 allocations: 131.57 MiB) with MadNLP (extra allocations appears since they are now in Julia rather than in C)
# 127.158 ms (604152 allocations: 96.15 MiB) with MadNLP and same tol as Ipopt

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