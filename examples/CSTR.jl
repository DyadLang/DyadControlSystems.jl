
using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.Symbolics
using OrdinaryDiffEq
using Test


sys = DyadControlSystems.ControlDemoSystems.cstr()
display(sys.sys)

##
dynamics = sys.dynamics
nu  = dynamics.nu # number of controls
nx  = dynamics.nx # number of states
Ts  = sys.Ts # sample time
N   = 20 # MPC prediction horizon
x0  = [0.8, 0.5, 134.14, 130] # Initial state
u0  = [10, -1000]
r   = SA[NaN, 0.6, NaN, NaN]

scale_x = [1,1,100,100]
scale_u = [100, 2000]

discrete_dynamics = MPC.rk4(dynamics, Ts)
# discrete_dynamics    = MPC.MPCIntegrator(dynamics, ODEProblem, Rodas5P(); p, Ts, nx, nu, dt=Ts/10, adaptive=false)
# discrete_dynamics = FunctionSystem(discrete_dynamics, dynamics.measurement, Ts; dynamics.x, dynamics.u, dynamics.y, dynamics.p)

t = 1
p = sys.p

lb = SA[0.1, 0.1, 50, 50, 5, -8500]
ub = SA[2, 2, 140, 140, 100, 0.0]


xmin = [0.1, 0.1, 50, 50]
xmax = [2, 2, 140, 142]
umin = [5, -8500]
umax = [100, 0.0]
bounds_constraints = BoundsConstraint(; xmin, xmax, umin, umax)


running_cost = StageCost() do si, p, t
    abs2(si.x[2]-si.r[2]) 
end

getter = (si,p,t)->SA[si.u[1], si.u[2]]
difference_cost = DifferenceCost(getter) do e, p, t
    0.005*(0.1/100*abs2(e[1]) + 1e-3/2000*abs2(e[2]) )# Control action penalty should be on differences
end

terminal_cost = TerminalCost() do ti, p, t
    abs2(ti.x[2]-ti.r[2])
end

objective = Objective(running_cost, difference_cost, terminal_cost)

x = zeros(nx, N+1) .+ x0
u = randn(nu, N) # lb[5:6] .* ones(1, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p, t)
oi = ObjectiveInput(x, u, r)

solver = MPC.IpoptSolver(;
        verbose                     = isinteractive(),
        tol                         = 1e-6,
        acceptable_tol              = 1e-3,
        max_iter                    = 500,
        max_cpu_time                = 10*10.0,
        max_wall_time               = 50.0,
        constr_viol_tol             = 1e-6,
        acceptable_constr_viol_tol  = 1e-5,
        acceptable_iter             = 5,
        exact_hessian               = true,
        # linear_inequality_constraints = true,
        mu_strategy                 = "adaptive",
        mu_init                     = 1e-6,
        acceptable_obj_change_tol = 0.01,

)

# using OptimizationMOI: MOI
# using NLopt
# solver = OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer, 
#                                                          "algorithm" => :LD_SLSQP)
# OptimizationMOI.MOI.set(solver, MOI.RawOptimizerAttribute("ftol_rel"), 1e-3)
# OptimizationMOI.MOI.set(solver, MOI.RawOptimizerAttribute("ftol_abs"), 1e-3)
# OptimizationMOI.MOI.set(solver, MOI.RawOptimizerAttribute("xtol_abs"), 1e-3)

disc = CollocationFinE(dynamics, false; n_colloc=2, scale_x)
# disc = Trapezoidal(dynamics, N, Ts, false; scale_x)
observer = StateFeedback(discrete_dynamics, x0)
prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraints],
    p,
    Ts,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
    scale_x,
    scale_u,
    # jacobian_method = :symbolics,
    disc,
    verbose = isinteractive(),
);


@time history = MPC.solve(prob; x0, T = 50, verbose = false, noise=false, dyn_actual = discrete_dynamics);
@test history.X[end][2] ≈ 0.6 atol = 0.01
plot(history, layout=(3, 2), sp=[1 2 3 4 1 2 3 4 5 6], title="", xlabel="Time [hrs]")
hline!(xmax[3:4]', ls=:dash, lw=1, color=:black, primary=false, sp=(3:4)')
# hline!(xmax', ls=:dash, lw=1, color=:black, primary=false)

# This problem appears to be dominated by solve time, so we can't really do super much other than using the ma57 solver that do-mpc uses.
# Changing StageConstraint to BoundsConstraint made the problem 2.6x faster



## Uncertain ===================================================
p_actual = Base.setindex(p, 0.95*p.E_A_ad, :E_A_ad) # Decrease E_A_ad by 5%
history_u = MPC.solve(prob; x0, T = 50, verbose = true, noise=false, dyn_actual = discrete_dynamics, p_actual);
plot(history_u, layout=(3, 2), sp=[1 2 3 4 1 2 3 4 5 6], title="", xlabel="Time [hrs]", topmargin=-5Plots.mm)
hline!(xmax[3:4]', ls=:dash, lw=1, color=:black, sp=(3:4)', lab="Upper bound")

##

p_uncertain = MPCParameters.([
    p, # nominal case
    Base.setindex(p, 0.949*p.E_A_ad, :E_A_ad), # 5% lower
    Base.setindex(p, 1.051*p.E_A_ad, :E_A_ad), # 5% higher
])

observer = StateFeedback(discrete_dynamics, x0)
prob_robust = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraints],
    p = p_uncertain,
    Ts,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
    scale_x,
    scale_u,
    #jacobian_method = :symbolics,
    disc,
    verbose = true,
    robust_horizon = 2, # Indicate that we are designing a robust MPC controller
)

#

# anim = Plots.Animation()
# function callback(actual_x, u, x, X, U)
#     n_robust = length(p_uncertain)
#     (; nx, nu) = dynamics
#     T = length(X)
#     tpast = 1:T
#     fig = plot(
#         tpast,
#         reduce(hcat, X)',
#         c       = (1:nx)',
#         layout  = nx+nu,
#         sp      = (1:nx)',
#         ylabel  = permutedims(state_names(dynamics)),
#         legend  = true,
#         lab     = "History",
#         xlims   = (1,60),
#     )
#     plot!(
#         tpast,
#         reduce(hcat, U)',
#         c       = (1:nu)',
#         sp      = (1:nu)' .+ nx,
#         ylabel  = permutedims(input_names(dynamics)),
#         legend  = true,
#         lab     = "History",
#         xlims   = (1,60),
#     )

#     xs = map(1:n_robust) do ri
#         xu, uu = get_xu(prob_robust, ri)
#         xu
#     end
#     Xs = reduce(vcat, xs)
#     tfuture = (1:size(Xs, 2)) .+ (T - 1)
#     lab = ""
#     plot!(tfuture, Xs'; c = (1:n_robust)', l = :dash, sp = (1:nx)', lab)

#     us = map(1:n_robust) do ri
#         xu, uu = get_xu(prob_robust, ri)
#         uu
#     end
#     Us = reduce(vcat, us)
#     tfuture = (1:size(Us, 2)) .+ (T - 1)
#     plot!(tfuture, Us'; c = (1:n_robust)', l = :dash, sp = (1:nu)'.+ nx, lab)
#     hline!([bounds_constraints.xmin; bounds_constraints.xmax]', l = (:gray, :dash), lab = "Constraint", sp=(1:nx)')
#     hline!([bounds_constraints.umin; bounds_constraints.umax]', l = (:gray, :dash), lab = "Constraint", sp=(1:nu)' .+ nx)
#     hline!([r[2]], l = (:black, :dash), lab = "Reference", sp=2)
#     Plots.frame(anim, fig)
# end


history_robust = MPC.solve(prob_robust; x0, T = 50, verbose = true, noise=false, dyn_actual = discrete_dynamics, p_actual)#, callback);
# gif(anim, fps = 5)

@test history_robust.X[end][2] ≈ 0.6 atol = 0.1