using DyadControlSystems
using DyadControlSystems.MPC
using Optimization
using StaticArrays, LinearAlgebra, Plots, Test


function linsys(x, u, p, _)
    Ad = SA[1.0;;] # [0.3679;;]
    B = SA[p[]*0.6321;;]
    Ad*x + B*u
end

Ts = 1
x0 = [10.0]
xr = [0.0]
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x, u=:u, y=:y)

running_cost = StageCost() do si, p, t
    e = si.x[]-si.r[]
    dot(e, e)
end
difference_cost = DifferenceCost() do e, p, t
    dot(e, e)
end
terminal_cost = TerminalCost() do ti, p, t
    e = ti.x[]-ti.r[]
    10dot(e, e)
end

bounds_constraint = BoundsConstraint(umin = [-4], umax = [4], xmin = [-0.7], xmax = [Inf])

objective = Objective(running_cost, terminal_cost, difference_cost)
p = [1.0]
N = 10

observer = StateFeedback(dynamics, x0)
solver = IpoptSolver(verbose=false, max_iter=1000, exact_hessian = true)


prob = GenericMPCProblem(
    dynamics;
    solver,
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    xr,
    # jacobian_method = :symbolics,
    presolve = true,
    verbose = isinteractive(),
);
X0, U0 = get_xu(prob)

@test X0[1] ≈ x0[]

@time hist = MPC.solve(prob; x0, T=20, verbose = isinteractive());
# 36.876 ms (125416 allocations: 9.56 MiB)
# 29.019 ms (73212 allocations: 5.48 MiB) StaticArray in linsys
# 30.236 ms (32037 allocations: 3.51 MiB) # no allocation in stagecost
# 29.365 ms (21093 allocations: 1.92 MiB) # gradient method = :reversediff (helpful when not using StaticArrays everywhere)
# 19.758 ms (27167 allocations: 3.19 MiB) # Switch to DiffInterface
# 20.752 ms (23342 allocations: 2.84 MiB) # store static array size in ObjectiveInput
# 19.935 ms (22492 allocations: 2.75 MiB) # SVector x0 in difference cost

@test abs(hist.X[end][]) < 2e-3
@test abs(hist.U[1][]) > 3


##


p_uncertain = MPCParameters.([1.0; 0.5; 5]) # Place the nominal parameters first
n_robust = length(p_uncertain)
observer = StateFeedback(dynamics, x0)
probu = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p = p_uncertain,
    xr,
    presolve = true,
    verbose = isinteractive(),
    robust_horizon = 1,
);

initial_xs = map(1:n_robust) do ri
    xu, uu = get_xu(probu, ri)
    xu[1]
end

@test allequal(initial_xs)

@time histu = MPC.solve(probu; x0, T=20, verbose = isinteractive(), p_actual=1.0);

plot(hist)
plot!(histu)

@test abs(histu.X[end][]) < 0.05
@test abs(histu.U[1][]) > 3


@test sum(abs2, first.(hist.U)) > sum(abs2, first.(histu.U)) # the non-robust controller uses more aggressive control signal


# fig = plot(layout=2)
# for ri = 1:n_robust
#     xu, uu = get_xu(probu, ri)
#     plot!(xu', sp=1)
#     plot!(uu', sp=2)
# end
# fig

initial_xs = map(1:n_robust) do ri
    xu, uu = get_xu(probu, ri)
    xu[1]
end
@test allequal(initial_xs)

initial_us = map(1:n_robust) do ri
    xu, uu = get_xu(probu, ri)
    uu[1]
end
@test allequal(initial_us)



# ##
# anim = Plots.Animation()
# function callback(actual_x, u, x, X, U)
#     (; nx, nu) = dynamics
#     T = length(X)
#     tpast = 1:T
#     fig = plot(
#         tpast,
#         reduce(hcat, X)',
#         c       = (1:nx)',
#         layout  = (nx+nu, 1),
#         sp      = (1:nx)',
#         ylabel  = "x",
#         legend  = true,
#         lab     = "History",
#     )

#     plot!(
#         tpast,
#         reduce(hcat, U)',
#         c       = (1:nu)',
#         sp      = (1:nu)' .+ nx,
#         ylabel  = "u",
#         legend  = true,
#         lab     = "History",
#     )

#     xs = map(1:n_robust) do ri
#         xu, uu = get_xu(probu, ri)
#         xu
#     end
#     Xs = reduce(vcat, xs)
#     tfuture = (1:size(Xs, 2)) .+ (T - 1)
#     plot!(tfuture, Xs', c = (1:nx)', l = :dash, sp = (1:nx)', lab = "Prediction")

#     us = map(1:n_robust) do ri
#         xu, uu = get_xu(probu, ri)
#         uu
#     end
#     Us = reduce(vcat, us)
#     tfuture = (1:size(Us, 2)) .+ (T - 1)
#     plot!(tfuture, Us', c = (1:nu)', l = :dash, sp = (1:nx)'.+ nx, lab = "Prediction")

#     # plot!(
#     #     tfuture,
#     #     prob.xr[:, 1:length(tfuture)]',
#     #     c   = :black,
#     #     l   = :dot,
#     #     sp  = (1:nx)',
#     #     lab = "Reference",
#     # )
#     Plots.frame(anim, fig)
# end

# # Run MPC controller
# hist_anim = MPC.solve(probu; x0, T=20, verbose = false, p_actual=1.0, callback);

# gif(anim, "/tmp/selfdriving.gif", fps = 7)


## Test robust MPC with weighted costs

running_cost = StageCost() do si, p, t
    e = p*(si.x[1]-si.r[1])
    dot(e, e)
end
difference_cost = DifferenceCost() do e, p, t
    p*dot(e, e)
end
terminal_cost = TerminalCost() do ti, p, t
    e = (ti.x[1]-ti.r[1])
    10*p*dot(e, e)
end

objective = Objective(running_cost, terminal_cost, difference_cost)

p_uncertain = MPCParameters.([1.0; 0.5; 5], [1, 0.01, 0.01]) # Place the nominal parameters first
n_robust = length(p_uncertain)
observer = StateFeedback(dynamics, x0)
probuw = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p = p_uncertain,
    xr,
    presolve = true,
    verbose = isinteractive(),
    robust_horizon = 1,
);

histuw = MPC.solve(probuw; x0, T=20, verbose = isinteractive(), p_actual=1.0);

@test abs(histuw.X[end][]) < 0.05
@test abs(histuw.U[1][]) > 3



# plot(hist, lab="Nominal", c=1)
# plot!(histu, lab="Robust", c=2)
# plot!(histuw, lab="Weighted robust", c=3)


## Robustness vs. performance
# fig = plot(layout=(3,2))
# figs = map(enumerate([0.5, 1.0, 5.0])) do (i, p_actual)
#     hist = MPC.solve(prob; x0, T=20, verbose = false, p_actual=p_actual);
#     histu = MPC.solve(probu; x0, T=20, verbose = false, p_actual=p_actual);
#     plot!(hist, lab="Nominal", c=1, plotr=false, sp=(1:2)' .+ (i-1)*2)
#     plot!(histu, lab="Robust", c=2, plotr=false, sp=(1:2)' .+ (i-1)*2, ylabel=["p = $p_actual" ""], leftmargin=3Plots.mm)
# end
# fig


## Test using MCM
MCM = MPC.MCM

p_uncertain = MPCParameters(MCM.Particles(10, MCM.Uniform(0.5, 5))) # Place the nominal parameters first
p_uncertain = MPC.expand_uncertain_operating_points(p_uncertain)

n_robust = length(p_uncertain)
observer = StateFeedback(dynamics, x0)
probu = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p = p_uncertain,
    xr,
    presolve = true,
    # jacobian_method = :symbolics,
    # gradient_method = :reversediff,
    symbolic_lag_h = true,
    verbose = isinteractive(),
    robust_horizon = 1,
);

initial_xs = map(1:n_robust) do ri
    xu, uu = get_xu(probu, ri)
    xu[1]
end

@test allequal(initial_xs)

@time histu = MPC.solve(probu; x0, T=20, verbose = false, p_actual=1.0);
# 257.183 ms (5102078 allocations: 343.80 MiB)
# 222.281 ms (2787862 allocations: 209.05 MiB) rm dyn.Ts type uncertainty
# 189.196 ms (1639162 allocations: 191.52 MiB) rm some more uncertainty
# 176.932 ms (1338312 allocations: 146.87 MiB) no allocation in stage cost
# 143.890 ms (997952 allocations: 106.00 MiB) jacobian method = symbolics
# 108.677 ms (285740 allocations: 23.86 MiB) gradient method = reversediff
# 135.143 ms (286728 allocations: 128.50 MiB) # Switch to DiffInterface + StaticArrays in ObjectiveInput
# 62.854 ms (145276 allocations: 13.90 MiB) # symbolic_lag_h

plot(hist)
plot!(histu)

@test abs(histu.X[end][]) < 0.1
@test abs(histu.U[1][]) > 3