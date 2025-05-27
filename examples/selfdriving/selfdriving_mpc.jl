using Pkg
Pkg.activate(@__DIR__())
using DyadControlSystems
using DyadControlSystems.MPC
using TrajectoryLimiters

## Generate a reference trajectory for the lateral position y with bounded velocities and accelerations

traj(t) = 1.5 * sign(sin(2pi * t / 20))   # Reference to be smoothed
ẍM = 0.4     # Maximum lateral acceleration
ẋM = 1.5     # Maximum velocity
Ts = 0.1     # Sample time
t = 0:Ts:30  # Time vector
R = traj.(t) # An array of sampled position references 

limiter = TrajectoryLimiter(Ts, ẋM, ẍM)

Y, Ẏ, Ÿ = limiter(R)

plot(
    t,
    [Y Ẏ Ÿ],
    plotu = true,
    c = :black,
    title = ["Lateral position \$y(t)\$" "Lateral velocity \$\\dot{y}(t)\$" "Lateral acceleration \$\\ddot y(t)\$"],
    ylabel = "",
    layout = (3, 1),
)
plot!(traj, extrema(t)..., sp = 1, lab = "", l = (:black, :dashdot))

##

# The "kinematic bicycle model" of a car, with an integrator from acceleration (input) to velocity (state)
function car_dynamics(x, u, p, t)
    y, v, ψ = x
    a, δ = u
    L = 2.9 # Length between the axels
    ẏ = v * sind(ψ)
    v̇ = a
    ψ̇ = v / L * tand(δ)
    return SA[ẏ, v̇, ψ̇]
end

measurement = (x, u, p, t) -> x
nx = 3
nu = 2
ny = 3
Ts = 0.1

# Names for states and inputs
xn = [:y, :v, :ψ]
un = [:a, :δ]
yn = xn

cont_dynamics = FunctionSystem(car_dynamics, measurement; x = xn, u = un, y = yn)

discrete_dynamics = MPC.rk4(cont_dynamics, Ts)
p = nothing

x0 = [
    0,        # y
    30 / 3.6, # v
    0,        # ψ
] # initial state
xr = [
    Y'                           # y
    fill(50 / 3.6, 1, length(Y)) # v
    fill(0, 1, length(Y))        # ψ
] # reference state


# Control limits
umin = [
    -10,  # a
    -25,  # δ
]
umax = [0.2 * 9.8, 25]

# State limits
xmin = [
    -1.53, # y
    0,     # v
    -6,    # ψ
]
xmax = [
    1.53,         # y
    120 / 3.6,    # v
    6,            # ψ
]

scale_x = [
    10,      # y
    30,      # v
    60.0,    # ψ
]
scale_u = [120 / 3.6, 60.0]

bounds_constraints = BoundsConstraint(; xmin, xmax, umin, umax)

running_cost = StageCost() do si, p, t
    abs2(si.x[1] .- si.r[1]) +     # y
    0.02abs2(si.x[2] .- si.r[2]) + # v
    0.01abs2(si.x[3] .- si.r[3]) + # ψ
    0.01si.u[1]^2 +                # a
    0.00001si.u[2]^2               # δ
end

# Penalize control-action differences
getter = (si,p,t)->SA[si.u[1], si.u[2]]
difference_cost = DifferenceCost(getter) do e, p, t
    abs2(e[1]) +      # a
    0.001abs2(e[2])   # δ
end

objective = Objective(running_cost, difference_cost)

terminal_constraint = TerminalStateConstraint(zeros(1), zeros(1)) do ti, p, t
    ti.x[1] - ti.r[1]
end


## Define objective input
N = 30
u = zeros(nu, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p, 0)
oi = ObjectiveInput(x, u, xr)


## Choose solver options
solver = MPC.IpoptSolver(;
    verbose                    = false,
    tol                        = 1e-6,
    acceptable_tol             = 1e-2,
    max_iter                   = 5000,
    max_cpu_time               = 20.0,
    max_wall_time              = 20.0,
    constr_viol_tol            = 1e-6,
    acceptable_constr_viol_tol = 1e-3,
    acceptable_iter            = 5,
    exact_hessian              = true,
    mu_init                    = 1e-6,
    acceptable_obj_change_tol  = 0.01,
)

observer = StateFeedback(discrete_dynamics, x0)



# disc = MultipleShooting(discrete_dynamics)
disc = Trapezoidal(cont_dynamics)
# disc = CollocationFinE(cont_dynamics, n_colloc=2)
prob = GenericMPCProblem(
    cont_dynamics;
    N,
    Ts,
    observer,
    objective,
    constraints = [bounds_constraints, terminal_constraint],
    p,
    # Ts,
    objective_input = oi,
    solver,
    xr,
    presolve = true,
    scale_x,
    scale_u,
    disc,
    verbose = true,
)


# x_sol, u_sol = get_xu(prob)
# plot(
#     plot(x_sol', layout=nx, lab=permutedims(state_names(cont_dynamics))),
#     plot(u_sol', layout=nu, lab=permutedims(input_names(cont_dynamics))),
# )

# Define a callback for creating an animation
anim = Plots.Animation()
function callback(actual_x, u, x, X, U)
    # T = length(X)
    # tpast = 1:T
    # tfuture = (1:size(x, 2)) .+ (T - 1)
    # fig = plot(
    #     tpast,
    #     reduce(hcat, X)',
    #     c       = (1:nx)',
    #     layout  = (3, 1),
    #     sp      = (1:3)',
    #     title   = ["Lateral position \$y(t)\$" "Velocity \$v(t)\$" "Angle \$ψ(t)\$"],
    #     ylabel  = "",
    #     legend  = true,
    #     lab     = "History",
    # )
    # plot!(tfuture, x', c = (1:nx)', l = :dash, sp = (1:3)', lab = "Prediction")
    # plot!(
    #     tfuture,
    #     prob.xr[:, 1:length(tfuture)]',
    #     c   = :black,
    #     l   = :dot,
    #     sp  = (1:3)',
    #     lab = "Reference",
    # )
    # Plots.frame(anim, fig)
end

# Run MPC controller
@time history =
    MPC.solve(prob; x0, T = 300, verbose = true, dyn_actual = discrete_dynamics, callback);
# gif(anim, "/tmp/selfdriving.gif", fps = 25)

# 1.057938 seconds (1.46 M allocations: 99.163 MiB)
# 250/1.057938

# Extract matrices
X,E,R,U,_ = reduce(hcat, history)

fx = plot(X', layout=nx, lab=permutedims(state_names(cont_dynamics)))
plot!(R')
plot(
    fx,
    plot(U', layout=nu, lab=permutedims(input_names(cont_dynamics))),
)

@test mean(abs, E[1, :]) < 0.1
@test mean(abs, E[2, 50:end]) < 0.5


# ## Animate
# function animation_dynamics(x, u, p, t)
#     x, y, v, ψ = x
#     a, δ = u
#     L = 2.9 # Length between the axels
#     ẋ = v * cosd(ψ)
#     ẏ = v * sind(ψ)
#     v̇ = a
#     ψ̇ = v / L * tand(δ)
#     return SA[ẋ, ẏ, v̇, ψ̇]
# end

# let
#     dyn = MPC.rk4(animation_dynamics, Ts)
#     x = [0;  x0]
#     Xanim = [x]
#     anim = Plots.Animation()

#     x_coords = [-1, 1, 1, -1, -1]
#     y_coords = 0.2*[-1, -1, 1, 1, -1]
#     car = Shape(x_coords, y_coords)

#     for i in 1:size(U, 2)#÷2
#         u = U[:, i]
#         x = dyn(x, u, p, i)
#         push!(Xanim, x)

#         fig = plot(first.(Xanim), getindex.(Xanim, 2),
#             # title   = ["Lateral position \$y(t)\$" "Velocity \$v(t)\$" "Angle \$ψ(t)\$"],
#             ylabel  = "",
#             legend  = false,
#             lab     = "History",
#             ylims = (-2.3, 2.3),
#             xlims = (-1, size(U, 2)),
#             size =  (1300, 100),
#             framestyle = :none,
#         )
#         hline!([-0.75 0.75], c = :black, l = :dash, primary=false)
#         hline!([-2.25 2.25], c = :black, primary=false)
#         cari = Plots.rotate(car, deg2rad(x[4]))
#         cari = Plots.translate(cari, x[1], x[2])
#         plot!(cari)
#         frame(anim, fig)
#     end
#     gif(anim, "/tmp/selfdriving_car.gif", fps = 25)
# end

