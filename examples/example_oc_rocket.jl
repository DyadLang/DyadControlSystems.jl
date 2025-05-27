using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.Symbolics
using LinearAlgebra
using StaticArrays

const h_0 = 1    # Initial height
const v_0 = 0    # Initial velocity
const m_0 = 1    # Initial mass
const g_0 = 1    # Gravity at the surface

const T_c = 3.5  # Used for thrust
const m_c = 0.6  # Fraction of initial mass left at end

const m_f = m_c * m_0              # Final mass
const T_max = T_c * g_0 * m_0        # Maximum thrust


function rocket(x, u, p, _=0)
    h_c = 500  # Used for drag
    v_c = 620  # Used for drag
    c = 0.5 * sqrt(g_0 * h_0)  # Thrust-to-fuel mass
    D_c = 0.5 * v_c * m_0 / g_0  # Drag scaling

    h, v, m = x
    T = u[]
    drag =  D_c * v^2 * exp(-h_c * (h - h_0) / h_0)
    grav = g_0 * (h_0 / h)^2
    SA[
        v
        (T - drag - m * grav) / m 
        -T/c
    ]
end

nu = 1           # number of controls
nx = 3           # number of states
N = 200           # Optimization horizon (number of time steps)
Ts = 0.001        # sample time
x0 = Float64[h_0, v_0, m_0]   # Initial state
r = zeros(nx)

measurement = (x,u,p,t) -> x                # The entire state is available for measurement
dynamics = FunctionSystem(rocket, measurement; x=[:h, :v, :m], u=:T, y=:y^nx)
discrete_dynamics = MPC.rk4(dynamics, Ts; supersample=3)


stage_constraint = StageConstraint([h_0, v_0, m_f, 0], [Inf, Inf, m_0, T_max], N) do si, p, t
    u = (si.u)[]
    h,v,m = si.x
    SA[
        h
        v
        m
        u
    ]
end

terminal_constraint = TerminalStateConstraint([m_f], [m_f]) do ti, p, t
    SA[ti.x[3]] # The final mass must be m_f
end

terminal_cost = TerminalCost() do ti, p, t
    h = ti.x[1]
    -h # Maximize the terminal altitude
end

objective = Objective(terminal_cost)
##
x = repeat(x0, 1, N+1)
u = [0.7T_max * ones(nu, N÷5)  T_max / 5 * ones(nu, 4N÷5) ]

x, u = MPC.rollout(discrete_dynamics, x0, u, 0, 0)
plot(x', layout=3)
##
oi = ObjectiveInput(x, u, r)

observer = StateFeedback(discrete_dynamics, x0)

solver = IpoptSolver(;
        verbose                    = true,
        tol                        = 1e-8,
        acceptable_tol             = 1e-5,
        max_iter                   = 2000,
        max_cpu_time               = 60.0,
        max_wall_time              = 60.0,
        constr_viol_tol            = 1e-8,
        acceptable_constr_viol_tol = 1e-5,
        acceptable_iter            = 10,
)


prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    Ts,
    objective,
    constraints = [stage_constraint, terminal_constraint],
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
    verbose = true,
    # jacobian_method = :symbolics,
    # discretization = :collocation,
    discretization = Trapezoidal(; dyn = dynamics),
);

using Plots
x_sol, u_sol = get_xu(prob)
fig = plot(
    plot(x_sol', title="States", lab=permutedims(state_names(dynamics)), layout=(nx,1)),
    plot(u_sol', title="Control signal", lab=permutedims(input_names(dynamics))),
)

fig


# con = prob.constraints.constraints[2]
# C = zeros(length(con))
# xn = [reshape(repeat(x[:, 1:end-1], 5, 1), nx,:) x[:, end]]
# oi2 = prob.objective_input
# MPC.evaluate!(C, con, oi2, 0, 0)