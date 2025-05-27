#=
This example implements an MPC controller for the cartpole system. The continuous dynamics are discretized with RK4 and a quadratic cost function is optimized using Optim.LBFGS
This is a very naive approach, but gives surprisingly okay performance. 
There are no contraints on control signals or states.
=#
using ForwardDiff
using Optim, Optim.LineSearches

"""
    rk4(f::F, Ts)

Discretize `f` using a Runge-Kutta 4 scheme.
"""
function rk4(f::F, Ts) where {F}
    # Runge-Kutta 4 method
    function (x, p, t)
        f1 = f(x, p, t)
        f2 = f(x + Ts / 2 * f1, p, t + Ts / 2)
        f3 = f(x + Ts / 2 * f2, p, t + Ts / 2)
        f4 = f(x + Ts * f3, p, t + Ts)
        x += Ts / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        return x
    end
end

"""
    x, u = rollout(f, x0::AbstractVector, u)
Simulate discrete system `f` from initial condition `x0` and input array `u`.
"""
function rollout(f, x0::AbstractVector, u)
    T = promote_type(eltype(x0), eltype(u))
    x = zeros(T, length(x0), size(u, 2))
    x[:, 1] .= x0
    rollout!(f, x, u)
end

"""
    x, u = rollout!(f, x, u)
Simulate discrete system `f` from initial condition `x[:, 1]` and input array `u`.
Modifies `x,u`.
"""
function rollout!(f, x, u)
    for i = 1:size(x, 2)-1
        x[:, i+1] = f(x[:, i], u[:, i], i) # TODO: i * Ts
    end
    x, u
end

function lq_cost(x, u, Q1, Q2)
    dot(x, Q1, x) + dot(u, Q2, u) # TODO: add terminal cost
end

function objective(disc_dyn, x, u, Q1, Q2)
    # x = zeros(n, N)
    all(isfinite, u) || return Inf
    rollout!(disc_dyn, x, u)
    lq_cost(x, u, Q1, Q2)
end

"""
    dx = cartpole(x, u)

Continuous-time dynamics for the cart-pole system with state `x` and control input `u`.
"""
function cartpole(x, u, p, _=0)
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81

    q  = x[SA[1, 2]]
    qd = x[SA[3, 4]]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp * g * l * s]
    B = @SVector [1, 0]

    qdd = -H \ (C * qd + G - B * u[1])
    return [qd; qdd]
end

# Package everything in a function and run multiple times
"""
    demo_mpc(dynamics::F = cartpole; Ts, N, x0, Q1, Q2, T)

Run an MPC controller for `T` time steps.

# Arguments:
- `dynamics`: The continuous time dynamics, a function f(x,u,p,t)
- `Ts`: Sample time
- `N`: MPC horizon
- `x0`: Initial state
- `Q1`: State penalty
- `Q2`: Control penalty
- `T`: Number of time steps to run the MPC controller.
"""
function demo_mpc(dynamics::F = cartpole; Ts, N, x0, Q1, Q2, T) where {F}

    disc_dyn = rk4(dynamics, Ts)
    # lin_dyn = linearize(disc_dyn)

    # Initial traj
    nx = size(Q1, 1)
    nu = size(Q2, 1)
    u = randn(nu, N)
    x = zeros(nx, N)
    x[:, 1] .= x0
    optfun = let x=x, u=u, disc_dyn=disc_dyn
        u -> objective(disc_dyn, x, u, Q1, Q2)
    end
    rollout!(disc_dyn, x, u)

    # output arrays
    X = Vector{Float64}[]
    U = Vector{Float64}[]

    for t = 1:T
        if t == 50 # Introduce a disturbance for demo purposes
            x[:, 1] .= randn.()
        end
        res = Optim.optimize(
            optfun,
            u,
            LBFGS(
                alphaguess = LineSearches.InitialStatic(alpha = 0.7), # Limit step size to avoid bad behavior
                linesearch = LineSearches.HagerZhang(),
            ),
            Optim.Options(
                store_trace       = true,
                show_trace        = false,
                show_every        = 1,
                iterations        = 50,
                allow_f_increases = false,
                time_limit        = 100,
                x_tol             = 0,
                f_abstol          = 0,
                g_tol             = 1e-3,
                f_calls_limit     = 0,
                g_calls_limit     = 0,
            ),
            # autodiff = :forward,
        )
        u = res.minimizer
        push!(U, u[:, 1]) # save the first control inputs
        push!(X, x[:, 1])
        # Advance time series as initial guess for next optimization
        @views u[:, 1:end-1] .= u[:, 2:end]
        @views u[:, end] .= u[:, end-1]
        @views x[:, 1:end-1] .= x[:, 2:end]
    end

    reduce(hcat, X), reduce(hcat, U)
end

##
nu = 1 # number of controls
nx = 4 # number of states
Ts = 0.1 # sample time
T = 2
x0 = ones(nx) # Initial state
@time X,U = demo_mpc(
    cartpole;
    Ts = 0.1,
    N = round(Int, T / Ts),
    x0,
    Q1 = Diagonal(@SVector ones(nx)), # state cost matrix
    Q2 = 0.1Diagonal(@SVector ones(nu)), # control cost matrix
    T = 10,
)

# X, _ = rollout(disc_dyn, x0, U)

plot(plot(X'), plot(U'))
