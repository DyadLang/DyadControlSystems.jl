using DyadControlSystems
using DyadControlSystems.MPC, StaticArrays

# Dynamics function in continuous time (x,u,p,t) = (states, control inputs, parameters, time)
function cartpole(x, u, p, _)
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


##
nu = 1  # number of control inputs
nx = 4  # number of states
ny = nx # number of outputs (here we assume that all states are measurable)
Ts = 0.02 # sample time
N = 150 # MPC optimization horizon
x0 = zeros(nx) # Initial state
x0[1] = 3 # cart pos
x0[2] = pi*0.5 # pendulum angle
xr = zeros(nx) # reference state

discrete_dynamics = rk4(cartpole, Ts)
Q1 = spdiagm(ones(nx)) # state cost matrix
Q2 = Ts * spdiagm(ones(nu)) # control cost matrix

# Control limits
umin = -5 * ones(nu)
umax = 5 * ones(nu)
# State limits (state constraints are soft by default)
xmin = -50 * ones(nx)
xmax = 50 * ones(nx)

## Define problem structure

observer = MPC.StateFeedback(discrete_dynamics, x0)

solver = OSQPSolver(
    eps_rel = 1e-3,
    max_iter = 500,        # in the QP solver
    check_termination = 5, # how often the QP solver checks termination criteria
    sqp_iters = 1,
    dynamics_interval = 2, # The linearized dynamics is updated with this interval
)
prob = LQMPCProblem(;
    dynamics = discrete_dynamics,
    observer, 
    Q1,
    Q2,
    umin,
    umax,
    xmin,
    xmax,
    N,
    xr,
    solver,
    Ts
)
history = MPC.solve(prob; x0, T = 500, verbose = false)

using Plots
plot(history) # State references are plotted with dashed lines

