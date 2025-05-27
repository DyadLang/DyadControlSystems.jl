# Adaptive MPC
In this example, we will design a self-learning MPC controller that adapts to changes in the plant. We will consider a quadruple tank system with two inputs and two outputs, where the effectiveness of one pump is reduced in half halfway into the simulation. 

![process](https://user-images.githubusercontent.com/3797491/166203096-40539c68-5657-4db3-bec6-f893286056e1.png)

This tutorial will use the [`QMPCProblem`](@ref), a nonlinear MPC problem with quadratic cost function. We discretize the continuous-time dynamics using the [`rk4`](@ref) function and set the prediction horizon for the MPC controller to `N = 5`.

We start by defining the dynamics and the MPC parameters, and use the parameter `p` to signal whether or not the change in the pump effectiveness is active or not. Since the problem is nonlinear, we use an [`UnscentedKalmanFilter`](@ref) for state estimation.

```@example ADAPTIVE_MPC
using DyadControlSystems
using DyadControlSystems.MPC
using Plots, LinearAlgebra
using StaticArrays
gr(fmt=:png) # hide
## Nonlinear quadtank
# p indicates whether or not k1 changes value at t = 1000
function quadtank(h, u, p, t)
    kc = 0.5
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a1, a3, a2, a4 = 0.03, 0.03, 0.03, 0.03
    γ1, γ2 = 0.3, 0.3
    if p == true && t > 1000
        k1 /= 2
    end

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0
    SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end

nu = 2 # number of control inputs
nx = 4 # number of state variables
ny = 2 # number of outputs
Ts = 1 # sample time

discrete_dynamics0 = rk4(quadtank, Ts) # Discretize the continuous-time dynamics
state_names = :h^4
measurement = (x,u,p,t) -> 0.5*x[1:2]
discrete_dynamics = FunctionSystem(discrete_dynamics0, measurement, Ts, x=state_names, u=:u^2, y=:h^2)


# Control limits
umin = 0 * ones(nu)
umax = 1 * ones(nu)

# State limits (state constraints are soft by default)
xmin = zeros(nx)
xmax = Float64[12, 12, 8, 8]
constraints = NonlinearMPCConstraints(; umin, umax, xmin, xmax)

x0 = [2, 1, 8, 3]       # Initial state
xr = [10, 10, 4.9, 4.9] # reference state
ur = [0.26, 0.26]

R1 = 1e-5*I(nx) # Dynamics covariance
R2 = 0.1I(ny)   # Measurement covariance

ukf = UnscentedKalmanFilter(discrete_dynamics, R1, R2)

solver = OSQPSolver(
    eps_rel = 1e-6,
    eps_abs = 1e-6,
    max_iter = 5000,
    check_termination = 5,
    sqp_iters = 1,
    dynamics_interval = 1,
    verbose = false,
    polish = true, 
)

N  = 5          # MPC horizon
Q1 = 10I(nx)    # State cost
Q2 = 1.0*I(nu)  # Control cost
qs = 100        # Soft constraint linear penalty
qs2 = 100000    # Soft constraint quadratic penalty

prob = QMPCProblem(discrete_dynamics; observer=ukf, Q1, Q2, qs, qs2, constraints, N, xr, ur, solver, p=false)
```

When we solve the problem, the prediction model uses `p=false`, i.e., it's not aware of the change in dynamics, but the actual dynamics has `p=true` so the simulated system will suffer from the change
```@example ADAPTIVE_MPC
hist = MPC.solve(prob; x0, T = 2000, verbose = false, dyn_actual=discrete_dynamics, p=false, p_actual=true)
plot(hist, plot_title="Standard Nonlinear MPC")
```
At ``t = 1000`` we see that the states drift away from the references and the controller does little to compensate for it. Let's see if an adaptive MPC controller can improve upon this!

To make the controller adaptive, we introduce an additional state in the dynamics. When the observer estimates the state of the system, it will automatically estimate the value of this parameter, and this state will thus serve to estimate the pump effectiveness, and takes the place of the `k1` parameter from before. If we have no idea exactly *how* the change will happen, we could say that the derivative of the unknown parameter is 0, and thus model a situation in which this state is driven by noise only. We'll take this route, but will make the dynamics of the unknown state ever so slightly stable for numerical purposes.
```@example ADAPTIVE_MPC
function quadtank_param(h, u, p, t)
    k2, g = 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a1, a3, a2, a4 = 0.03, 0.03, 0.03, 0.03
    γ1, γ2 = 0.3, 0.3

    k1 = h[5]

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0
    SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
        1e-4*(1.6 - k1) # The parameter state is not controllable, we thus make it ever so slightly exponentially stable to ensure that the Riccati solver can find a solution.
    ]
end
state_names = [:h^4; :k1]
discrete_dynamics_param = FunctionSystem(rk4(quadtank_param, Ts), measurement, Ts, x=state_names, u=:u^2, y=:h^2)
nx = 5 # number of state variables, one more this time due to the parameter state

xmin = 0 * ones(nx)
xmax = Float64[12, 12, 8, 8, 100]
constraints = NonlinearMPCConstraints(; umin, umax, xmin, xmax)

x0 = [2, 1, 8, 3, 1.6]       # Initial state
xr = [10, 10, 4.9, 4.9, 1.6] # reference state, provide a good guess for the uncertain parameter since this point determins linearization.
ur = [0.26, 0.26]

R1 = cat(1e-5*I(nx-1), 1e-5, dims=(1,2)) # Dynamics covariance

ukf = UnscentedKalmanFilter(
    discrete_dynamics_param.dynamics,
    discrete_dynamics_param.measurement,
    R1,
    R2,
    MvNormal(x0, Matrix(R1))
    ;
    nu,
    ny
)
#
Q1 = cat(10I(nx-1), 0, dims=(1,2)) # We do not penalize the parameter state, it is not controllable
prob = QMPCProblem(discrete_dynamics_param; observer=ukf, Q1, Q2, qs, qs2, constraints, N, xr, ur, solver, p=true)
hist = MPC.solve(prob; x0, T = 2000, verbose = false, dyn_actual=discrete_dynamics, x0_actual = x0[1:nx-1], p=false, p_actual=true)
plot(hist, plot_title="Adaptive Nonlinear MPC")
```
This time, the simulation looks much better and the controller quickly recovers after the change in dynamics at ``t = 1000``. 

We can check what value of the estimated parameter the state estimator converged to
```@example ADAPTIVE_MPC
using Test
@test ukf.x[5] ≈ 0.8 atol=0.02
```
which should be close to ``1.6 / 2 = 0.8`` if the observer has worked correctly.