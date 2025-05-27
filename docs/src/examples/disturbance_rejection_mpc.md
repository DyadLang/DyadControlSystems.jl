# Disturbance modeling and rejection with MPC controllers

This example will demonstrate how you can add disturbance models to ta plant model and achieve effective disturbance rejection using an MPC controller. For simplicity, we will consider a simple first-order system $G$

```math
\begin{aligned}
\dot{x} &= -ax + b(u + d) \\
y &= cx
\end{aligned}
```

where a load disturbance $d$ is acting on the input of the system. This is a simple and very common model for load disturbances. In this example, we will let $d$ be a unit step at time $t=10$.

We will begin by setting up the MPC problem and solve it without andy disturbance model. For details regarding the setup of an MPC problem, see, the MPC documentation.

We start by defining the process model and discretize it using zero-order hold.

```@example MPC_DIST
using DyadControlSystems, DyadControlSystems.MPC, Plots, LinearAlgebra
gr(fmt=:png) # hide
Ts = 1 # Sample time
G = c2d(ss(tf(1, [10, 1])), Ts) # Process model
```

```@example MPC_DIST
nx  = G.nx
nu  = G.nu
ny  = G.ny
N   = 10 # Prediction horizon
x0  = zeros(G.nx) # Initial condition
r   = zeros(nx)   # reference state

# Control limits
umin = -1.1 * ones(nu)
umax = 1.1 * ones(nu)
constraints = MPCConstraints(; umin, umax)

solver = OSQPSolver(
    verbose           = false,
    eps_rel           = 1e-6,
    max_iter          = 15000,
    check_termination = 5,
    polish            = true,
)

Q1 = 100spdiagm(ones(G.nx)) # state cost matrix
Q2 = 0.01spdiagm(ones(nu))  # control cost matrix

kf = KalmanFilter(ssdata(G)..., 0.001I(nx), I(ny))
model = LinearMPCModel(G, kf; constraints, x0)
prob = LQMPCProblem(model; Q1, Q2, N, r, solver)

disturbance = (u, t) -> t * Ts ≥ 10 # This is our load disturbance
hist = MPC.solve(prob; x0, T = 100, verbose = false, disturbance, noise = 0)
plot(hist, ploty = true)
```

As we can see, the controller appears to do very little to suppress the disturbance. The problem is that the observer does not have a model for such a disturbance, and its estimate of the state will thus be severely biased.

The next step is to add a disturbance model to the plant model. Since the disturbance if of low-frequency character (indeed, its transfer function is $1/s$), we make use of the function `add_low_frequency_disturbance`

```@example MPC_DIST
Gd = add_low_frequency_disturbance(G, ϵ = 1e-6) # The ϵ moves the integrator pole slightly into the stable region
nx = Gd.nx
```

There is no point trying to penalize the disturbance state in the MPC optimization, it's not controllable, we thus penalize the output only, which we can write as

```math
y^T Q_1 y = (Cx)^T Q_1 Cx = x^T (C^T Q_1C) x
```

```@example MPC_DIST
C  = Gd.C
Q1 = 100C'diagm(ones(G.nx)) * C # state cost matrix
x0 = zeros(nx)
r  = zeros(nx)
nothing # hide
```

We also create a new Kalman filter where the entry of the state-covariance matrix that corresponds to the disturbance state (the second and last state) determines how fast the Kalman filter integrates the disturbance. We choose a large value, implying fast integration

```@example MPC_DIST
kf   = KalmanFilter(ssdata(Gd)..., diagm([0.001, 1]), I(ny))
model = LinearMPCModel(Gd, kf; constraints, x0)
prob = LQMPCProblem(model; Q1, Q2, N, r, solver)
@time hist = MPC.solve(prob; x0, T = 100, verbose = false, disturbance, noise = 0)
plot(hist, ploty = true)
ylims!((-0.05, 0.3), sp = 1)
```

This time around we see that the controller indeed rejects the disturbance and the control signal settles on -1 which is exactly what's required to counteract the load disturbance of +1.

Before we feel confident about deploying the MPC controller, we investigate its closed-loop properties.

```@example MPC_DIST
lqg = LQGProblem(Gd, Q1, Matrix(Q2), kf.R1, kf.R2)
w = exp10.(LinRange(-3, log10(pi / Ts), 200))
gangoffourplot(lqg, w, lab = "", legend = :bottomright)
```

We see that our design led to a system with a rather high peak in sensitivity. This is an indication that we perhaps added too much "integral action" by a too fast observer pole related to the disturbance state. Let's see how a slightly more conservative design fares:

```@example MPC_DIST
kf   = KalmanFilter(ssdata(Gd)..., diagm([0.001, 0.1]), I(ny))
model = LinearMPCModel(Gd, kf; constraints, x0)
prob = LQMPCProblem(model; Q1, Q2, N, r, solver)
@time hist = MPC.solve(prob; x0, T = 100, verbose = false, disturbance, noise = 0)
f1 = plot(hist, ploty = true)
ylims!((-0.05, 0.3), sp = 1)
lqg = LQGProblem(Gd, Q1, Matrix(Q2), kf.R1, kf.R2)
f2 = gangoffourplot(lqg, w, lab = "", legend = :bottomright)
plot(f1, f2, titlefontsize=10)
```

We see that we now have a slightly larger disturbance response than before, but in exchange, we lowered the peak sensitivity and complimentary sensitivity from (1.5, 1.25) to (1.25, 1.07), a much more robust design. We also reduced the amplification of measurement noise ($CS = C/(1+PC)$). To be really happy with the design, we should probably add high-frequency roll-off as well.
