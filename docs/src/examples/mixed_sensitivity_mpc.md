# Mixed-sensitivity $\mathcal{H}_2$ design for MPC controllers

This example will demonstrate how you can utilize the mixed-sensitivity $\mathcal{H}_2$ design methodology to augment a plant model and achieve effective disturbance rejection using an MPC controller. For simplicity, we will consider a simple first-order system $G$

```math
\begin{aligned}
\dot{x} &= -ax + b(u + d) \\
y &= cx
\end{aligned}
```

where a load disturbance $d$ is acting on the input of the system. This is a simple and very common model for load disturbances. In this example, we will let $d$ be a unit step at time $t=10$.

We will begin by setting up the MPC problem and solve it without any disturbance model. For details regarding the setup of an MPC problem, see, the MPC documentation.


## Standard controller
We start by defining the process model and discretize it using zero-order hold.

```@example MIXSYN
using DyadControlSystems, DyadControlSystems.MPC, Plots, LinearAlgebra
gr(fmt=:png) # hide
Ts = 1 # Sample time
disc(G) = c2d(ss(G), Ts)
G = tf(1, [10, 1]) |> disc # Process model
```
This gave us a discrete-time statespace model that we can use to construct the MPC controller. The next step is to define the controller, we define the prediction horizon $N$ and the initial condition $x_0$. We also define the reference state $r$ and the control limits $u_{\min}, u_{\max}$ using an object of type [`MPCConstraints`](@ref). To solve the problem, we will use the OSQP solver, which is a quadratic programming solver that is well suited for MPC problems. To estimate the state of the system, which is linear, we use a [`KalmanFilter`](@ref). The plant model `G` and the Kalman filter are combined into a [`LinearMPCModel`](@ref) object that is used to construct the MPC problem.

```@example MIXSYN
nx  = G.nx
nu  = G.nu
ny  = G.ny
N   = 10 # Prediction horizon
x0  = zeros(G.nx) # Initial condition
r  = zeros(nx)    # reference state

# Control limits
umin = -1.1 * ones(nu)
umax = 1.1 * ones(nu)
constraints = MPCConstraints(; umin, umax)

solver = OSQPSolver(
    verbose           = false,
    eps_rel           = 1e-10,
    max_iter          = 15000,
    check_termination = 5,
    polish            = true,
)

Q1 = 100spdiagm(ones(G.nx)) # state cost matrix
Q2 = 0.01spdiagm(ones(nu))  # control cost matrix

kf = KalmanFilter(ssdata(G)..., 0.001I(nx), I(ny))
model = LinearMPCModel(G, kf; constraints, x0)
prob = LQMPCProblem(model; Q1, Q2, N, r, solver)

disturbance = (u, t) -> 1#t * Ts ≥ 10 # This is our load disturbance
hist = MPC.solve(prob; x0, T = 100, verbose = false, disturbance, noise = 0)
plot(hist, ploty = true)
```

As we can see, our initial controller appears to do very little to suppress the disturbance. The problem is that the observer (Kalman filter) does not have a model for such a disturbance, and its estimate of the state will thus be severely biased.

## Mixed-sensitivity $\mathcal{H}_2$ controller

The next step is to design the performance weights, the function `hinfpartition` is helpful in creating a plant model that contains all the necessary performance outputs. We select the weights $W_U$ and $W_S$ in order to minimize the norm
```math
\begin{Vmatrix}
W_S S \\
W_U CS
\end{Vmatrix}_2
```
where $S$ is the sensitivity function and $C$ is the controller transfer function. The function `hinfpartition` forms a system $P$ such that $\operatorname{lft}_l(P, C)$ is the transfer function we ar minimizing the norm of.


```@example MIXSYN
WS = makeweight(1000, (.03, 5), 1)*tf(1,[0.1, 1])    |> disc
WU = 0.01makeweight(1e-4, 1, 10)                     |> disc
Gd = hinfpartition(G, WS, WU, [])
lqg = LQGProblem(Gd)
nothing # hide
```

Already at this stage, it's a good idea to verify the closed-loop properties of the system, we do this by plotting the relevant sensitivity functions.
```@example MIXSYN
S,_,CS,T = RobustAndOptimalControl.gangoffour(lqg)
specificationplot([S,CS,T], [WS,WU,[]], wint=(-5, log10(pi/Ts)))
```
In the "specification plot" we see the achieved sensitivity functions by the designed controller as well as the *inverse* of the weighting functions. We may also use the function `gangoffourplot` to show each sensitivity function in a separate pane together with relevant peak values:
```@example MIXSYN
w = exp10.(LinRange(-3, log10(pi / Ts), 200))
gangoffourplot(lqg, w, lab = "", legend = :bottomright)
```

We see that the design appears to be robust with low peaks in the sensitivity functions and high-frequency roll-off limiting the noise gain at high frequencies.

We may now extract the cost matrices $Q_1, Q_2$ for the MPC problem and the feedback gain for the Kalman filter from the `lqg` object and form the MPC problem:
```@example MIXSYN
(; Q1,Q2) = lqg
K = kalman(lqg) # Kalman gain
Gs = -system_mapping(Gd, identity) # The - is due to the sign convention in hinfpartition
nx = Gs.nx
x0 = zeros(nx)
kf = FixedGainObserver(Gs, x0, -K)
r  = zeros(nx)
model = LinearMPCModel(Gs, kf; constraints, x0)
prob = LQMPCProblem(model; Q1, Q2, N, r, solver)
nothing # hide
```

When we simulate, we provide the actual dynamics `G` as well as `Cz_actual` that indicates that we measure actual performance in terms of the original output of `G` only (this is the first state in the augmented plant `Gd`).
```@example MIXSYN
x0 = zeros(G.nx)
@time hist = MPC.solve(prob; x0, T = 100, verbose = false, disturbance, noise = 0, dyn_actual=G, Cz_actual = [G.C; 0; 0; 0])
plot(hist, ploty = true)
```

This time around we see that the controller indeed rejects the disturbance and the control signal settles on -1 which is exactly what's required to counteract the load disturbance of +1.

## Concluding remarks

The astute reader might have noticed that we did not use a `KalmanFilter` as the observer when we used mixed-sensitivity tuning of the controller. The `KalmanFilter` type does not support the cross-term between dynamics and measurement noise, but this term is required in order for the LQG problem to be equivalent to the $\mathcal{H}_2$ problem. Hence, we calculate the infinite-horizon Kalman gain using a Riccati solver that supports the cross term and use the fixed-gain observer instead. The cross-term is available as
```@example MIXSYN
lqg.SR
```