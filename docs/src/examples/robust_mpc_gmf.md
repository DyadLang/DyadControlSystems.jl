# Robust MPC tuning using the Glover McFarlane method
In this example, we will show how robust-control tuning methods like the Glover-McFarlane method may be used to tune an MPC controller.

!!! note 
    This example will take a manual approach to the problem to illustrate each step in more detail, the entire procedure is packaged in the form of [`RobustMPCModel`](@ref) and the user may choose to use this simplified interface instead, illustrated in the example [Integral action and robust tuning](@ref).

The procedure is
1. Perform traditional loop shaping on the plant $G$ by selecting weights $W_2, W_1$ to obtain the "shaped plant" $G_s = W_2 G W_1$.
2. Find a robustifying controller for the shaped plant using the Glover-McFarlane method.
3. Obtain cost matrices for the MPC controller that makes the MPC controller mimic the Glover-McFarlane controller when no constraints are active (using inverse optimal control).
4. Use the observer found by the Glover-McFarlane method to estimate the state of to initialize the MPC controller each iteration.

The plant we will consider is a slow first order system with additional, much faster, second-order actuator/sensor dynamics
```math
G(s) = \dfrac{200}{10s + 1} \dfrac{1}{(0.05s + 1)^2}
```

## Preliminary design
The Glover-McFarlane method prompts you to select pre- and post-compensators $W_1, W_2$ that shape the loop-transfer function such that $G_s = W_2 G W_1$ forms the shaped loop-transfer function. See [`glover_mcfarlane`](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/api/#RobustAndOptimalControl.glover_mcfarlane) and the references therein for additional details.

We start by designing such a robust controller. We select $W_1$ as a PI-controller that increases the loop gain for low frequencies, and leave $W_2$ unspecified. 
```@example GMF
using DyadControlSystems, DyadControlSystems.MPC, LinearAlgebra, Plots
using ControlSystems: balance_statespace
gr(fmt=:png) # hide
default(size=(800, 600))
Ts = 0.01 # Sample time
disc = G -> c2d(balance_statespace(ss(G))[1], Ts)  # A function that discretizes a continuous-time model
G  = tf(200, [10, 1])*tf(1, [0.05, 1])^2 |> disc
W1 = tf([1, 2], [1, 1e-2])               |> disc # A PI controller (with numerically stabilized integrator, hence the 1e-2 factor)
gmf = K, γ, info = glover_mcfarlane(G, 1.1; W1) # Robustify the controller
γ
```
The controller achieves a performance level $\gamma$ (lower is better), typically, we want $\gamma$ less than 4 for a robust design.
(Technically, the controller achieves the performance level `inv(ncfmargin(G, K)[1])`, which may be slightly different than $\gamma$).

We can also check the diskmargin, for both the loop shaped controller and the robustified controller
```@example GMF
w = exp10.(LinRange(-2, 2, 200))
plot(diskmargin(G*W1, 0, w), lab="Loop shaping", c=1)
plot!(diskmargin(G*K, 0, w), lab="Robust", c=2)
ylims!((0, 5), subplot=1, yscale=:linear)
```
This figure shows disk-based upper and lower gain margins as well as phase margin. The gain margin is slightly above two for the robust controller while it's dangerously close to 1 for the non-robustified controller. For an introduction to diskmargins, see ["An Introduction to Disk Margins", Seiler et al.](https://arxiv.org/abs/2003.04771). 

The next step is to use the result of `glover_mcfarlane` to automatically tune the MPC controller. To this purpose, we use the function [`inverse_lqr`](@ref) with an argument of type [`GMF`](@ref) (Glover McFarlane)
```@example GMF
sys = info.Gs # Extract the "shaped" plant G*W1
method = GMF(gmf)
Q1,Q2 = inverse_lqr(method)
Q2 = Symmetric(Matrix(1.0*Q2)) 
Q1
```
using this method, $Q_2$ will always be equal to the identity matrix, but $Q_1$ may not be something you would have found by hand tuning.

This gives us the cost matrices for the MPC controller. We also need to construct an observer that uses the observer gains stored in `gmf`.
```@example GMF
x0 = [1.0,2,3,0] # Initial state
H  = -info.Hkf # We need negative feedback
kf = DyadControlSystems.FixedGainObserver(sys, x0, H)
```

Next, we may specify and solve the MPC problem as usual
```@example GMF
nx = sys.nx
nu = sys.nu
N  = 10 # Prediction horizon
r = zeros(nx) # reference state

# Control limits
umin = -20 * ones(nu)
umax =  20 * ones(nu)

# State limits (state constraints are soft by default)
xmin   = -1000 * ones(nx)
xmax   =  1000 * ones(nx)
constraints = MPCConstraints(; umin,umax,xmin,xmax)
solver = OSQPSolver()
T      = 150 # Simulation horizon
model = LinearMPCModel(sys, kf; constraints, x0)
prob   = LQMPCProblem(model; Q1,Q2,N,r,solver)
hist   = MPC.solve(prob; x0, T, noise=0.1)
f1 = plot(hist)
```
The state trajectories shown in the plot represent the state of the shaped plant. In order to simulate how the controller would have performed on the original plant, we may extract the control signal and perform a simulation on the original plant directly:
```@example GMF
u = reduce(hcat, hist.U) # Control signal
plot(lsim(sys, u; x0), plotu=true)
```

Above, we performed a simulation under ideal circumstances, where the prediction model used by the MPC controller was identical to the actual system dynamics. To verify the robustness of the controller, we can simulate a scenario in which the prediction model differs from the actual dynamics (as is usually the case in practice).

The function [`solve`](@ref) takes a keyword argument `dyn_actual` where the true dynamics can be supplied, if it's left out, it defaults to use the same as the prediction model. For simulation purposes, we construct a slightly different plant where the gain is off by 2x (recall the gain margin, this is getting close) and the main time constant is slower than modeled, but the sensor dynamics is faster than modeled:
```@example GMF
Gact  = tf(2*200, [12, 1])*tf(1, [0.05, 1])*tf(1, [0.02, 1]) |> disc 
bodeplot([G, Gact], lab=["Prediction model" "Prediction model" "Actual dynamics" "Actual dynamics"])
```

```@example GMF
Gact = Gact + ss([0], [0], [0], 0, Gact.timeevol) # Add an empty state to the model corresponding to W1
hist2   = MPC.solve(prob; x0 = x0, T, noise=0.1, reset_observer=true, dyn_actual=Gact)
plot!(f1, hist2, l=:dash, c=[1 2 3 4 1 2])
```
The response is worse this time around (dashed), but still not too bad. Without the robust tuning, the controller would have likely made the system unstable, remember, the disk-based gain margin of the loop-shaping controller was much smaller than the gain error (2x) we simulated.

The interested reader might recall that we mentioned the [`nfcmargin`](@ref) above. A theorem from robust control says that the controller $K$ will stabilize all plants $\tilde{G}$ that differ no more than `nfcmargin(G, K)` from $G$ in the $\nu$-gap metric. Let's verify that this holds
```@example GMF
nugap(G, Gact)
```
```@example GMF
ncfmargin(G, K)[1]
```
in this case, the actual plant $G_{act}$ is in fact too different from $G$ for this theorem to prove stability of the system, fortunately, the condition of the theorem is only sufficient but not necessary, and indeed, we have
```@example GMF
ncfmargin(Gact, K)[1]
```


## Linear system verification
Below, we show how to perform the robustness verification without using the MPC controller, we compare the loop-shaping controller $W_1$, without the additional robustification of Glover McFarlane, to the final controller.

The closed-loop transfer function from a load disturbance to the output is given by
```math
PS(s) = \dfrac{G_d(s)}{I + G(s)K(s)} = G_d(s)S(s)
```
where $G_d$ is the disturbance model and $S$ is the sensitivity function.

```@example GMF
Gd = tf(100, [12, 1]) |> disc # Model from load disturbance to output
PSloop = Gd*output_sensitivity(Gact, W1) # Gd / (I + G*W1) is the transfer function from an input disturbance to output
PSrobust = Gd*output_sensitivity(Gact, K)
plot(step(PSloop), lab="loop shaping K", plot_title="Step responses from load disturbance")
plot!(step(PSrobust), lab="robust K")
```

Also this test indicates that the robustified controller performs much better than the PI controller $W_1$.
