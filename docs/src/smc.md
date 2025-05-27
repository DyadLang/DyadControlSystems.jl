# Sliding-Mode Control

[Sliding-Mode Control](https://en.wikipedia.org/wiki/Sliding_mode_control) is a nonlinear control technique, sometimes referred to as "model free". SMC can be characterized as a high-gain controller with good robustness properties and easy tuning, capable of rejecting unknown disturbances and robust w.r.t. model errors. The textbook version of a sliding-mode controller typically exhibits a discontinuous control law, large high-frequency gain and "chattering" around the sliding surface, an undesired behavior which more elaborate versions of SMC aim to mitigate.

The design of an SMC controller consists of two main parts:
- Design of the switching function ``σ`` that produces the sliding variable ``s = σ(x, p, t)``.
- Design of the control law ``u(s, t)``.

The sliding surface ``s=0`` must be chosen such that the sliding variable ``s`` exhibits desireable properties, i.e., converges to the desired state with stable dynamics. The control law is chosen to drive the system from any state ``x : s ≠ 0`` to the sliding surface ``s=0``. The sliding surface is commonly chosen as an asymptotically stable system with order equal to the ``n_x-n_u``, where ``n_x`` is the number of states in the system to be controlled, and ``n_u`` is the number of inputs.

DyadControlSystems implements a number of sliding-mode controllers, listed below:
- [`SlidingModeController`](@ref): A standard SMC controller, with user-configurable control law.
- [`SuperTwistingSMC`](@ref): An implementation of the "Super-twisting SMC", a second-order SMC which limits high-frequency gain.

A video tutorial making use of the sliding-mode controllers to control a physical pendulum is available below.
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/RE-6UsNFXow" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

Next, we illustrate the use of these controllers by means of an example.

## Example
In this example, we will simulate the performance of three different sliding-mode controllers on a mass-spring-damper system given by the dynamics
```math
m\ddot q + b \dot q + kq = u + d
```
where ``[q, q̇]`` is the state, ``u`` is the control input and ``d`` is an unknown disturbance. The application is tracking, and the desired reference trajectory ``q_r(t)`` as well as the system parameters are given by
```math
\begin{aligned}
q_r(t) &= \sin(2t)              \\
\dot{q}_r(t) &= 2\cos(2t)       \\
d(t) &= 2 + 2\sin(3t) + \sin(5t) \\
m &= 2                          \\
b &= 5                          \\
k &= 2
\end{aligned}
```
We start by defining the dynamics, the reference and disturbance functions. The disturbance ``d(t)`` is defined for simulation purposes, the controller is not aware of this disturbance other than through its effect on the measurement of the state component ``q``.
```@example SMC
using DyadControlSystems, Plots, StaticArrays
using DyadControlSystems: rk4
gr(fmt=:png) # hide

d(t)   = 2 + 2sin(3t) + sin(5t) # Disturbance
qr(t)  = sin(2t)  # Reference for q
qdr(t) = 2cos(2t) # Reference for qd

function dyn(x, u, p, t)
    m,b,k = 2,5,2
    q, qd = x
    SA[
        qd,
        (-b*qd + -k*q + u[] + d(t))/m
    ]
end
nothing # hide
```

Since the dynamics is of relative degree ``r=2``, we will choose a switching surface corresponding to a stable first-order system (``r-1=1``). We will choose the system
```math
ė = -e
```
which yields the switching variable ``s = ė + e``, encoded in the function ``s = σ(x)``:
```@example SMC
function σ(x, p, t)
    q, qd = x
    e = q - qr(t)
    ė = qd - qdr(t)
    ė + e
end
nothing # hide
```
It's easy to see that if the control law manages to drive the state ``x`` to the surface ``s=0`` and keep it there, the dynamics will be governed by
```math
\begin{aligned}
s &= ė + e = 0 \\
ė &= -e
\end{aligned}
```
i.e., the control error ``e`` will go to zero as a first-order system with time constant 1.

The linear model-based SMC controller uses an error-function rather than a sliding surface, in this case, it looks like this:
```@example SMC
function err(x, p, t)
    q, qd = x
    e = q - qr(t)
    ė = qd - qdr(t)
    [e, ė]
end
```

Next up, we define four kinds of sliding-mode controllers. The first one, labeled "standard", is the textbook version with a discontinuous control law ``u(s) = -k \operatorname{sign}(s)``. The second "smooth" version of SMC is very similar, but uses the smoother control law ``u(s) = -k \tanh(γs)``, where ``γ`` is a parameter that controls the smoothess. The third is model based and thus takes a linear system model as argument. It also takes a specification of the desired poles of the reduced-order system that evolves along the sliding surface. The last controller is a second-order SMC variant called "Super-twisting SMC". This controller can achieve a lower tracking error than the smoothed controller, without the large high-frequency control action of the standard controller.

To simplify simulation of the controllers, we wrap them all in a [`Stateful`](@ref) wrapper. This makes the controllers remember their own state so that we do not have to handle that in the simulation loop. 
```@example SMC
Ts = 0.01 # Discrete sample time
sysc = ss(ControlSystemsBase.linearize(dyn, [0,0], [0], 0, 0)..., [0 1], 0)
sys = c2d(sysc, Ts) # For Linear model-based SMC

smc_standard   = SlidingModeController(σ, (s,t) -> -20*sign(s))   |> Stateful
smc_smooth     = SlidingModeController(σ, (s,t) -> -20*tanh(10s)) |> Stateful
smc_linear     = LinearSlidingModeController(; sys, k=1, ks=10, desired_poles=[0.991], err) |> Stateful
smc_supertwist = SuperTwistingSMC(50, σ, Ts)                      |> Stateful
nothing # hide
```

We now implement a little simulation function that runs a simulation for a fixed duration and saves the results in arrays for later plotting. We discretize the continuous-time dynamics using the [`rk4`](@ref) function. 
```@example SMC
function simulate_smc(; T, Ts, smc::Stateful, dyn, x0, p = 0)
    X = Float64[]
    U = Float64[]
    R = Float64[]
    x = x0
    smc.x = 0 # Reset the state of the stateful controller

    for i = 0:round(Int, T/Ts)
        t = i*Ts
        q, qd = x
        ui = smc(x, p, t)

        push!(X, q)
        push!(R, qr(t))
        push!(U, ui)
        x = dyn(x, ui, p, t)
    end
    t = range(0, step=Ts, length=length(X))
    X, U, R, t
end

function simulate_and_plot(smc)
    T    = 10   # Simulation duration
    ddyn = rk4(dyn, Ts; supersample=2) # Discretized dynamics using RK4
    x0   = SA[0.0, 0.0] # Initial state of the system (not including any state in the controller)
    X, U, R, t = simulate_smc(; T, Ts, smc, dyn=ddyn, x0)
    plot(t, [X U], layout=(1,3), lab=["x" "u"], ylims=[(-1.1, 1.1) (-21, 21)], sp=[1 3], framestyle=:zerolines)
    plot!(t, R, sp=1, lab="r")
    plot!(t, X-R, lab="e", ylims=(-0.3, 0.1), sp=2, framestyle=:zerolines)
end
nothing # hide
```

```@example SMC
fig1 = simulate_and_plot(smc_standard)
plot!(fig1, plot_title="Standard SMC", topmargin=-7Plots.mm)

fig2 = simulate_and_plot(smc_smooth)
plot!(fig2, plot_title="Smooth SMC", topmargin=-7Plots.mm)

fig3 = simulate_and_plot(smc_linear)
plot!(fig3, plot_title="Linear SMC", topmargin=-7Plots.mm)

fig4 = simulate_and_plot(smc_supertwist)
plot!(fig4, plot_title="SuperTwisting SMC", topmargin=-7Plots.mm)

plot(fig1, fig2, fig3, fig4, layout=(4, 1), size=(1000, 600))
```

The simulations indicate that all four controllers do a good job at tracking the reference and rejecting the disturbance. The standard controller suffers from a very large high-frequency content in the control signal, a common problem with the naive SMC controller. The smoothed controller removes the high-frequency control action, at the expense of a slightly larger tracking error. The SuperTwisting SMC has a very low tracking error, while also having limited high-frequency gain, a nice compromise!


## Index

```@index
Pages = ["smc.md"]
```
```@autodocs
Modules = [DyadControlSystems]
Pages = ["smc.jl"]
Private = false
```
