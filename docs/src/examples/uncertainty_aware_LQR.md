# Uncertainty-aware LQR
This example demonstrates how an LQR controller that is aware of the uncertainty in a system model is more robust than one that is not aware of the uncertainty.

The nominal plant model in this example is a third-order resonant process
```math
P(s) = \dfrac{1}{s(s^2 + \omega s + \omega^2)}
```
where we are uncertain about ``\omega``, maybe because it can change with time etc.

We first construct the nominal plant model, and then use [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl) to construct an uncertain parameter ``\omega``, which we use in the uncertain plant model ``P_u``.

```@example uncertainty_aware_LQR
using DyadControlSystems, MonteCarloMeasurements, Plots, LinearAlgebra
gr(fmt=:png) # hide
unsafe_comparisons(true; verbose=false)
ω = 1
P = tf(1.0, [1, ω, ω^2, 0]) |> ss

N  = 6 # Number of uncertain realizations
ωᵤ = Particles(N, Uniform(0.5, 1.5)) # ±50% uncertainty
Pᵤ = tf(1.0, [1, ωᵤ, ωᵤ^2, 0]) |> ss 
```

We then design an LQR controller with a feedback gain ``L`` and a Kalman filter gain ``K`` for the nominal plant model ``P``. We also design an LQR controller with a feedback gain ``L_u`` and a Kalman filter gain ``K_u`` for the uncertain plant model ``P_u`` by calling the same two design functions, [`lqr`](@ref) and [`kalman`](@ref). When called with an uncertain model, these two functions will automatically return feedback gains that have been designed with consideration of the uncertainty in the model.
```@example uncertainty_aware_LQR
Q = diagm(ones(P.nx))
R = [1.0;;]

L  = lqr(P, Q, R)
Lᵤ = lqr(Pᵤ, Q, R)

K  = kalman(P, Q, R)
Kᵤ = kalman(Pᵤ, Q, R)

C  = observer_controller(P, L, K)
Cᵤ = observer_controller(P, Lᵤ, Kᵤ) # We use nominal plant here, we can't have an uncertain controller
nothing # hide
```

To evaluate the resulting closed-loop systems, we form the gang-of-four transfer functions and plot their bode plots
```@example uncertainty_aware_LQR
G  = extended_gangoffour(Pᵤ, C) # We evaluate both controllers with the uncertain plant
Gᵤ = extended_gangoffour(Pᵤ, Cᵤ)
w  = exp10.(LinRange(-2, 2, 300))
kwargs = (; plotphase=false, ri=false, N)
bodeplot(G, w; lab="Nominal design", title=["S" "PS" "CS" "T"], c=1, kwargs...)
bodeplot!(Gᵤ, w; lab="Uncertainty-aware design", legend=:bottomleft, c=2, kwargs...)
```
The Bode plots show a slightly lower peak in the sensitivity function ``S`` for the uncertainty-aware controller (orange), but a more interesting view is the Nyquist plot:
```@example uncertainty_aware_LQR
kwargs = (; points=true, ylims=(-3,1), xlims=(-3,3), markeralpha=0.7, markerstrokewidth=[0.0 1])
nyquistplot(Pᵤ*C, w; lab="Nominal design", c=1, Ms_circles=[2], unit_circle=true, kwargs...)
nyquistplot!(Pᵤ*Cᵤ, w; lab="Uncertainty-aware design", legend=:bottomleft, c=2, kwargs...)
```
Here we see that the controller designed for the nominal plant model has a change of being unstable, one realization is encircling the critical point! The uncertainty-aware controller, on the other hand, is stable for all the uncertain realizations in ``P_u``.

## Use for nonlinear and time-varying systems
Weakly nonlinear systems can be treated as uncertain linear systems by linearizing them "everywhere", i.e., in sufficiently many points to cover the relevant parts of the state space. The [`lqr`](@ref) and [`kalman`](@ref) will internally attempt to stabilize not only all realizations in the uncertain model separately, but any arbitrary and infinitely fast switching between all the realizations. If the [`lqr`](@ref) function returns the status `OPTIMAL`, the state-feedback gain will stabilize the system no matter how fast it switches between any of the realizations in ``P_u``.