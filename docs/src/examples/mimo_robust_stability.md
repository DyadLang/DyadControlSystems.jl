# Robustness analysis of a MIMO system

In the following, we show an example of robustness analysis of a MIMO system, adapted from Sec. 8.11.3 in Skogestad, "Multivariable Feedback Control: Analysis and Design".

The process to be controlled is a spinning satellite.
```math
P(s) = \dfrac{1}{s^2+a^2}
\begin{bmatrix}
s - a^2 & a(s+1) \\
-a(s+1) & s-a^2
\end{bmatrix}
, \quad a = 10
```
We start by defining the nominal process model (a minimal realization of $P(s)$ rather than a transfer-function matrix)


```@example satellite
using DyadControlSystems, LinearAlgebra, Plots
gr(fmt=:png) # hide
a = 10
P = ss([0 a; -a 0], I(2), [1 a; -a 1], 0)
w = exp10.(LinRange(-2, 3, 200))
sigmaplot(P, w)
```
As we can see, we have a large peak in the Bode plot, in fact, the system is only marginally stable with two eigenvalues on the imaginary axis (there is little air resistance in space after all).


We proceed to model the uncertainty we have about the system. We'll model a 20% actuator uncertainty at low frequencies, and let this uncertainty grow to 200% at large frequencies. This kind of uncertainty, diagonal, complex input uncertainty, is always present in physical systems.

```@example satellite
W0 = makeweight(0.2, (1,1), 2) |> ss # Weight that goes from 0.2 at low frequencies to 2 at high frequencies
W = I(2) + W0*I(2) * uss([δc(), δc()]) # Create a diagonal complex uncertainty weighted in frequency by W0
Ps = P*W # Uncertain system
```
`Ps` is now represented as a upper linear fractional transform (upper LFT).

We can draw samples from this uncertainty representation (sampling of $\Delta$ and closing the loop `starprod(Δ, Ps)`) like so
```@example satellite
Psamples = rand(Ps, 100)
sigmaplot(Psamples, w)
```

We can also extract the nominal model using

```@example satellite
system_mapping(Ps)
```
And obtain $M$ and $\Delta$ from the $M\Delta$ formulation when the loop is closed with $K$
```@example satellite
K = ss(I(2))
lft(Ps, -K).M
```
```@example satellite
Ps.Δ # Ps.delta also works
```
Most importantly, we can calculate the robustness margin of the system under the modeled uncertainty. For this, we calculate the structured singular value, $\mu$


```@example satellite
μ = mussv(lft(Ps, -K))
```

The value of $\mu$ is very high, whenever $\mu > 1$, the system is not stable with respect to the modeled uncertainty.
The tolerated uncertainty is only $\dfrac{1}{||\mu||_\infty}$

```@example satellite
1/μ
```
of the modeled uncertainty, this number is called the robust stability margin, and we would ideally want it to be $\geq 1$, indicating that we are robust with respect to at least 100% of the modeled uncertainty.

When we call [`mussv`](@ref), an algorithm operating on the statespace representation is used to pinpoint the exact location of the peak structured singular value. Compare this to the grid-based method
```@example satellite
plot(w, structured_singular_value(lft(Ps, -K), w), xscale=:log10, title="μ", xlabel="Frequency")
```


### Unmodeled actuator delay

If we had a known delay in the actuator, we could choose to either model it:
```julia
Pd = P*delay(0.1)
```
creates a model of the system with an input delay of 0.1 seconds. Another approach is to consider the delay as unmodeled dynamics and include an uncertainty description of it. To this purpose, we can create the system as
```@example satellite
Pd = P * (I(2) + neglected_delay(0.1).*I(2)*uss([δc(), δc()]))
sigmaplot(rand(Pd, 100), w)
```

We see that neglecting a delay of no more than 0.1s has some effect on the dynamics around the resonance peak, but the uncertainty becomes rather large after that.

With such an uncertainty, the stability margin over frequencies is
```@example satellite
μ = structured_singular_value(lft(Pd, -K), w)
plot(w, inv.(μ), scale=:log10, lab="Stability margin", xlabel="Frequency")
hline!([1], lab="Stability boundary", l=(:red, :dash))
```

indicating that our primitive feedback controller `K` can just barely handle the uncertainty.


