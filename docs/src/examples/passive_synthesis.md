# $\mathcal{H}_2$ Synthesis of a passive controller
In this example, we will design a *passive* controller that optimizes a $\mathcal{H}_2$ objective function. The feedback interconnection of two passive systems is always stable, so if the plant under control is known to be passive, a passive controller is guaranteed to render the feedback loop stable, irrespective of model errors.

The system we will consider is a flexible Euler–Bernoulli beam that requires active vibration reduction. If the sensor and actuator are collocated (placed at the same point), this system is known to be passive. Additional details on this example are given in section 4 of
> James Richard Forbes (2018): Synthesis of strictly positive real H2 controllers using dilated LMIs, International Journal of Control, DOI: 10.1080/00207179.2018.1453615

We start by defining the system, we will truncate the number of modes to 5:

```@example passive
using DyadControlSystems, Plots
gr(fmt=:png) # hide

l = pi      # beam length
ζ = 1/100   # Relative damping
α = 1:5     # mode indices
w2 = α .^ 4 # Squared natural frequencies
As = [
    [0 1; -w2 -2ζ*sqrt(w2)] for w2 in w2
]
A = cat(As..., dims=(1,2))
pa = 0.55*l
ba = sin.(α.*pa)

C2s = [
    [0 ba] for ba in ba
]

C2 = reduce(hcat, C2s)
B2 = C2'
B1 = [B2 0B2]
D12 = [0; 1.9]
D21 = D12'

pe = 0.7*l
# The C1 matrix determines the point along the beam at which we want to dampen vibrations
C1s = [
    [0 sin(α*pe)
    0 0] for α in α
]
C1 = reduce(hcat, C1s)


P0 = ss(A,B1,B2,C1,C2; D12,D21)
nothing # hide
```
We are now ready to perform the synthesis of the controller, for this, we call the function [`spr_synthesize`](@ref) (strictly-positive real)
```@example passive
K, Gcl, sν = spr_synthesize(P0)
ispassive(-K)
```
The returned controller `K` is supposed to be used with positive feedback, hence the minus sign in the call to [`ispassive`](@ref).

In order to satisfy the constraint that the controller must be passive, we will in general sacrifice some performance. We can compare the passive controller to one synthesized without the passivity constraint:
```@example passive
K_trad, Gcl_trad = h2synthesize(P0) # traditional H2 synthesis

h2norm(Gcl), h2norm(Gcl_trad)
```
We did indeed loose some in terms of $\mathcal{H}_2$ loss, but in return, we get a robust controller.

The controllers can be visualized in several ways, let's compare the closed-loop transfer functions

```@example passive
w = exp10.(LinRange(-2, 3, 2000))
opts = (; ylabelfontsize = 8, xlabelfontsize=8, size=(700,1000), leftmargin=5Plots.mm)
bodeplot(Gcl_trad, w; lab="H₂", opts...)
bodeplot!(Gcl, w; lab="Passive", opts...)
```
We want these to be as small as possible, they indicate the transfer from disturbances to vibration output. Notice how the passive controller typically performs slightly worse at dampening vibrations.

If we look at the controller Bode plot, we see that the passive controller uses more gain, but keeps its phase within ±90° at all times.
```@example passive
bodeplot(-[K_trad, K], w; lab = ["H₂" "" "Passive" ""])
```

Passivity of the controller can further be verified in the Nyquist plane, where a passive SISO system has a Nyquist curve that lies completely in the RHP.
```@example passive
plot(
    nyquistplot(-[K_trad, K], w; xlims=(-0.05,0.35), ylims=(-0.3, 0.3), lab = ["H₂" "Passive"]),
    passivityplot(-[K_trad, K], w; yscale=:identity, title="Relative passivity index", lab = ["H₂" "Passive"])
)
```
The second plot shows the *relative passivity index* as a function of frequency. This index is always $<1$ for passive systems.