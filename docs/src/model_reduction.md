# Model reduction

Model reduction refers to a reduction of complexity, sometimes by approximation, of a model by a simpler model. Many control-design workflows benefit from model reduction, some examples are
- A plant model is derived by linearization of a detailed model implemented in ModelingToolkit. Such a model may contain 1000s of states, but may be accurately described in a particular operating point by a low-order linear model.
- When only subsets of all inputs and outputs of a large MIMO model are required, some states of the original model may no longer be required.
- Some control-design techniques, such as $\mathcal{H}_\infty$ design or [`glover_mcfarlane`](@ref) may result in high-order controllers. Such a controller may sometimes be reduced while maintaining most of the stability margin.
- Some modes in a large model may be unobservable or uncontrollable.

Reduced-order models require less effort to simulate and may improve the numerical performance of some algorithms. Model reduction of linear time-invariant models is a well-developed field, and this page lists some of the available functionality. For reduction of nonlinear models, consider a linearization-based approach, or [build a nonlinear surrogate model](https://help.juliahub.com/juliasimsurrogates/stable/).


Model reduction using balanced truncation is available through the functions
- [Model reduction GUI](@ref), available as [`DyadControlSystems.app_modelreduction`](@ref).
- [`minreal`](@ref) obtain a minimal realization containing only controllable and observable modes.
- [`sminreal`](@ref) obtain a structurally minimal realization of a state-space model.
- [`baltrunc2`](@ref) for standard model reduction.
- [`frequency_weighted_reduction`](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/api/#RobustAndOptimalControl.frequency_weighted_reduction) for reduction with frequency focus.
- [`baltrunc_coprime`](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/api/#RobustAndOptimalControl.baltrunc_coprime-Union{Tuple{Any},%20Tuple{F},%20Tuple{Any,%20Any}}%20where%20F) for normalized-coprime factor reduction.
- [`baltrunc_unstab`](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/api/#RobustAndOptimalControl.baltrunc_unstab) for unstable models.
- [`controller_reduction`](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/api/#RobustAndOptimalControl.controller_reduction) reduce the order of controllers with guaranteed stability margins. See [example](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/#Example-of-controller-reduction:).



## Reduction of very large models
Balanced truncation requires the solution to a Lyapunov equation which may be prohibitively expensive for large systems. For systems of order above about 500, a method based on frequency-domain fitting may be substantially faster if the desired model order is less than about 100. The following example illustrates the procedure.
```@example
using DyadControlSystems, ControlSystemIdentification, Plots
gr(fmt=:png) # hide
ny,nu,nx = 5,5,1000                     # number of outputs, inputs and states
Ts = 1                                  # Sample time
G = ssrand(ny,nu,nx; Ts, proper=true);  # Generate a random system

N = 200                                 # Number of frequency points
w = range(0, stop=pi/Ts-1/N, length=N)  # Frequency vector

frd = FRD(w, G);                        # Build a frequency-response data object
nxr = 60                                # Reduced model order
@time Gh, x0 = subspaceid(frd, G.Ts, nxr; r=nxr+1, zeroD=true); # Fit frequency response

sigmaplot([G, Gh], w[2:end], lab=["Full order" "Reduced order"])
```

The frequency-fitting method does not have support for exact DC matching like the balanced-truncation method does, but there exists an option for frequency-based weighting which can achieve similar results. See [`subspaceid`](https://baggepinnen.github.io/ControlSystemIdentification.jl/stable/ss/#ControlSystemIdentification.subspaceid) for additional details.
