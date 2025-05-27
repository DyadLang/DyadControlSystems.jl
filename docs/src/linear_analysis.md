# Linear analysis
Linear analysis refers to a set of techniques that operate on linear dynamical systems, e.g., linear statespace models on the form
```math
\begin{aligned}
\dot{x}(t) &= Ax(t) + Bu(t) \\
y(t) &= Cx(t) + Du(t)
\end{aligned}
```
or transfer functions. Nonlinear dynamical systems are commonly *linearized* in an operating point of interest in order to obtain a linear model suitable for linear analysis.

Linear analysis is used for a wide range of applications, including
- Frequency-response analysis or modal analysis
- Stability analysis

This page details how to perform a number of common linear analysis tasks using JuliaSim Control.
## Linearization
Linearization refers to the process of approximating a nonlinear dynamical system with a linear dynamical system. Linearization is commonly performed in a stationary point, referred to as an *operating point*, or along a trajectory of the nonlinear system. A nonlinear dynamics model may be implemented directly as a function ``\dot x = f(x, u)``, or as a ModelingToolkit model. The following two section detail how to linearize models in these two cases.

### Differential equations
Linearization of differential equations encoded as Julia functions like
``\dot x = f(x, u, p, t)``
can be performed by simply computing the Jacobians of ``f`` with respect to ``x`` and ``u``, see [Linearizing nonlinear dynamics](https://juliacontrol.github.io/ControlSystems.jl/dev/examples/automatic_differentiation/#Linearizing-nonlinear-dynamics) for an example demonstrating how to do this.

### ModelingToolkit models
ModelingToolkit models can be linearized using the function
```julia
lsys_matrices, ssys = linearize(sys::ODESystem, u::Vector{Num}, y::Vector{Num}; op::Dict)
```
where `u` and `y` denote the inputs and outputs, and `op` is  `Dict` containing the operating point to linearize around. If `op` is not specified, or only specifies some of the variables n `sys`, default values are used for non-specified variables. 

- `lsys_matrices` is a NamedTuple of statespace matrices `A,B,C,D` that can be transformed to a `ControlSystemsBase.StateSpace` object using `ss(lsys_matrices...)` or a linear `ODESystem` using `ModelingToolkitStandardLibrary.Blocks.StateSpace(lsys_matrices...)`.
- `ssys` is a simplified version of the original `sys` that indicates the order of the state variables in the linearized statespace representation.

Internally, [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is used for linearization. To instead obtain symbolic Jacobians, use `ModelingToolkit.linearize_symbolic`.

More details on the linearization in ModelingToolkit is available in [the documentation for ModelingToolkit](https://mtk.sciml.ai/stable/basics/Linearization/).

Sometimes, numerical linearization fails, e.g., if the system to be linearized
- contains discontinuities, in particular at the linearization point (Coulomb friction is a common example)
- throws an error when ForwardDiff.jl is used

In these situations, it might be better to linearize the system using simulation-based methods, such as [Frequency-response analysis](@ref).

A video tutorial on using linearization and analysis points is available below.
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/-XOux-2XDGI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

#### Batch linearization and gain scheduling
See the tutorial [Batch Linearization and gain scheduling](https://juliacontrol.github.io/ControlSystemsMTK.jl/dev/batch_linearization/). In this tutorial, use of the functions [`batch_ss`](https://juliacontrol.github.io/ControlSystemsMTK.jl/dev/api/#ControlSystemsMTK.batch_ss-Tuple) and [`trajectory_ss`](https://juliacontrol.github.io/ControlSystemsMTK.jl/dev/api/#ControlSystemsMTK.trajectory_ss-NTuple{4,%20Any}) for linearization around several operating points, or along a trajectory, is demonstrated.

## Analysis points
Analysis points provide an interface to give names to points of interest in a causal ModelingToolkit model, such as the model of a control system. This allows the user to linearize models and derive, e.g., sensitivity functions and loop-transfer functions with a simple interface. See [Linear analysis with ModelingToolkitStandardLibrary](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) and the video above for more details.

## Frequency-response analysis
Not all dynamical systems are amenable to analytic linearization, and some that technically are, are better to linearize with finite methods. A commonly used technique is frequency-response analysis (FRA), where the systems response to various frequencies are directly measured. The input signal used during FRA can technically be anything, but it is important that the signal is sufficiently exciting in the frequency range of interest. We provide two input methods for FRA, one based on a logarithmic [chirp signal](https://en.wikipedia.org/wiki/Chirp), and one based on a set of sinusoidal inputs. We will illustrate their usages by means of two examples. 

### Sinusoidal input
To perform FRA using sinusoidal inputs, send a frequency vector as second argument to [`frequency_response_analysis`](@ref). The system will be simulated once for each frequency.
```@example fra
using DyadControlSystems, Plots
gr(fmt=:png) # hide
import ModelingToolkitStandardLibrary.Blocks as lib
P0 = ss(tf(1, [1,1,1])) # An example system of second order
P  = lib.StateSpace(ssdata(P0)..., name=:P) # Create a ModelingToolkit StateSpace system
Ts = 0.001 # Sample rate

w = exp10.(LinRange(-1.2, 1, 12)) # Frequency vector
G = frequency_response_analysis(P, w, P.input.u[1], P.output.u[1]; Ts, amplitude=1, settling_time=20, limit_der=true)

bodeplot(P0, w, lab="True response", l=(4, 0.5))
scatter!(G, sp=1, lab="Estimated response")
scatter!(w, rad2deg.(angle.(G.r)), sp=2, lab="Estimated response")
```
The result `G` is a frequency-response data object which represents the response of the system at each frequency in `w`. At this stage, you may obtain a rational transfer function like so:
```@example fra
using ControlSystemIdentification
Ghat = tfest(G, tf(2.0, [1,2,1])) # Provide initial guess that determines the order of the transfer function
```
```@example fra
bodeplot!(Ghat, G.w, lab="Rational estimate", l=:dash)
```


###  Chirp input
If you do not pass a frequency vector, but instead provide the keyword arguments `f0` and `f1`, denoting the initial and final frequencies, a chirp input will be used. This method returns not only the estimated frequency response, but also an estimate of disturbances, $H(i\omega)$ (such as nonlinearities, stochasitcity or transient effects)
```@example fra
P0  = ss(tf(1, [1,1,1]))
P   = lib.StateSpace(ssdata(P0)..., name=:P)
res = frequency_response_analysis(P, P.input.u[1], P.output.u[1]; Ts, amplitude=1, f0=w[1]/2pi, f1 = w[end]/2pi, settling_periods=2)
G = res.G
bodeplot(P0, G.w,  lab="True system", l=(6, 0.5))
plot!(G,     sp=1, lab="Estimate", l=(3, 0.8))
plot!(res.H, sp=1, lab="Disturbance estimate", ylims=(1e-2, Inf))
# And the rational estimate
Ghat = tfest(G, tf(2.0, [1,2,1])) # Provide initial guess that determines the order of the transfer function
bodeplot!(Ghat, G.w, lab="Rational estimate", l=:dash, c=4)
```

You may also estimate a statespace system directly from the data using a subspace-based algorithm. This may be useful for MIMO systems. (Only single input, multiple output is supported by `frequency_response_analysis` at the moment, MIMO systems thus require several calls to the function.)
```@example fra
Gd = c2d(G, Ts) # Perform a bilinear transformation to discrete time frequency vector
Ph, _ = subspaceid(Gd, Ts, 2, zeroD=true) # (G, Ts, nx, ...)
bodeplot!(Ph, G.w, lab="Statespace estimate", l=:dashdot, c=5)
```

Note, the different lines are hard to distinguish in the plot since they appear on top of each other.

### Handling multiple inputs
The analysis methods described above currently only support single-input systems. For multiple inputs, the analysis may be repeated several times. Parametric estimates, such as transfer function and statespace systems, may be appended in the horizontal direction in order to add inputs, e.g. (pseudocode),
```julia
G1f = frequency_response_analysis(..., input1)
G2f = frequency_response_analysis(..., input2)
G1 = subspaceid(G1f)
G2 = subspaceid(G2f)
G = [G1 G2] 
G,_ = balreal(G) # use baltrunc to simplify the model further
```
The last call to `balreal` is optional, it converts the model to a balanced realization, possibly removing unobservable and uncontrollable states in the process.

### Estimating linearity
Before performing linear analysis, it may be useful to determine if a linear model is an accurate description of the dynamics of the system at the operating point and with the input considered. To this end, we may call any of the functions `coherence` or `coherenceplot` (we use the former here to be able to crop the data in the frequency domain before plotting).
```@example fra
ch = coherence(res.d)[0rad:w[end]*rad]
plot(ch, title="Magnitude-squared coherence", yscale=:identity, ylims=(0, 1.01))
```
A coherence close to 1 is a sign that the data describes a linear input-output system. Note, some decrease in coherence towards the edges of the frequency span is expected, it can be reduced by extending the frequency span for the analysis and cropping the resulting estimate. The `FRD` objects can be indexed by frequency like
```julia
G[1Hz : 3Hz]   # Hz
G[1rad : 3rad] # rad/s
```

## Example: Modal analysis of a series of masses and springs
This example will demonstrate how to make use of the function `ModelingToolkit.linearize` to perform modal analysis of a system with several masses (inertias) connected in series. This kind of system is common in engineering applications, it arises as, e.g., 
- Finite approximation of continuous beams.
- In drive trains, where several rotating inertias are connected through flexible shafts. 

We start by creating a function `mass_spring_damper_chain` that allows us to connect several masses, springs and dampers in series.
```@example MODAL
using ModelingToolkit, ModelingToolkitStandardLibrary.Mechanical.Rotational, LinearAlgebra
import ModelingToolkitStandardLibrary.Blocks
connect = ModelingToolkit.connect

function mass_spring_damper_chain(N = 4; input=false, random=false)
    @named r = Blocks.Step(offset=1, start_time=1, height=-1)
    @named t = Torque(use_support=false)
    @named sens = Rotational.AngleSensor()
    systems = input ? [t, sens, r] : [t, sens]
    local m, s, d, eqs
    for i in 1:N
        if random
            m2 = Inertia(J = 10rand(), name = Symbol("m$i"), phi=0.0)
            s2 = Spring(c = 100 * rand(), name = Symbol("s$i"))
            d2 = Damper(d = rand(), name = Symbol("d$i"))
        else
            m2 = Inertia(J = 10, name = Symbol("m$i"), phi=0.0)
            s2 = Spring(c = 100, name = Symbol("s$i"))
            d2 = Damper(d = 1e-2, name = Symbol("d$i"))
        end

        if i == 1
            eqs = if input 
                [ModelingToolkit.connect(t.flange, m2.flange_a); ModelingToolkit.connect(r.output, :u, t.tau)]
            else
                [ModelingToolkit.connect(t.flange, m2.flange_a)]
            end
        end

        # Connect to following spring-damper
        if i < N
            push!(eqs, ModelingToolkit.connect(m2.flange_b, s2.flange_a, d2.flange_a))
        end

        # Connect mass to previous spring-damper
        if 1 < i
            push!(eqs, ModelingToolkit.connect(m2.flange_a, s.flange_b, d.flange_b))
        end
        s = s2
        d = d2
        m = m2
        push!(systems, m)
        if i < N # Do not add any more spring-dampers at the last point
            push!(systems, d)
            push!(systems, s)
        end
    end
    push!(eqs, connect(m.flange_b, sens.flange))
    @named mass_spring_damper_chain = ODESystem(eqs, ModelingToolkit.get_iv(systems[1]);
                                                systems)
end
```
Next, we instantiate a model with `N = 3` masses.
```@example MODAL
using DyadControlSystems, StaticArrays, Plots
N = 3
model = mass_spring_damper_chain(N; input=false) |> complete
```
We call the function `complete` on the model in order to be able to refer to state names using the syntax `model.m1.w` below. With our `ODESystem` model in hand, we may call [`named_ss`](@ref), which internally calls `linearize`, to get a [`NamedStateSpace`](@ref) object


```@example MODAL
op = Dict(model.t.tau.u => 0.0)
lsys = named_ss(model, [model.t.tau.u], [model.m1.w,model.m2.w,model.m3.w]; op)
```
The system does not have a minimal realization, and we may remove one state using [`minreal`](@ref), let's do that before we continue
```@example MODAL
lsys = minreal(lsys)
nothing # hide
```
We may plot the Bode curve of the system to visualize its frequency response
```@example MODAL
w = exp10.(LinRange(-1, 1, 500)) # A frequency vector
bodeplot(lsys, w, xticks=exp10.(LinRange(-1, 1, 5)), plotphase=false, layout=1)
```
We may also hit the system with a hammer and check its time response (impulse response)
```@example MODAL
plot(impulse(lsys, 10))
```


Before we start with the modal analysis, we check the output of [`dampreport`](@ref) to get a quick overview of the systems vibration characteristics:
```@example MODAL
dampreport(lsys)
```

Next up we carry on with the modal analysis. We extract the system matrices, ``A,B,C,D`` so that we may perform eigen analysis on the ``A`` matrix and project the result down to the output space using ``C``. The mode shapes are given by the eigen vectors to the ``A`` matrix, and the frequencies are given by the eigenvalues. Since we are dealing with a real-valued system, each complex eigenvalue is paired with a complex conjugate, we thus access the frequency of the second mode as `freqs[3]` below.
```@example MODAL
A,B,C,D = ssdata(lsys)

lab = string.(permutedims(output_names(lsys)))
fig_modes = plot(real.(C*eigen(A).vectors)[:, 1:2:end]; c=(1:3)', layout=(1,3), plot_title="Mode shapes", lab=permutedims(["Mode $i" for i in 1:3]), m=:o)
hline!([0 0 0], l=(:black, ), primary=false, topmargin=-10Plots.mm)

freqs = abs.(eigvals(lsys.A))

u(x,t) = SA[cos(freqs[1]*t)]
res1 = lsim(lsys, u, 0:0.01:20)
fig_mode1 = plot(res1; ylims=(-0.5, 0.5), layout=1, sp=1, lab, ylabel="Mode 1")

u(x,t) = SA[cos(freqs[3]*t)]
res2 = lsim(lsys, u, 0:0.01:20)
fig_mode2 = plot(res2; ylims=(-0.5, 0.5), layout=1, sp=1, lab, ylabel="Mode 2")

plot(fig_modes, fig_mode1, fig_mode2, layout=(3,1), size=(800, 800))
```
The top figure illustrates the mode shapes. In the first mode, the masses 1 and 3 vibrate in phase, and mass 2 is 180° out of phase. This corresponds to the time-domain plot in the middle, where we simulate the system with a sinusoidal input force at the frequency of the first mode. We also see that the amplitude of the second mass is slightly higher than the amplitudes of masses 1 and 3, which is also indicated by the mode shape. The second mode shape indicates that the first and third masses vibrate out of phase, while the second mass is almost still. This is verified in the last plot, where we see a time-domain simulation with the frequency of the second mode driving the system. The third mode is not visualized in simulation, it corresponds to an integrating mode and is not very interesting (all three masses move together as a single mass, a rigid-body mode). 

For fun, let's also animate the movements of the inertias to illustrate the mode shapes. Visualizing the movement of rotating inertias is not very easy, we thus treat them as linear displacement masses instead, and displace the coordinates slightly to make the movement easier to track by eye. 
```@example MODAL
function plot_msd(res, title)
    y = copy(res.y)
    y .+= 0:size(y, 1)-1 # Move coordinates to make a nice plot
    @gif for i in 1:10:size(y, 2)
        plot([y[1:2, i] y[2:3, i]], zeros(2,2); m=:square, c=:black, ms=7, legend=false, xlims=(-1, 3), ylims=(-0.1, 0.1), size=(400, 100), xticks=false, yticks=false, title, framestyle=:none)
    end
end

plot_msd(res1, "Mode 1")
```
```@example MODAL
plot_msd(res2, "Mode 2")
```

### Control design
Say that we are interested in designing a disturbance-rejection regulator with feedback from a sensor measuring the position of the first mass. Such a system is known to be *passive*, i.e., it is known to have a Nyquist curve that lies entirely in the right half plane. We can verify this with the following three plots
```@example MODAL
w = exp10.(LinRange(-1, 1, 5000))
sys1 = lsys[Symbol("m1₊w(t)"), Symbol("t₊tau₊u(t)")]
plot(
    bodeplot(sys1, w),
    nyquistplot(sys1, w, xlims=(-0.1, 0.1), ylims=(-3, 3)),
    passivityplot(sys1, w),
    layout = (1,3),
    size   = (800, 300),
    legend = false
)
```
These plots all give us indications that the system is passive, in the Bode plot, we see that the phase curve never goes outside of the band ±90°, and in the Nyquist plot we see that the curve lies in the right half plane. In the last plot, the [`passivityplot`](@ref), we see the *passivity index* as a function of frequency. This index is never above 1, indicating that the system is passive for all frequencies. 

The feedback interconnection of two passive systems is known to be passive, we thus have the opportunity to design a *passive controller*, and will automatically have a *robustly stable closed-loop system* no matter how large model errors we have! To design a passive controller, we make use of the function [`spr_synthesize`](@ref) (SPR, Strictly Positive Real).
```@example MODAL
P0 = ExtendedStateSpace(sys1.sys, D21=[0 1], B1 = [sys1.B zeros(sys1.nx)]) # Add a measurement-noise model
# P0.A .-= 1e-6*I(P0.nx)
K, Gcl, sν = spr_synthesize(P0)
ispassive(-K)
```
As we can see, the synthesized controller is passive. We can also check the closed-loop system from input disturbances to the output
```@example MODAL
Gzw = Gcl[1,1]
w = exp10.(LinRange(0, 1, 5000))
plot(
    bodeplot([sys1, Gzw], w, lab=["Open loop" "Closed loop"], plotphase=false, ylims=(1e-3, Inf)),
    nyquistplot(Gzw, w, lab="Closed loop"),
    plot_title = "Input disturbance to output",
    size   = (800, 350),
)
```
Also this system is passive. We have thus designed a robust, passive controller using the knowledge that the system is passive.

For more details on passive controller synthesis, see [$\mathcal{H}_2$ Synthesis of a passive controller](@ref).

## Index
```@index
Pages = ["linear_analysis.md"]
```

## Docstrings
```@autodocs
Modules = [DyadControlSystems]
Pages = ["linear_analysis.jl"]
Private = false
```
