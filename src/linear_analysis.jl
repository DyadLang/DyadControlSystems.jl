using ModelingToolkit
import ModelingToolkit: get_iv
using OrdinaryDiffEq: Rodas4, solve # TODO: Rodas4 bad choice, it can be very inaccurate


"""
    chirp(Ts, f0, f1, Tf; logspace = true)

A [chrip signal](https://en.wikipedia.org/wiki/Chirp) between frequencies (Hz) `f0` and `f1` with sample time `Ts` and duration `Tf` (seconds). `logspace` determines if the frequency change is logarithmic or linear. For experiments, it makes sense to use `logspace=true` to get a similar number of periods of excitation for all frequencies. A chirp with logarithmically spaced frequencies is also called an exponential chirp, or geometric chirp.
"""
function chirp(Ts, f0, f1, Tf; logspace=true)
    t = range(0, step=Ts, stop=Tf)
    N = length(t)
    f = logspace ? exp10.(LinRange(log10(f0), log10(f1), N)) : LinRange(f0, f1, N)
    q = @. sin(2π*f*t)
    reshape(q, :, 1)
end


"""
    chirp(t::Num, f0, f1, Tf; logspace = true)

If `t` is a symbolic variable, a symbolic expression in `t` is returned.
"""
function chirp(t::Union{Num, Sym, Symbolics.Symbolic}, f0, f1, Tf; logspace=true)
    f = logspace ? f0*(f1/f0)^(t/Tf) : f0 + t/Tf*(f1-f0)
    sin(2π*f*t)
end

"""
    G, H, sol, d = frequency_response_analysis(sys::ODESystem, u, y; f0, f1, Tf = 5/f0, Ts = 0.1/f1, amplitude = 1, offset = 0)

Linearize `sys` through simulation. Internally, the system is simulated with an exponential chirp input that sweeps the frequencies from `f0` to `f1`.

The returned system `G` is of type `FRD` and contains the comple response `G.r` and the frequency vector `G.w`.
`H::FRD` is an estimate of the error as a function of frequency. If the error is large, try increasing `Tf` or change the amplitude. 

The default chirp duration `Tf` is 5 periods of the lowest frequency.

The system identification is performed by sampling the output of the system with a frequency `10f1`

# Arguments:
- `sys`: System to be linearized
- `u,y`: input and output of `G` to perform the analysis between. Currently, only SIMO analysis supported.
- `f0`: Start frequency
- `f1`: End frequency
- `Tf`: Duration of the chirp experiment
- `Ts`: The sample time of the identified model.
- `amplitude`: The amplitude of the chirp signal. May be a number of a symbolic expression of `t = ModelingToolkit.get_iv(sys)`.
- `offset`: A bias/offset around which the chirp is oscillating.
- `settling_periods = 2`: number of periods of the lowest frequency to run before the chirp starts.
- `solver = Rodas4()`
- `kwargs...` are passed to `solve`.

# Example
```julia
using ModelingToolkitStandardLibrary.Blocks, Plots
P0 = ss(tf(1, [1,1,1]))
P = Blocks.StateSpace(ssdata(P0)..., name=:P)
res = frequency_response_analysis(P, P.u[1], P.y[1], Ts = 0.001, amplitude=1, f0=w[1]/2pi, f1 = w[end]/2pi, settling_periods=2)
G = res.G
bodeplot(P0, G.w)
plot!(G, sp=1)

# At this stage, you may obtain a rational transfer function like so:
using ControlSystemIdentification
Ghat = tfest(G, tf(2.0,[1,2,1])) # Provide initial guess that determines the order of the transfer function
bodeplot!(Ghat, G.w, lab="Rational estimate")
```
"""
function frequency_response_analysis(sys::ODESystem, u, y; # Chirp version
    f0, f1,
    Tf = 5 / f0,
    Ts = 0.1/(5*f1),
    amplitude = 1,
    offset = 0,
    settling_periods = 2,
    solver=Rodas4(),
    kwargs...
)
    f00 = 0.2*f0 # start at 1/5 of the desired lowest frequency to be able to filter out initial transients. The settling time does not appear to be enough (but does help)
    f10 = 5*f1
    T0 = 1/f00 # Period of first frequency
    settling_time = T0*settling_periods
    t = ModelingToolkit.get_iv(sys)
    # TODO: add settling-time, one period of the lowest freq, then iflse the input and if less than settling time, use standard sin, then discard the settling time
    u_chirp = ifelse(t > settling_time,
        amplitude*chirp(t-settling_time, f00, f10, Tf),
        amplitude*sin(2π*f00*t)
    ) + offset
    @named iosys = ODESystem([u_chirp ~ u], t, systems=[sys])
    sysr = structural_simplify(iosys)
    prob = ODEProblem(sysr, Pair[], (0, Tf+settling_time))
    sol = solve(prob, solver; saveat = 0:Ts:(Tf+settling_time), tstops=settling_time, kwargs...)
    yd = sol[y]
    if length(y) > 1
        yd = reduce(hcat, yd) |> transpose
    end
    chirpinds = round(Int, settling_time/Ts) + 1 : length(yd)
    yd = yd#[chirpinds] # Testing indicates that it's beneficial to keep the settling_time data. Both actual freqresp error and estimated `H` is lower if we do
    ud = sol[u]#[chirpinds]
    d = iddata(yd, ud, Ts)
    if offset != 0
        d = detrend(d)
    end
    G, H = tfest(d)
    G = G[f0*Hz : f1*Hz]
    H = H[f0*Hz : f1*Hz]
    (; G, H, sol, d)
end


"""
	G(iω) = frequency_response_analysis(G::ODESystem, Ω::AbstractVector, u, y; kwargs...)

Frequency-response analysis of `G u->y`. Returns the frequency-response \$G(iω)\$ as a
`FRD` object that contains the comple response `G.r` and the frequency vector `G.w`.

Note: this is a time-consuming process.

# Arguments
- `Ω`: A vector of frequencies
- `u,y`: input and output of `G` to perform the analysis between. Currently, only SIMO analysis supported.
- `Ts`: Sample rate
- `settling_time = 2`: In seconds, rounded up to closest integer periods
- `nbr_of_periods = 5`: to collect data from.
- `amplitude = 1`: Scalar or vector of same length as Ω. Very low freqs might require smaller amplitude
- `offset`: A bias/offset around which the signal is oscillating.
- `diff_order = 0`: Order of differentiation to apply to the input signal before estimating the frequency response. This is useful to mitigate problems due to non-stationarity, such as trends and drift. The resulting frequency-response is automatically integrated in the frequency domain to account for the applied differentiation. Try setting `diff_order = 1` if the result is poor for high frequencies when using `offset`.
- `threads = false`: use threads to parallelize the analysis.
- `solver = Rodas4()`
- `kwargs...` are passed to `solve`.

# Example:
```
using ModelingToolkitStandardLibrary.Blocks, Plots
P0 = ss(tf(1, [1,1,1]))
P = Blocks.StateSpace(ssdata(P0)..., name=:P)

w = exp10.(LinRange(-1.2, 1, 12)) # Frequency vector
G = frequency_response_analysis(P, w, P.u[1], P.y[1], Ts=0.001, amplitude=1, settling_time=20, limit_der=true, threads=true)

bodeplot(P0, w, lab="True response")
plot!(G, sp=1, lab="Estimated response")
plot!(w, rad2deg.(angle.(G.r)), sp=2)

# At this stage, you may obtain a rational transfer function like so:
using ControlSystemIdentification
Ghat = tfest(G, tf(2.0,[1,2,1])) # Provide initial guess that determines the order of the transfer function
bodeplot!(Ghat, w, lab="Rational estimate")
```
"""
function frequency_response_analysis(P::ODESystem, Ω::AbstractVector, u, y; # Sin version
            Ts,
			settling_time  = 2,
			nbr_of_periods = 5,
			amplitude 	   = 1,
            offset         = 0,
            threads        = false,
            limit_der      = false,
            diff_order     = 0,
            solver         = Rodas4(),
            kwargs...)

    t = ModelingToolkit.get_iv(P)
    blend(t, tf) = evalpoly(t, (0.0, 0.0, 0.0, 20/2tf^3, -30/2tf^4, 12/2tf^5)) # Fifth order poly from 0-> with zero acc/vel in endopints. This polynomial is used to reduce the jerk in the initial transient which may otherwise excite higher-order dynamics disturbing the analysis in the frequency range of interest.
    integrate(fun,data,ω,h) = h*sum(fun(ω*(i-1)*h)*data[i] for i = eachindex(data))

    # mapfun = threads ? tmap : map
    mapfun = map # NOTE: segfaults in with latest MTK
    # NOTE: this implementation can be made much more efficient by introducing parameters and avoiding structural_simplify and ODEProblem for each frequency

    G = mapfun(eachindex(Ω)) do i
    # Threads.@threads for i = eachindex(Ω)
        ω = Ω[i]
        T = 2π/ω # Period time
        settling_time_i = ceil(Int, settling_time/T)*T # Settling time even number of periods
        A = amplitude isa Real ? amplitude : amplitude[i]
        u_sin = if limit_der
            A*sin(ω*t) * ifelse(t<=settling_time_i/4, blend(t, settling_time_i/4), 1)
        else
            A*sin(ω*t)
        end
        Tf = nbr_of_periods*T + settling_time_i

        @named iosys = ODESystem([u_sin + offset ~ u], t, systems=[P])
        sysr = structural_simplify(iosys)
        prob = ODEProblem(sysr, Pair[], (0, Tf))
        sol = solve(prob, solver; saveat = 0:Ts:Tf, kwargs...)
        
        Gy = map(y) do y
            data = sol[y][(round(Int, settling_time_i/Ts)+1):end]
            for i in 1:diff_order
                data = diff(data) ./ Ts
            end
            sin_channel = integrate(sin, data, ω, Ts)
            cos_channel = integrate(cos, data, ω, Ts)
            2/(A*(Tf-settling_time_i))*complex(sin_channel, cos_channel) / (im*ω)^diff_order
        end
        length(y) == 1 ? Gy[1] : Gy
	end
    if length(y) == 1
        return FRD(Ω, G) 
    else
        G = reshape(reduce(hcat, G), length(y), 1, :) # Make MIMO FRD ny × nu × nω, nu = 1
        return FRD(Ω, G)
    end
end