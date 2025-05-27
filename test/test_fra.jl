using DyadControlSystems
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit
using OrdinaryDiffEq
using Test, LinearAlgebra


P0 = ss(tf(1.2, [1,1,1]))

P = Blocks.StateSpace(; P0.A, P0.B, P0.C, P0.D, name=:P)
w = exp10.(LinRange(-1.2, 2, 24))


## Traditional FRA with sinusoidal input

@time G = frequency_response_analysis(P, w, P.input.u[1], P.output.u[1], Ts = 0.001, amplitude=1, offset=0, settling_time=20, limit_der=true, diff_order=0)
@test sum(abs2, G.r .- freqrespv(P0, w)) < 1e-6
if isinteractive()
    bodeplot(P0, w)
    plot!(G, plotphase=true)
end

# Diff order 1
@time G = frequency_response_analysis(P, w, P.input.u[1], P.output.u[1], Ts = 0.001, amplitude=1, offset=0, settling_time=20, limit_der=true, diff_order=1)
@test sum(abs2, G.r .- freqrespv(P0, w)) < 1e-5
if isinteractive()
    bodeplot(P0, w)
    plot!(G, plotphase=true)
end

# with non-zero offset (less accurate)
@time G = frequency_response_analysis(P, w, P.input.u[1], P.output.u[1], Ts = 0.001, amplitude=1, offset=1, settling_time=20, limit_der=true)
@test sum(abs2, G.r .- freqrespv(P0, w)) < 1e-3

if isinteractive()
    bodeplot(P0, w)
    plot!(G, plotphase=true)
end

# with non-zero offset and diff_order=1
@time G = frequency_response_analysis(P, w, P.input.u[1], P.output.u[1], Ts = 0.001, amplitude=1, offset=1, settling_time=20, limit_der=true, diff_order=1)
@test sum(abs2, G.r .- freqrespv(P0, w)) < 1e-5

if isinteractive()
    bodeplot(P0, w)
    plot!(G, plotphase=true)
end


## With chirp input
@time res = frequency_response_analysis(P, P.input.u[1], P.output.u[1], Ts = 0.001, amplitude=1, offset=0, f0=w[1]/2pi, f1 = w[end]/2pi, settling_periods=2)
G = res.G

@test sum(abs2, G.r .- freqrespv(P0, G.w)) < 1e-4


if isinteractive()
    bodeplot(P0, G.w, lab="True system")
    plot!(G, lab="Estimate")
    plot!(res.H, lab="Disturbance estimate")
end

# with non-zero offset (less accurate)
@time res = frequency_response_analysis(P, P.input.u[1], P.output.u[1], Ts = 0.001, amplitude=1, offset=1, f0=w[1]/2pi, f1 = w[end]/2pi, settling_periods=2)
G = res.G
@test sum(abs2, G.r .- freqrespv(P0, G.w)) < 1e-2


## MIMO
P0 = ss(tf(1, [1,1,1]))
P0 = let (A,B,C,D) = ssdata(P0)
    ss(A,B,I(2),0)
end

P = Blocks.StateSpace(; P0.A, P0.B, P0.C, P0.D, name=:P)
w = exp10.(LinRange(-1.2, 2, 24))


## MIMO Traditional FRA with sinusoidal input
@time G = frequency_response_analysis(P, w, P.input.u[1], collect(P.output.u), Ts = 0.001, amplitude=1, settling_time=20, limit_der=true)

@test size(G.r) == (2,1,24)

@test sum(abs2, G.r .- freqresp(P0, w)) < 1e-4

if isinteractive()
    bodeplot(P0, w, plotphase=false, lab="True system")
    plot!(G, plotphase=false, lab="Estimated response")
end


## MIMO With chirp input

@time res = frequency_response_analysis(P, P.input.u[1], collect(P.output.u), Ts = 0.001, amplitude=1, f0=w[1]/2pi, f1 = w[end]/2pi, settling_periods=2);
G = res.G

@test (size(G.r,1), size(G.r,2)) == (2,1)

@test sum(abs2, G.r .- freqresp(P0, G.w)) < 1e-3

if isinteractive()
    bodeplot(P0, G.w, plotphase=false, lab="True system")
    plot!(G, plotphase=false, lab="Estimated response (chirp)")
end
