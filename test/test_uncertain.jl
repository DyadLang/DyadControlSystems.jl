using DyadControlSystems
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit
using OrdinaryDiffEq
using MonteCarloMeasurements
# t = Blocks.t

k = 1 ± 0.1 #+ 0.1 * Particles(20)
T = 1 ± 0.2 #+ 0.2 * Particles(20)
@named fo = FirstOrder(; k, T)
@named iosys = ODESystem([fo.u ~ 1], t, systems = [fo])
sys = structural_simplify(iosys)
prob = ODEProblem(sys, Pair[fo.u => 1.0, fo.x => 0.0], (0.0, 10.0))
sol = solve(prob, Tsit5(), saveat = 0:0.1:10)
plot(sol, fillalpha = 0, linealpha = 0.8)

