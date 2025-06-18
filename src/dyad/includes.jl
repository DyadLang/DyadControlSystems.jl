gen_path = joinpath(@__DIR__, "..", "..", "generated")

if isdir(gen_path) 
    try
        include(joinpath(gen_path, "types.jl"))
        include(joinpath(gen_path, "definitions.jl"))
        include(joinpath(gen_path, "tests.jl"))
        include(joinpath(gen_path, "experiments.jl"))
    catch
    end
end

#=
using DyadInterface, Plots
asol1 = DCMotorTuning()
plot(asol1.sol)
asol2 = DCMotorLinearAnalysis()
asol3 = DCMotorClosedLoopAnalysis()
gangoffourplot(asol3.P, asol3.C)
asol4 = DCMotorClosedLoopSensitivityAnalysis()
fig = artifacts(asol4, :BodePlot); plot!(legend = :bottomright)
=#