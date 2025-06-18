using ModelingToolkit
using ControlSystemsBase
using Printf

abstract type AbstractClosedLoopSensitivityAnalysisSpec <: AbstractAnalysisSpec end

export AbstractClosedLoopSensitivityAnalysisSpec, ClosedLoopSensitivityAnalysisSpec

"""
    ClosedLoopSensitivityAnalysisSpec{M}

Analyze the closed-loop sensitivity function S = 1/(1+PC) at a specified analysis point.

# Arguments:
- `name::Symbol`: The name of the analysis specification.
- `model::M`: The model to be analyzed.
- `analysis_points::String`: The name of the analysis point where sensitivity is computed.
- `loop_openings::Vector{String}`: Names of loop openings to break feedback if present. Default is an empty vector, meaning no loop openings.
- `wl::Float64`: The lower frequency bound for the analysis. Set to -1 for automatic selection.
- `wu::Float64`: The upper frequency bound for the analysis. Set to -1 for automatic selection.
# Result visualization
The analysis artifact displays:
- `:BodePlot`: Bode plot of the closed-loop sensitivity function S.
"""
@kwdef struct ClosedLoopSensitivityAnalysisSpec{M} <: AbstractClosedLoopSensitivityAnalysisSpec
    name::Symbol
    model::M
    analysis_points::Vector{String} # Array of analysis points
    loop_openings::Vector{String} = String[]
    wl = -1.0  # Lower frequency bound for the analysis
    wu = -1.0  # Upper frequency bound for the analysis
end

Base.nameof(spec::ClosedLoopSensitivityAnalysisSpec) = spec.name

struct ClosedLoopSensitivityAnalysisSolution{SP <: ClosedLoopSensitivityAnalysisSpec} <: AbstractAnalysisSolution
    spec::SP
    S
end

function Base.show(io::IO, m::MIME"text/plain", sol::ClosedLoopSensitivityAnalysisSolution)
    spec = sol.spec
    S = sol.S
    n_analysis_points = length(spec.analysis_points)
    Ms, wMs = hinfnorm2(S)
    ϕ_m = rad2deg(2 * asin(1 / (2 * Ms)))
    g_m = Ms / (Ms - 1)
    println(io, "ClosedLoopSensitivityAnalysisSolution")
    println(io, "Analysis points: $(spec.analysis_points)")
    println(io, "Loop openings: $(spec.loop_openings)")
    println(io, "Frequency bounds: [$(spec.wl), $(spec.wu)]")
    println(io, "H∞ norm ||S(s)||: $(few(Ms))")
    println(io, "Phase margin (disk based): $(few(ϕ_m))°")
    println(io, "Gain margin (disk based): $(few(g_m))")
    return nothing
end

function DyadInterface.run_analysis(spec::ClosedLoopSensitivityAnalysisSpec)
    analysis_points = Symbol.(spec.analysis_points)
    loop_openings = Symbol.(spec.loop_openings)
    # Compute the sensitivity function for the given analysis points (vector of points, but single S)
    S = get_named_sensitivity(spec.model, analysis_points; loop_openings, warn_empty_op=false)
    stripped_spec = @set spec.model = nothing
    ClosedLoopSensitivityAnalysisSolution(stripped_spec, S)
end

function DyadInterface.AnalysisSolutionMetadata(spec::ClosedLoopSensitivityAnalysisSpec)
    plt_names = [
        :BodePlot
        :NyquistPlot
        :MarginPlot
    ]
    plt_types = fill(CannedVisualizationType.PlotlyPlot, 3)
    plt_titles = [
        "Sensitivity function \$S(s)\$"
        "Nyquist plot of \$L(s)"
        "Margin plot of \$L(s)\$"
    ]
    plt_descriptions = [
        "Bode plot (single analysis point) or sigmaplot (multiple analysis points) of the closed-loop sensitivity function S."
        "Nyquist plot of the loop-transfer function"
        "Margin plot of the loop-transfer function"
    ]
    cannedresults = DyadInterface.ArtifactMetadata.(
        plt_names, plt_types, plt_titles, plt_descriptions
    )
    if length(spec.analysis_points) > 1
        cannedresults = cannedresults[1:1] # For MIMO analysis we only support the bode/sigmaplot
    end

    allowed_symbols = []
    AnalysisSolutionMetadata(cannedresults, allowed_symbols)
end

function few(x)
    isinf(x) && return "∞"
    @sprintf("%.1f", x)
end


function DyadInterface.artifacts(sol::ClosedLoopSensitivityAnalysisSolution, name::Symbol)
    spec = sol.spec
    S = sol.S
    n_analysis_points = length(spec.analysis_points)
    Ms, wMs = hinfnorm2(S)
    ϕ_m = rad2deg(2 * asin(1 / (2 * Ms)))
    g_m = Ms / (Ms - 1)

    if spec.wl < 0 || spec.wu < 0
        w = ControlSystemsBase._default_freq_vector(S, Val{:bode}())
    else
        wl = spec.wl
        wu = spec.wu
        wN = 4000
        w = exp10.(LinRange(log10(wl), log10(wu), wN))
    end

    if name == :BodePlot
        if n_analysis_points == 1
            lab = ["\$H_\\infty\$ norm: $(few(Ms)), \$ϕ_m ≥ $(few(ϕ_m))°, g_m ≥ $(few(g_m))\$" ""]
            plt = bodeplot(S, w; title=["Sensitivity Function \$S(s)\$" ""], lab, legend=:bottomright)
            return plt
        else
            lab = "\$H_\\infty\$ norm: $(few(Ms))"
            # Multiple analysis points: show sigmaplot
            plt = sigmaplot(S, w; title="Sensitivity Function singular values \$S(s)\$", legend=:bottomright, lab)
            return plt
        end
    elseif name === :NyquistPlot
        L = inv(S) - I(S.ny)
        return nyquistplot(L.sys, w, Ms_circles=Ms)
    elseif name === :MarginPlot
        L = inv(S) - I(S.ny)
        marginplot(L.sys, w)
    else
        error("Unknown artifact name: $name")
    end
end

# Add additional visualization and metadata functions as needed.
