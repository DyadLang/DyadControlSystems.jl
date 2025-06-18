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

function DyadInterface.run_analysis(spec::ClosedLoopSensitivityAnalysisSpec)
    analysis_points = Symbol.(spec.analysis_points)
    loop_openings = Symbol.(spec.loop_openings)
    # Compute the sensitivity function for the given analysis points (vector of points, but single S)
    S = get_named_sensitivity(spec.model, analysis_points; loop_openings)
    stripped_spec = @set spec.model = nothing
    ClosedLoopSensitivityAnalysisSolution(stripped_spec, S)
end

function DyadInterface.AnalysisSolutionMetadata(::ClosedLoopSensitivityAnalysisSpec)
    plt_names = [:BodePlot]
    plt_types = [CannedVisualizationType.PlotlyPlot]
    plt_titles = ["Sensitivity Plot"]
    plt_descriptions = ["Bode plot (single analysis point) or sigmaplot (multiple analysis points) of the closed-loop sensitivity function S."]
    cannedresults = DyadInterface.ArtifactMetadata.(
        plt_names, plt_types, plt_titles, plt_descriptions
    )
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
    else
        error("Unknown artifact name: $name")
    end
end

# Add additional visualization and metadata functions as needed.
