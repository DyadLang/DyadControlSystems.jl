using JSONSchema
using RelocatableFolders
using Serialization
using Plots
using ControlSystemsBase
using DataFrames

abstract type AbstractLinearAnalysisSpec <: AbstractAnalysisSpec end

export AbstractLinearAnalysisSpec, LinearAnalysisSpec

"""
    LinearAnalysisSpec{M}

Linear analysis of a plant model. This analysis performs linearization of the provided model
and provides tools for frequency and time domain analysis, including bode plots, mode shapes,
damping report, and step response.

# Arguments:
- `name::Symbol`: The name of the analysis specification.
- `model::M`: The model to be analyzed.
- `inputs::Vector{String}`: Names of the input variables.
- `outputs::Vector{String}`: Names of the output variables.
- `wl::Real`: Lower frequency bound for Bode plot (optional, default -1 for auto).
- `wu::Real`: Upper frequency bound for Bode plot (optional, default -1 for auto).
- `num_frequencies::Int`: Number of frequency points (default 3000).
- `duration::Real`: Duration for the step response plot (default -1 for auto).

# Result visualization
The analysis artifact displays:
- `:BodePlot`: Bode plot of the system.
- `:MarginPlot`: Bode plot with gain and phase margins.
- `:StepResponse`: Step response of the system.
- `:DampReport`: DataFrame with damping report showing modal analysis results.
- `:StepInfo`: DataFrame with step response characteristics (see [`stepinfo`](@ref) for more info).
- `:StepInfoPlot`: Plot of step response characteristics.
- `:RootLocusPlot`: Root locus plot of the system.
- `:PoleZeroMap`: Pole-zero map of the system.
- `:NyquistPlot`: Nyquist plot of the system.
- `:RGAPlot`: Relative Gain Array (RGA) plot of the system.
"""
@kwdef struct LinearAnalysisSpec{M,WL,WU,DU} <: AbstractLinearAnalysisSpec
    name::Symbol
    model::M
    inputs::Vector{String}
    outputs::Vector{String}
    wl::WL = -1
    wu::WU = -1
    num_frequencies::Int = 3000
    duration::DU = -1
end

Base.nameof(spec::LinearAnalysisSpec) = spec.name

function Base.show(io::IO, m::MIME"text/plain", spec::LinearAnalysisSpec)
    print(io, "Linear Analysis specification for ")
    printstyled(io, "$(nameof(spec))\n", color = :green, bold = true)
    println(io, "inputs: ", spec.inputs)
    println(io, "outputs: ", spec.outputs)
    println(io, "wl: ", spec.wl)
    println(io, "wu: ", spec.wu)
    println(io, "num_frequencies: ", spec.num_frequencies)
end

function setup_prob(spec::LinearAnalysisSpec)
    # Convert argument types
    inputs = Symbol.(spec.inputs)
    outputs = Symbol.(spec.outputs)

    # Linearize
    sys = named_ss(spec.model, inputs, outputs)
    if spec.wl < 0 || spec.wu < 0
        w = ControlSystemsBase._default_freq_vector(sys, Val{:bode}())
    else
        wl = spec.wl
        wu = spec.wu
        wN = spec.num_frequencies
        w = exp10.(LinRange(log10(wl), log10(wu), wN))
    end
    sys = sminreal(sys)
    return sys, w
end

struct LinearAnalysisSolution{SP <: LinearAnalysisSpec, SYS} <: AbstractAnalysisSolution
    spec::SP
    sys::SYS
    w::AbstractVector
end

function DyadInterface.run_analysis(spec::LinearAnalysisSpec)
    sys, w = setup_prob(spec)
    stripped_spec = @set spec.model = nothing
    LinearAnalysisSolution(stripped_spec, sys, w)
end

function DyadInterface.AnalysisSolutionMetadata(sol::LinearAnalysisSolution)
    plt_names = [
        :BodePlot,
        :MarginPlot,
        :StepResponse,
        :StepInfoPlot,
        :RootLocusPlot,
        :PoleZeroMap,
        :NyquistPlot,
        :RGAPlot,
    ]
    plt_types = [
        ArtifactType.PlotlyPlot,
        ArtifactType.PlotlyPlot,
        ArtifactType.PlotlyPlot,
        ArtifactType.PlotlyPlot,
        ArtifactType.PlotlyPlot,
        ArtifactType.PlotlyPlot,
        ArtifactType.PlotlyPlot,
        ArtifactType.PlotlyPlot,
    ]
    plt_titles = [
        "Bode plot",
        "Gain and phase margins",
        "Step response",
        "Step response characteristics",
        "Root locus plot",
        "Pole-zero map",
        "Nyquist plot",
        "Relative Gain Array (RGA) plot",
    ]
    plt_descriptions = [
        "Frequency response (Bode plot)",
        "Gain and phase margins",
        "Step response",
        "Step response characteristics (stepinfo)",
        "Root locus of the system",
        "Poles and zeros of the system. Poles are drawn as crosses, zeros as circles.",
        "Nyquist plot",
        "Relative Gain Array (RGA) plot.",
    ]

    tbl_names = [
        :DampReport,
        :StepInfo,
        :ObservabilityReport,
    ]
    tbl_types = [
        ArtifactType.DataFrame
        ArtifactType.DataFrame
        ArtifactType.DataFrame
    ]
    tbl_titles = [
        "Damp report (modal analysis)",
        "Step info",
        "Observability report",
    ]
    tbl_descriptions = [
        "Damping report (modal analysis). Poles with negative imaginary parts are omitted from the table.",
        "Step response characteristics (stepinfo)",
        "Various information related to the observability of the system.",
    ]

    names = [plt_names; tbl_names]
    types = [plt_types; tbl_types]
    titles = [plt_titles; tbl_titles]
    descriptions = [plt_descriptions; tbl_descriptions]

    cannedresults = DyadInterface.ArtifactMetadata.(names, types, titles, descriptions)

    AnalysisSolutionMetadata(cannedresults, [])
end

function DyadInterface.artifacts(sol::LinearAnalysisSolution, name::Symbol)
    sys = sol.sys
    w = sol.w
    spec = sol.spec

    if name ∈ (:StepResponse, :StepInfo, :StepInfoPlot)
        duration = if spec.duration > 0
            spec.duration
        else
            Wn, zeta, ps = ControlSystemsBase.damp(sys)
            duration = maximum(zip(Wn, zeta)) do (Wn, zeta)
                t_const = 1 / (Wn * zeta)
                t_const < 1e6 ? 10t_const : 100.0 # Guard against integrators (including close calls)
            end
                
        end
        res = step(sys, duration)
    end


    if name === :BodePlot
        fig = bodeplot(sys, w)
    elseif name === :MarginPlot
        fig = marginplot(sys, w)
    elseif name === :StepResponse
        fig = plot(res, output_names = string.(spec.outputs), input_names = string.(spec.inputs))
    elseif name === :StepInfo
        info = ControlSystemsBase.stepinfo(res)
        df = DataFrame(
            InitialValue = info.y0,
            FinalValue = info.yf,
            StepSize = info.stepsize,
            Peak = info.peak,
            PeakTime_s = info.peaktime,
            Overshoot_pct = info.overshoot * 100,
            Undershoot_pct = info.undershoot * 100,
            SettlingTime_s = info.settlingtime,
            SettlingTimeIndex = info.settlingtimeind,
            RiseTime_s = info.risetime,
        )
        return df
    elseif name === :StepInfoPlot
        fig = plot(stepinfo(res))
    elseif name === :RootLocusPlot
        fig = rlocusplot(sys)
    elseif name === :PoleZeroMap
        fig = pzmap(sys, lab=sys.name, legend=true)
    elseif name === :NyquistPlot
        fig = nyquistplot(sys)
    elseif name === :RGAPlot
        fig = rgaplot(sys, w)
    elseif name === :DampReport
        # Return a DataFrame with columns: Pole, DampingRatio, Frequency_rad_s, Frequency_Hz, TimeConstant_s
        Wn, zeta, ps = ControlSystemsBase.damp(sys)

        pos_im_inds = findall(imag.(ps) .>= 0)
        Wn, zeta, ps = Wn[pos_im_inds], zeta[pos_im_inds], ps[pos_im_inds]
        t_const = 1 ./ (Wn .* zeta)

        df = DataFrame(
            Pole = ps,
            DampingRatio = zeta,
            Frequency_rad_s = Wn,
            Frequency_Hz = Wn ./ (2π),
            TimeConstant_s = t_const
        )
        return df
    elseif name === :ObservabilityReport
        O = obsv(sys)
        nx = sys.nx
        minreal_nx = minreal(sys, 1e-10).nx
        obs = observability(sys)
        unobservable_subspace = nullspace(O)
        df = DataFrame(
            state_dimension = nx,
            minreal_state_dimension = minreal_nx,
            isobservable = obs.isobservable,
            unobservable_subspace = (unobservable_subspace,),
            state_names = (string.(sys.x), ),
            output_names = (string.(sys.y), ),
        )
    else
        error("Unknown canned visualization: $name")
    end
end

