using JSONSchema
using RelocatableFolders
using Serialization
using Plots

# const ClosedLoopAnalysisSchema = Schema(read(
#     open(@path(joinpath(@__DIR__, "..", "..", "assets", "ClosedLoopAnalysis.json")), "r"), String))

abstract type AbstractClosedLoopAnalysisSpec <: AbstractAnalysisSpec end

export AbstractClosedLoopAnalysisSpec, ClosedLoopAnalysisSpec


"""
    ClosedLoopAnalysisSpec{M, WL, WU}

Analyze the closed-loop properties of the feedback interconnection depicted below through linearization.
```
              d             
     ┌─────┐  │  ┌─────┐    
r  e │     │u ▼  │     │ y  
──+─►│  C  ├──+─►│  P  ├─┬─►
  ▲  │     │     │     │ │  
 -│  └─────┘     └─────┘ │  
  │                      │  
  └──────────────────────┘  
```

The analysis requires specification of analysis points in the model, corresponding to the measurement signal `y` and the control input `u`.

# Arguments:
- `name::Symbol`: The name of the analysis specification.
- `model::M`: The model to be analyzed.
- `measurement::Vector{String}`: The name of the analysis point or points located where the plant measurement that is fed back to the controller is located.
- `control_input::Vector{String}`: The name of the analysis point or points located where the controller output is fed to the plant.
- `wl::WL`: The lower frequency bound for the analysis. Set to -1 for automatic selection.
- `wu::WU`: The upper frequency bound for the analysis. Set to -1 for automatic selection.
- `num_frequencies::Int64`: The number of frequencies to be used in the analysis.
- `pos_feedback::Bool`: Whether the feedback is positive in the analysis. Default is true since a negative feedback is commonly already built into the model.
- `duration::Float64`: The duration of the step-response experiment in the analysis. Default is -1 for automatic selection.
- `loop_openings::Vector{String}`: Names of loop openings to break feedback if present. Default is an empty vector. 

# Result visualization
The analysis artifact displays
- Closed-loop transfer functions ``S = 1/(1+PC)``, ``T = PC/(1+PC)``, ``C / (1+PC)``, ``P/(1+PC)``
- Disk margin, a measure of the robustness w.r.t. combined gain and phase variations
- Gain and phase margins, a measure of the robustness w.r.t. individual gain and phase variations
- Step response of the closed-loop system. The step response is simulated using the four closed-loop transfer functions, and have as inputs the signals ``r`` and ``d`` shown in the diagram above, i.e., disturbances entering at ``y`` and ``u``. The four responses indicate
    - ``u -> u``: The control-signal response to a unit step input disturbance
    - ``u -> y``: The plant output response to a unit step input disturbance
    - ``y -> u``: The control-signal response to a unit reference step
    - ``y -> y``: The plant output response to a unit reference step

# Details
In order to isolate plant from controller, the connection from the controller to the plant is always broken in this analysis. This means that it can be hard to analyze the sensitivity of cascaded control systems, where the output of one controller is the input to another. 

When MIMO systems are analyzed, some sensitivity functions are drawn as Sigma plots rather than Bode plots. The diskmargin is in this case computed for the output loop-transfer function ``L = PC``, that is, the margin for simultaneous output-perturbations is analyzed. This is generally a conservative analysis. See [`loop_diskmargin`](@ref) for single-loop margins.
"""
@kwdef struct ClosedLoopAnalysisSpec{M,WL,WU} <:
        AbstractClosedLoopAnalysisSpec
    name::Symbol
    model::M
    measurement::Vector{String}
    control_input::Vector{String}
    wl::WL = -1
    wu::WU = -1
    num_frequencies::Int64 = 300
    pos_feedback::Bool = true # Default to true for MTK models since the negative gain is usually built into the model
    duration::Float64 = -1.0
    loop_openings::Vector{String} = String[] 
end


Base.nameof(spec::ClosedLoopAnalysisSpec) = spec.name

function sysiszero(sys::AbstractStateSpace)
    iszero(sys.D) || return false
    iszero(sys.C) || iszero(sys.B)
end

# TODO: Fix this up with the new fields
function Base.show(io::IO, m::MIME"text/plain", spec::ClosedLoopAnalysisSpec)
    print(io, "PID Autotuning Analysis specification for ")
    printstyled(io, "$(nameof(spec))\n", color = :green, bold = true)
    println(io, "measurement: ", spec.measurement)
    println(io, "control_input: ", spec.control_input)
    println(io, "wl: ", spec.wl)
    println(io, "wu: ", spec.wu)
    println(io, "num_frequencies: ", spec.num_frequencies)
    println(io, "pos_feedback: ", spec.pos_feedback)
    println(io, "duration: ", spec.duration)
end

function setup_prob(spec::ClosedLoopAnalysisSpec)
    # TODO: Add value validation

    # convert argument types
    outputs = Symbol.(spec.measurement)
    inputs = Symbol.(spec.control_input)


    # linearize
    # We break any existing feedback from the plant output with loop_openings = [measurement]
    loop_openings = unique([spec.loop_openings; inputs])
    P = named_ss(spec.model, inputs, outputs; loop_openings, warn_empty_op=false)
    C = named_ss(spec.model, outputs, inputs; loop_openings, warn_empty_op=false)

    if spec.wl < 0 || spec.wu < 0
        w = ControlSystemsBase._default_freq_vector(P, Val{:bode}())
    else
        wl = spec.wl
        wu = spec.wu
        wN = spec.num_frequencies
        w = exp10.(LinRange(log10(wl), log10(wu), wN))
    end
    P = sminreal(P)
    C = sminreal(C)
    if sysiszero(C)
        @info "Linearized controller is zero. Performing analysis with unit feedback. If this is unintended, check the model for saturation or other similar nonlinearities such as use of functions `max, min, clamp` that might yield a zero linearization at the specified operating point."
        # C = nothing 
        C = ss(-I(P.ny))
    end
    if sysiszero(P)
        error("Linearized plant model is zero. This may be caused by a missing connection or the linearization at the specified operating point is zero. Check the model for saturation or other similar nonlinearities such as use of functions `max, min, clamp`.")
    end

    sminreal(P), C
end

struct ClosedLoopAnalysisSolution{SP <: ClosedLoopAnalysisSpec} <: AbstractAnalysisSolution
    spec::SP
    P
    C
end

function Base.show(io::IO, m::MIME"text/plain", sol::ClosedLoopAnalysisSolution)
    spec = sol.spec
    dm = diskmargin((spec.pos_feedback ? -1 : 1)*sol.P*sol.C)
    println(io, "ClosedLoopAnalysisSolution")
    println(io, "Phase margin (disk based): $(few(dm.phasemargin))°")
    println(io, "Gain margin (disk based): $(few.(dm.gainmargin))")
    return nothing
end

function DyadInterface.run_analysis(spec::ClosedLoopAnalysisSpec)
    P, C = setup_prob(spec)
    stripped_spec = @set spec.model = nothing
    ClosedLoopAnalysisSolution(stripped_spec, P, C)
end

# returns a serializable description of the visualizations that can be constructed from the solution object.
function DyadInterface.AnalysisSolutionMetadata(spec::ClosedLoopAnalysisSpec)
    plt_names = [
        # :SensitivityFunctions
        # :Margins
        # :Diskmargin
        # :StepResponse
        :all
    ]
    plt_types = [
        # CannedVisualizationType.PlotlyPlot
        # CannedVisualizationType.PlotlyPlot
        # CannedVisualizationType.PlotlyPlot
        # CannedVisualizationType.PlotlyPlot
        CannedVisualizationType.PlotlyPlot
    ]
    plt_titles = [
        # "Sensitivity functions"
        # "Margins"
        # "Disk margin"
        # "Step response"
        ""
    ]
    plt_descriptions = [
        # "Sensitivity functions"
        # "Margins"
        # "Disk margin"
        # "Step response"
        "Result"
    ]


    cannedresults = DyadInterface.ArtifactMetadata.(
        plt_names,
        plt_types,
        plt_titles,
        plt_descriptions,
    )

    allowed_symbols = []
    AnalysisSolutionMetadata(cannedresults, allowed_symbols)
end

# returns the canned visualization of name `name`. The allowed `canned_visualization`s are defined by the `AnalysisSolutionMetadata` provided by `get_metadata`.
function DyadInterface.artifacts(sol::ClosedLoopAnalysisSolution, name::Symbol)
    (; P, C, spec) = sol
    analyze_robustness(P, C; spec.pos_feedback, Tf=spec.duration > 0 ? spec.duration : nothing)
end

# returns a visualization object. For example, for a `PlotlyVisualizationSpec`, this would return a Plots.jl plot built by the Plotly backend.
# function customizable_visualization(::ClosedLoopAnalysisSolution, ::AbstractVisualizationSpec)
# end

