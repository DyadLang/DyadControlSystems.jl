using JSONSchema
using RelocatableFolders
using Serialization
using Plots
using DataFrames

const PIDAutotuningAnalysisSchema = Schema(read(
    open(@path(joinpath(@__DIR__, "..", "..", "assets", "PIDAutotuningAnalysis.json")), "r"), String))

abstract type AbstractPIDAutotuningAnalysisSpec <: AbstractAnalysisSpec end

export AbstractPIDAutotuningAnalysisSpec, PIDAutotuningAnalysisSpec

@kwdef struct PIDAutotuningAnalysisSpec{M,WL,WU} <:
        AbstractPIDAutotuningAnalysisSpec
    name::Symbol
    model::M
    measurement::String
    control_input::String
    step_input::String
    step_output::String
    ref::Float64 = 0.0
    Ts::Float64
    duration::Float64
    Ms::Float64
    Mt::Float64
    Mks::Float64
    disc::String = "tustin"
    kp_lb::Float64 = 0.0
    ki_lb::Float64 = 0.0
    kd_lb::Float64 = 0.0
    Tf_lb::Float64 = 1e-16
    kp_ub::Float64 = Inf
    ki_ub::Float64 = Inf
    kd_ub::Float64 = Inf
    Tf_ub::Float64 = Inf
    kp_guess::Float64 = -1.0
    ki_guess::Float64 = -1.0
    kd_guess::Float64 = -1.0
    Tf_guess::Float64 = -1.0
    timeweight::Bool = false
    filter_order::Int = 2
    optimize_d::Bool = false
    wl::WL
    wu::WU
    num_frequencies::Int64
end

"json_inf(x) = x >= 1e300 ? Inf : x"
json_inf(x) = x >= 1e300 ? Inf : x

Base.nameof(spec::PIDAutotuningAnalysisSpec) = spec.name

# TODO: Fix this up with the new fields
function Base.show(io::IO, m::MIME"text/plain", spec::PIDAutotuningAnalysisSpec)
    print(io, "PID Autotuning Analysis specification for ")
    printstyled(io, "$(nameof(spec))\n", color = :green, bold = true)
    println(io, "measurement: ", spec.measurement)
    println(io, "control_input: ", spec.control_input)
    println(io, "step_input: ", spec.step_input)
    println(io, "step_output: ", spec.step_output)
    println(io, "ref: ", spec.ref)
    println(io, "Ts: ", spec.Ts)
    println(io, "duration: ", spec.duration)
    println(io, "Ms: ", spec.Ms)
    println(io, "Mt: ", spec.Mt)
    println(io, "Mks: ", spec.Mks)
    println(io, "disc: ", spec.disc)
    println(io, "lb: ", [spec.kp_lb, spec.ki_lb, spec.kd_lb, spec.Tf_lb])
    println(io, "ub: ", [spec.kp_ub, spec.ki_ub, spec.kd_ub, spec.Tf_ub])
    println(io, "guess: ", [spec.kp_guess, spec.ki_guess, spec.kd_guess, spec.Tf_guess])
    println(io, "timeweight: ", spec.timeweight)
    println(io, "filter_order: ", spec.filter_order)
    println(io, "optimize_d: ", spec.optimize_d)
    println(io, "w: ", [spec.wl, spec.wu])
    println(io, "num_frequencies: ", num_frequencies)
end

function setup_prob(spec)
    # TODO: Add value validation

    # convert argument types
    measurement = Symbol(spec.measurement)
    control_input = Symbol(spec.control_input)
    step_input = Symbol(spec.step_input)
    step_output = Symbol(spec.step_output)

    ref = spec.ref
    Ts = spec.Ts
    Tf = spec.duration
    Ms = json_inf(spec.Ms)
    Mt = json_inf(spec.Mt)
    Mks = json_inf(spec.Mks)
    disc = Symbol(spec.disc)
    lb = [spec.kp_lb, spec.ki_lb, spec.kd_lb, spec.Tf_lb]
    ub = [json_inf(spec.kp_ub), json_inf(spec.ki_ub), json_inf(spec.kd_ub), json_inf(spec.Tf_ub)]
    timeweight = spec.timeweight
    filter_order = spec.filter_order
    optimize_d = spec.optimize_d

    # linearize
    inputs = [
        control_input
        step_input
    ] |> unique
    outputs = [
        measurement
        step_output
    ] |> unique

    # We break any existing feedback from the plant output with loop_openings = [measurement]
    P = named_ss(spec.model, inputs, outputs, loop_openings = [measurement])

    if spec.wl < 0 || spec.wu < 0
        w = ControlSystemsBase._default_freq_vector(P, Val{:bode}())
    else
        wl = spec.wl
        wu = spec.wu
        wN = spec.num_frequencies
        w = exp10.(LinRange(log10(wl), log10(wu), wN))
    end

    prob = AutoTuningProblem2(P; w, measurement, control_input, step_input, step_output, ref, Ts, Tf, Ms, Mt, Mks, disc, lb, ub, timeweight, filter_order, optimize_d)
    return prob
end

struct PIDAutotuningAnalysisSolution{SP <: PIDAutotuningAnalysisSpec, S} <: AbstractAnalysisSolution
    spec::SP
    sol::S
end

function DyadInterface.run_analysis(spec::PIDAutotuningAnalysisSpec)
    prob = setup_prob(spec)

    p0 = [spec.kp_guess, spec.ki_guess, spec.kd_guess, spec.Tf_guess]
    if all(<(0), p0)
        p0 = initial_guess(prob)
        @debug "Initial guess:" p0
    end
    sol = solve(prob, p0, solver=IpoptSolver(verbose = true, tol=1e-6, acceptable_tol=1e-6, acceptable_constr_viol_tol=0.1, constr_viol_tol=0.01, printerval=5))

    stripped_spec = @set spec.model = nothing
    PIDAutotuningAnalysisSolution(stripped_spec, sol)
end

# returns a serializable description of the visualizations that can be constructed from the solution object.
function DyadInterface.AnalysisSolutionMetadata(sol::PIDAutotuningAnalysisSolution)
    plt_names = [
        :SensitivityFunctions
        :NoiseSensitivityAndController
        :OptimizedResponse
        :NyquistPlot
    ]
    plt_types = [
        ArtifactType.PlotlyPlot
        ArtifactType.PlotlyPlot
        ArtifactType.PlotlyPlot
        ArtifactType.PlotlyPlot
    ]
    plt_titles = [
        "Sensitivity functions"
        "Noise sensitivity and controller (KS / K)"
        "Optimized response"
        "Nyquist plot"
    ]
    plt_descriptions = [
        "Sensitivity functions"
        "Noise sensitivity and controller (KS / K)"
        "Optimized response"
        "Nyquist plot of loop-transfer function"
    ]

    tbl_names = [
        :OptimizedParameters
    ]
    tbl_types = [
        ArtifactType.DataFrame
    ]
    tbl_titles = [
        "Optimized parameters"
    ]
    # TODO: Migrate the link to `stable` once it is available there
    tbl_descriptions = [
        "Optimized parameters for a PID controller on parallel form. See [the documentation](https://help.juliahub.com/DyadControlSystems/dev/autotuning2/) for details."
    ]

    cannedresults = [
        DyadInterface.ArtifactMetadata(name, type, title, description)
        for (name, type, title, description) in zip(
            vcat(plt_names, tbl_names),
            vcat(plt_types, tbl_types),
            vcat(plt_titles, tbl_titles),
            vcat(plt_descriptions, tbl_descriptions)
        )
    ]

    AnalysisSolutionMetadata(cannedresults, [])
end

# TODO: Parameters for canned artifacts, e.g. to configure plots
# returns the canned artifact of name `name`. The allowed `artifact`s are
# defined by the `AnalysisSolutionMetadata` provided by `get_metadata`.
function DyadInterface.artifacts(sol::PIDAutotuningAnalysisSolution, name::Symbol)
    res = sol.sol
    prob = res.prob
    (; Ts, Tf, disc, Ms, Mt, w, P) = prob
    C = res.K
    tv = 0:Ts:Tf

    Pfb = P[prob.measurement, prob.control_input]

    function sim()
        Gd = c2d(res.G[prob.step_output, prob.step_input], Ts, disc)   # Discretize the system
        prob.response_type(Gd.sys, tv) # Simulate the step response
    end

    S, PS, CS, T = ControlSystemsBase.gangoffour(Pfb.sys, C.sys)

    if name === :SensitivityFunctions
        fig = bodeplot([S, T], w, label = ["S" "T"], plotphase=false)
        hline!([prob.Ms], linestyle=:dash, linecolor=1, label="Ms")
        hline!([prob.Mt], linestyle=:dash, linecolor=2, label="Mt")
    elseif name === :NoiseSensitivityAndController
        fig = bodeplot([CS, C], w, label = ["KS" "K"], plotphase=false)
        hline!([prob.Mks], linestyle=:dash, linecolor=1, label="Mks")
    elseif name === :OptimizedResponse
        fig = plot()
        for i in eachindex(vcat(prob.step_input)), j in eachindex(vcat(prob.step_output))
            step1 = sim() |> deepcopy # Deepcopy due to dstep using cached workspaces
            plot!(step1.t, step1.y[j,:,i], label="$(vcat(prob.step_input)[i]) → $(vcat(prob.step_output)[j])")
        end
    elseif name === :NyquistPlot
        fig = nyquistplot(Pfb*C, w, label="K")
        cs = -1             # Ms center
        rs = 1 / Ms         # Ms radius
        ct = -Mt^2/(Mt^2-1) # Mt center
        rt = Mt/(Mt^2-1)    # Mt radius
        θ = range(0, stop=2π, length=100)
        Sin, Cos = sin.(θ), cos.(θ)
        re, im = nyquist(Pfb, w)
        plot!(fig, vec(re), vec(im), label="P")
        plot!(fig, cs.+rs.*Cos, rs.*Sin, linestyle=:dash, linecolor=1, label="S = $Ms")
        plot!(fig, ct.+rt.*Cos, rt.*Sin, linestyle=:dash, linecolor=2, label="T = $Mt")
        scatter!(fig, [-1], [0], markershape=:xcross, seriescolor=:red, markersize=5, seriesstyle=:scatter, xguide="Re", yguide="Im", framestyle=:zerolines, title="Nyquist plot", xlims=(-3, 1), ylims=(-3, 1), label="")
    elseif name == :OptimizedParameters
        parameters = [
            :kp => res.p[1]
            :ki => res.p[2]
            :kd => res.p[3]
            :Tf => res.p[4]
        ]

        if sol.spec.optimize_d
            push!(parameters, :d => res.p[5])
        end

        return DataFrame(parameters...)
    else
        error("Unknown canned visualization: $name")
    end
    fig
end

# TODO: Remove once JuliaComputing/DyadInterface.jl#65 is resolved
function DyadInterface.customizable_visualization(
    ::PIDAutotuningAnalysisSolution, ::PlotlyVisualizationSpec
)
    missing
end
