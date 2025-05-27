#=
Dev docs

Each objective defines the following functions
- `show`
- A plot recipe


=#


using ControlSystems
using Printf
using Statistics
import DifferentiationInterface as DI
using ADTypes

const FILL_COLOR = :red
const FILL_ALPHA = 0.08
const SERIES_ALPHA = 0.7
const TARGET_COLOR = :green

# Trait that indicate whether the objective is to be considered a constraint or a loss
abstract type ObjectiveType end
struct LossObjective <: ObjectiveType end
struct ConstraintObjective <: ObjectiveType end

abstract type TuningObjective end
abstract type AbstractStepObjective <: TuningObjective end
# NOTE: the objective does not contain any information regarding the model, but it must know which signals it operates on


symstr(x) = Symbol(x isa AnalysisPoint ? x.name : string(x))

abstract type WeightedMagnitudeObjective <: TuningObjective end

## Sensitivity objective =======================================================
"""
    MaximumSensitivityObjective(; weight, output, loop_openings)

A tuning objective that upper bounds the sensitivity function at the signal `output`.

The output [sensitivity function](https://en.wikipedia.org/wiki/Sensitivity_(control_systems)) ``S_o = (I + PC)^{-1}`` is the transfer function from a reference input to control error, while the input sensitivity function ``S_i = (I + CP)^{-1}`` is the transfer function from a disturbance at the plant input to the total plant input. For SISO systems, input and output sensitivity functions are equal. In general, we want to minimize the sensitivity function to improve robustness and performance, but pracitcal constraints always cause the sensitivity function to tend to 1 for high frequencies. A robust design minimizes the peak of the sensitivity function, ``M_S``. The peak magnitude of ``S`` is the inverse of the distance between the open-loop Nyquist curve and the critical point -1. Upper bounding the sensitivity peak ``M_S`` gives lower-bounds on phase and gain margins according to
```math
ϕ_m ≥ 2\\text{asin}(\\frac{1}{2M_S}), g_m ≥ \\frac{M_S}{M_S-1}
```
Generally, bounding ``M_S`` is a better objective than looking at gain and phase margins due to the possibility of combined gain and pahse variations, which may lead to poor robustness despite large gain and pahse margins.

# Fields:
- `weight`: An LTI system (statespace or transfer function) whose magnitude forms the upper bound of the sensitivity function.
- `output`: The analysis point in which the sensitivity function is computed. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on analysis points.
- `loop_openings`: A list of analysis points in which the loop will be opened before computing the objective. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on loop openings.

To penalize transfer functions such as the amplification of measurement noise in the control signal, ``CS = C(I + CP)^{-1}``, use a [`MaximumTransferObjective`](@ref).

See also: [MaximumTransferObjective](@ref) [`sensitivity`](@ref), [`comp_sensitivity`](@ref), [`gangoffour`](@ref).
"""
Base.@kwdef struct MaximumSensitivityObjective{W} <: WeightedMagnitudeObjective
    weight::W
    output::Any = nothing
    loop_openings = []
end

objective_type(::MaximumSensitivityObjective) = ConstraintObjective()

MaximumSensitivityObjective(weight, output=nothing) = MaximumSensitivityObjective(weight, output, [])

function Base.show(io::IO, o::MaximumSensitivityObjective)
    if !isempty(o.loop_openings)
        print(io, "Max sens at ", symstr(o.output))
        print(io, " with openings ", symstr.(o.loop_openings))
    else
        print(io, "Max sensitivity at ", symstr(o.output))
    end
end

"""
    MaximumTransferObjective(; weight, input, output, loop_openings)

A tuning objective that upper bounds the transfer function from `input` to `output`.

# Fields:
- `weight`: An LTI system (statespace or transfer function) whose magnitude forms the upper bound of the transfer function.
- `input`: A named analysis point in the model. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on analysis points.
- `output`: A named analysis point in the model.
- `loop_openings`: A list of analysis points in which the loop will be opened before computing the objective. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on loop openings.

# Example:
To bound the transfer function from measurement noise appearing at the plant output, to the control signal ``CS = \\dfrac{C}{I + PC}``, create the following objective:
```julia
mto   = MaximumTransferObjective(weight, :y, :u) # CS
```
`weight` can be a simple upper bound like `tf(1.2)`, or a frequency-dependent transfer function.

See also: [`MaximumSensitivityObjective`](@ref).
"""
Base.@kwdef struct MaximumTransferObjective{W} <: WeightedMagnitudeObjective
    weight::W
    input::Any = nothing
    output::Any = nothing
    loop_openings = []
end

objective_type(::MaximumTransferObjective) = ConstraintObjective()

MaximumTransferObjective(weight, input=nothing, output=nothing) = MaximumTransferObjective(weight, input, output, [])

function Base.show(io::IO, o::MaximumTransferObjective)
    print(io, "Max transfer between ", symstr(o.input), " and ", symstr(o.output))
    if !isempty(o.loop_openings)
        print(io, " with openings ", symstr.(o.loop_openings))
    end
end

## Step-response objectives ====================================================
"""
    OvershootObjective(; max_value, input, output, loop_openings)

A tuning objective that upper bounds the overshoot of the step response from analysis point `input` to `output`. The nonlinear system will be linearized between input and output for each operating point.

# Fields:
- `max_value`: The maximum allowed overshoot for a step of magnitude 1.
- `input`: A named analysis point in the model. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on analysis points.
- `output`: A named analysis point in the model.
- `loop_openings`: A list of analysis points in which the loop will be opened before computing the objective. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on loop openings.
"""
Base.@kwdef struct OvershootObjective <: AbstractStepObjective
    max_value::Float64
    input::Any = nothing
    output::Any = nothing
    loop_openings = []
    function OvershootObjective(max_value, input = nothing, output = nothing, loop_openings = [])
        max_value ≥ 1 || throw(ArgumentError("max_value must be ≥ 1"))
        new(Float64(max_value), input, output, loop_openings)
    end
end

objective_type(::OvershootObjective) = ConstraintObjective()

Base.show(io::IO, o::OvershootObjective) = print(io, "Overshoot $(symstr(o.input)) → $(symstr(o.output)) ≤ ", o.max_value)

## RiseTimeObjective ===========================================================
"""
    RiseTimeObjective(; min_value, time, input, output, loop_openings)

A tuning objective that upper bounds the rise time of the step response from analysis point `input` to `output`. The nonlinear system will be linearized between input and output for each operating point.

# Fields:
- `min_value`: A step of magnitude 1 have to stay above at least this value after `time` has passed.
- `time`: The time after which the step has to stay above `min_value`.
- `input`: A named analysis point in the model. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on analysis points.
- `output`: A named analysis point in the model.
- `loop_openings`: A list of analysis points in which the loop will be opened before computing the objective. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on loop openings.
"""
Base.@kwdef struct RiseTimeObjective <: AbstractStepObjective
    min_value::Float64
    time::Float64
    input::Any = nothing
    output::Any = nothing
    loop_openings = []
end

RiseTimeObjective(min_value, time) =
    RiseTimeObjective(Float64(min_value), Float64(time), nothing, nothing, [])

objective_type(::RiseTimeObjective) = ConstraintObjective()

Base.show(io::IO, o::RiseTimeObjective) = print(io, "Rise time $(symstr(o.input)) → $(symstr(o.output)) ≥ ", o.min_value, " after t=", o.time)

# SettlingTimeObjective ========================================================
"""
    SettlingTimeObjective(; final_value, time, tolerance, input, output, loop_openings)

A tuning objective that upper bounds the settling time of the step response from analysis point `input` to `output`. The nonlinear system will be linearized between input and output for each operating point.

# Fields:
- `final_value`: The desired final value after a step of magnitude 1.
- `time`: The time after which the step has to stay within `tolerance` of `final_value`.
- `tolerance::Float64`: The maximum allowed relative error from `final_value`.
- `input`: A named analysis point in the model. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on analysis points.
- `output`: A named analysis point in the model.
- `loop_openings`: A list of analysis points in which the loop will be opened before computing the objective. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on loop openings.
"""
Base.@kwdef struct SettlingTimeObjective <: AbstractStepObjective
    final_value::Float64
    time::Float64
    tolerance::Float64
    input::Any = nothing
    output::Any = nothing
    loop_openings = []
end

SettlingTimeObjective(final_value, time, tolerance) = SettlingTimeObjective(
    Float64(final_value),
    Float64(time),
    Float64(tolerance),
    nothing,
    nothing,
    nothing,
)

objective_type(::SettlingTimeObjective) = ConstraintObjective()

Base.show(io::IO, o::SettlingTimeObjective) = print(io, "Settling time $(symstr(o.input)) → $(symstr(o.output)) ≤ ", o.time, " tol: ", o.tolerance)

## StepTrackingObjective =======================================================
"""
    StepTrackingObjective(; reference_model, tolerance, input, output, loop_openings)

A tuning objective that upper bounds the tracking error of the step response from analysis point `input` to `output`. The nonlinear system will be linearized between input and output for each operating point.

- `reference_model:`: A model indicating the desired step response
- `tolerance = 0.05`: A tolerance relative the final value of the reference step within which there is no penalty, the default is 5% of the final value.
- `loop_openings`: A list of analysis points in which the loop will be opened before computing the objective. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on loop openings.
"""
Base.@kwdef struct StepTrackingObjective{R} <: AbstractStepObjective
    reference_model::R
    tolerance::Float64 = 0.05
    input::Any = nothing
    output::Any = nothing
    loop_openings = []
end

StepTrackingObjective(reference_model) =
    StepTrackingObjective(reference_model, 0.05, nothing, nothing, [])

objective_type(::StepTrackingObjective) = LossObjective()

Base.show(io::IO, o::StepTrackingObjective) = print(io, "Step tracking $(symstr(o.input)) → $(symstr(o.output)) with tol ", o.tolerance)

## GainMarginObjective =========================================================
"""
    GainMarginObjective(; margin, output, loop_openings)

A tuning objective that lower bounds the gain margin of the system.

# Fields:
- `margin`: The desired margin in absolute scale, i.e. for `margin = 2`, the closed-loop is robust w.r.t. gain variations within a factor of two.
- `output`: The signal to compute the gain margin in.
- `loop_openings`: A list of analysis points in which the loop will be opened before computing the objective. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on loop openings.
"""
Base.@kwdef struct GainMarginObjective <: TuningObjective
    margin::Float64
    output::Any = nothing
    loop_openings = []
end

GainMarginObjective(margin, output=nothing) = GainMarginObjective(Float64(margin), output, [])

objective_type(::GainMarginObjective) = ConstraintObjective()

Base.show(io::IO, o::GainMarginObjective) = print(io, "Gain margin at $(symstr(o.output)) ≥ ", o.margin)

## PhaseMarginObjective ========================================================
"""
    PhaseMarginObjective(; margin, output, loop_openings)

A tuning objective that lower bounds the phase margin of the system.

# Fields:
- `margin`: The desired margin in degrees.
- `output`: The signal to compute the phase margin in.
- `loop_openings`: A list of analysis points in which the loop will be opened before computing the objective. See [MTK stdlib: Linear analysis](http://mtkstdlib.sciml.ai/stable/API/linear_analysis/) for more info on loop openings.
"""
Base.@kwdef struct PhaseMarginObjective <: TuningObjective
    margin::Float64
    output::Any = nothing
    loop_openings = []
end

PhaseMarginObjective(margin, output=nothing) = PhaseMarginObjective(Float64(margin), output, [])

objective_type(::PhaseMarginObjective) = ConstraintObjective()

Base.show(io::IO, o::PhaseMarginObjective) = print(io, "Phase margin at $(symstr(o.output)) ≥ ", o.margin)

## SimulationObjective

"""
    SimulationObjective

A tuning objective that uses a custom cost function to evaluate the performance of a simulation of the full nonlinear system model.

# Fields:
- `costfun`: A function that takes an ODESolution and returns a scalar cost value.
- `prob`: An ODEProblem to be solved
- `solve_args`: A tuple with the arguments to pass to `solve(prob, solve_args...)`. This includes the choice of the solver algorithm, e.g., `solver_args = (Tsit5(), )` is used to form the call `solve(prob, Tsit5())`.
- `solve_kwargs`: A named tuple with the keyword arguments to pass to `solve(prob, solve_args...; solve_kwargs...)`
"""
Base.@kwdef struct SimulationObjective{F,P,TA,TB} <: TuningObjective
    costfun::F
    prob::P
    solve_args::TA = ()
    solve_kwargs::TB = (;)
end

objective_type(::SimulationObjective) = LossObjective()

function Base.show(io::IO, o::SimulationObjective)
    println(io, "Simulation objective with cost function ", o.costfun)
    println(io, " solver args ", o.solve_args)
    println(io, " solver keyword arguments ", o.solve_kwargs)
end

# ==============================================================================
## Cost functions
# ==============================================================================
# σ(x) = 1 / (1 + exp(-x)) + max(zero(x), 0.001x)
σ(x) = 1 / (1 + exp(-x + 2)) #+ max(zero(x), 0.001x)
elu(x) = (x > 0 ? x : exp(x) - 1) + 1
logmag(x) = log(abs(x))

function get_tf(ssys, lin_fun, op)
    matrices = linearize(ssys, lin_fun; op)
    ss(matrices...) |> sminreal
end

# General method that works for many objectives, but not sensitivity objectives
function get_tf(o::TuningObjective, sys, op)
    named_ss(sys, o.input, o.output; op)
end

function get_tf(o::MaximumSensitivityObjective, sys, op)
    matrices, ssys = get_sensitivity(sys, o.output; op, o.loop_openings)
    named_ss(sminreal(ss(matrices...)), x = :Sx, u=:d, y=Symbol.(ModelingToolkit.outputs(ssys)))
end

function get_tf(o::MaximumTransferObjective, sys, op)
    matrices, ssys = linearize(sys, o.input, o.output; op, o.loop_openings)
    named_ss(sminreal(ss(matrices...)), x = :Sx, u=:d, y=Symbol.(ModelingToolkit.outputs(ssys)))
end

function get_tf(o::Union{GainMarginObjective, PhaseMarginObjective}, sys, op)
    matrices, ssys = get_looptransfer(sys, o.output; op, o.loop_openings)
    named_ss(sminreal(ss(matrices...)), x = :Lx, u=Symbol.(ModelingToolkit.inputs(ssys)), y=Symbol.(ModelingToolkit.outputs(ssys)))
end

function get_step!(cache, o, ssys, lin_fun, op, t)
    cacheval = get(cache, (:step, o.input, o.output), nothing)
    if cacheval === nothing
        lsys = get_tf(ssys, lin_fun, op)
        res = step(lsys, t)
        cache[(:step, o.input, o.output)] = res
    else
        res = cacheval
    end
    res
end

function get_costfunction(o::DyadControlSystems.StepTrackingObjective, sys, w, t, example_op)
    y_ref = step(o.reference_model, t).y
    lin_fun, ssys = linearization_function(sys, o.input, o.output; op=example_op)
    tolerance = o.tolerance * abs(y_ref[end])
    function costfun_StepTrackingObjective(cache, op)
        res = get_step!(cache, o, ssys, lin_fun, op, t)
        cost = 0.0
        for ti in eachindex(t)
            error = abs(y_ref[ti] - res.y[ti])
            cost += σ(1 * (error - tolerance))
        end
        cost / length(t)
    end
end

function get_costfunction(o::DyadControlSystems.RiseTimeObjective, sys, w, t, example_op)
    istart = findfirst(>=(o.time), t)
    lin_fun, ssys = ModelingToolkit.linearization_function(sys, o.input, o.output; op=example_op)
    function costfun_RiseTimeObjective(cache, op)
        res = get_step!(cache, o, ssys, lin_fun, op, t)
        cost = sum(istart:length(t)) do ti
            error = (o.min_value - res.y[ti]) * ti/length(t) # higher weight for later times
            max(error, 0.0)
        end
        cost
    end
end

function get_costfunction(o::DyadControlSystems.SettlingTimeObjective, sys, w, t, example_op)
    istart = findfirst(>=(o.time), t)
    lin_fun, ssys = linearization_function(sys, o.input, o.output; op=example_op)
    function costfun_SettlingTimeObjective(cache, op)
        res = get_step!(cache, o, ssys, lin_fun, op, t)
        cost = sum(istart:length(t)) do ti
            error = abs(res.y[ti] - o.final_value) * ti/length(t) # higher weight for later times
            max(error - o.tolerance, 0.0)
        end
        cost
    end
end

function get_costfunction(o::DyadControlSystems.MaximumSensitivityObjective, sys, w, t, example_op)
    m_ref = logmag.(freqresp(o.weight, w))
    lin_fun, ssys = get_sensitivity_function(sys, o.output; o.loop_openings, op=example_op)
    function costfun_MaximumSensitivityObjective(cache, op)
        cacheval = get(cache, (:mag, :nothing, o.output), nothing)
        if cacheval === nothing
            lsys = get_tf(ssys, lin_fun, op)
            m = logmag.(freqresp(lsys, w))
            cache[(:mag, :nothing, o.output)] = m
        else
            m = cacheval
        end
        activation = max.(m .- m_ref, 0.0)
        maximum(activation) #+ mean(activation)
    end
end

function get_costfunction(o::DyadControlSystems.MaximumTransferObjective, sys, w, t, example_op)
    m_ref = logmag.(freqresp(o.weight, w))
    lin_fun, ssys = linearization_function(sys, o.input, o.output; o.loop_openings, op=example_op)
    function costfun_MaximumTransferObjective(cache, op)
        cacheval = get(cache, (:mag, :nothing, o.output), nothing)
        if cacheval === nothing
            lsys = get_tf(ssys, lin_fun, op)
            m = logmag.(freqresp(lsys, w))
            cache[(:mag, :nothing, o.output)] = m
        else
            m = cacheval
        end
        activation = max.(m .- m_ref, 0.0)
        # activation = elu.(100 .* (m .- m_ref))
        maximum(activation) # + mean(activation)
    end
end

function get_costfunction(o::DyadControlSystems.OvershootObjective, sys, w, t, example_op)
    lin_fun, ssys = linearization_function(sys, o.input, o.output; op=example_op)
    function costfun_OvershootObjective(cache, op)
        res = get_step!(cache, o, ssys, lin_fun, op, t)
        sum(1:length(t)) do ti
            max(res.y[ti] - o.max_value, 0.0)
        end
    end
end

function get_costfunction(o::Union{GainMarginObjective, PhaseMarginObjective}, sys, w, t, example_op)
    lin_fun, ssys = get_looptransfer_function(sys, o.output; op=example_op)
    function costfun_GainPhaseMarginObjective(cache, op)
        cacheval = get(cache, (:tf, :nothing, o.output), nothing)
        if cacheval === nothing
            lsys = get_tf(ssys, lin_fun, op)
            cache[(:tf, :nothing, o.output)] = lsys
        else
            lsys = cacheval
        end

        cacheval2 = get(cache, (:dm, :nothing, o.output), nothing)
        if cacheval2 === nothing
            dm = diskmargin(-lsys) # - since loop has built in negative feedback
            cache[(:dm, :nothing, o.output)] = dm
        else
            dm = cacheval2
        end
        margin = o isa GainMarginObjective ? dm.gainmargin[2] : rad2deg(dm.ϕm)
        isfinite(margin) || return zero(margin) # Gain margin can be Inf
        sum(max.(o.margin .- margin, 0.0))
    end
end


function get_costfunction(o::SimulationObjective, sys, w, t, example_op)
    function costfun_SimulationObjective(cache, op)
        cacheval = get(cache, (:sim, :nothing, op), nothing)
        if cacheval === nothing
            p = filter(kv->ModelingToolkit.isparameter(kv[1]), op)
            u0 = filter(kv->!ModelingToolkit.isparameter(kv[1]), op)
            prob = remake(o.prob; p, u0)
            sol = solve(prob, o.solve_args...; o.solve_kwargs...)
        else
            sol = cacheval
        end
        o.costfun(sol)
    end
end


# ==============================================================================
## StructuredAutoTuningProblem
# ==============================================================================

"""
    StructuredAutoTuningProblem(sys, w, t, objectives, operating_points, tunable_parameters)
    StructuredAutoTuningProblem(; sys, w, t, objectives, operating_points, tunable_parameters)

An autotuning problem structure for parameter tuning in ModelingToolkit models.

# Fields:
- `sys`: An ODESystem to tune controller parameters in
- `w`: A vector of frequencies to evaluate objectives on
- `t`: A vector of time points to evaluate objectives on
- `objectives`: A vector of tuning objectives
- `operating_points`: A vector of operating points in which to linearize and optimize the system. An operating point is a dict mapping symbolic variables to numerical values.
- `tunable_parameters`: A vector of pairs of the form `(parameter, range)` where `parameter` is a
  parameter in `sys` and `range` is a tuple of values that lower and upper bound the feasible parameter space.
"""
Base.@kwdef struct StructuredAutoTuningProblem
    sys::Any
    w::Any
    t::Any
    objectives::Any
    operating_points::AbstractVector{<:Union{<:Dict, AbstractVector{<:Pair}}}
    tunable_parameters::Any
end

function Base.show(io::IO, p::StructuredAutoTuningProblem)
    print(io, "StructuredAutoTuningProblem\n")
    @printf(io, "%-20s: %-20s\n", "sys", p.sys.name)
    @printf(io, "%-20s: %d between %-4.2g and %4.2g\n", "num frequencies", length(p.w), extrema(p.w)...)
    @printf(io, "%-20s: %d between %-4.2g and %4.2g\n", "num time points", length(p.t), extrema(p.t)...)
    @printf(io, "%-20s: %d\n", "num operating points", length(p.operating_points))
    @printf(io, "%-20s:\n", "objectives")
    foreach(x->println(io, "\t"^3, x), p.objectives)
    @printf(io, "%-20s:\n", "tunable parameters")
    foreach(x->println(io, "\t"^3, x), p.tunable_parameters)
end

"""
    StructuredAutoTuningResult

# Fields:
- `prob::StructuredAutoTuningProblem`: The problem that was solved to create the result.
- `optprob::OptimizationProblem`: The optimization problem that was solved underneath the hood.
- `sol`: The solution to the inner optimization problem.
- `op`: The optimal operating points, a vector of the same length as `prob.operating_points`.
- `cost_functions`: A vector of internal cost functions, one for each objective.
- `objective_status`: A vector of dicts, one for each operating point, that indicates how well each objective was met.
"""
struct StructuredAutoTuningResult
    prob::StructuredAutoTuningProblem
    optprob::OptimizationProblem
    sol::Any
    op::Any
    cost_functions::Any
    objective_status::Any
end

function Base.show(io::IO, r::StructuredAutoTuningResult)
    print(io, "StructuredAutoTuningResult\n")
    @printf(io, "%-20s: %-20s\n", "sys", r.prob.sys.name)
    @printf(io, "%-20s: %d between %-4.2g and %4.2g\n", "num frequencies", length(r.prob.w), extrema(r.prob.w)...)
    @printf(io, "%-20s: %d between %-4.2g and %4.2g\n", "num time points", length(r.prob.t), extrema(r.prob.t)...)
    @printf(io, "%-20s: %d\n", "num operating points", length(r.prob.operating_points))
    @printf(io, "%-20s\n", "objectives")
    foreach(x->println(io, "\t", x), r.prob.objectives)
    @printf(io, "%-20s\n", "tunable parameters")
    foreach(x->println(io, "\t", x), r.prob.tunable_parameters)
    @printf(io, "%-20s: %s\n", "optimization status", r.sol.retcode)
    if length(r.objective_status) == 1
        @printf(io, "%-20s\n", "objective status")
        foreach(x->println(io, "\t", x), r.objective_status[])
    end
    @printf(io, "%-20s\n", "minimizer")
    foreach(((pp,x),)->println(io, "\t", first(pp)=>x), zip(r.prob.tunable_parameters, r.sol.u))
    @printf(io, "%-20s: %s\n", "objective value", r.sol.minimum)
end

function Base.getproperty(res::StructuredAutoTuningResult, s::Symbol)
    s ∈ fieldnames(typeof(res)) && return getfield(res, s)
    getproperty(getfield(res, :prob), s)
end


geometric_mean((x, y),) = sqrt(x * y)
geometric_mean(pair::Pair) = geometric_mean(last(pair))

function evaluate_cost(prob::StructuredAutoTuningProblem, x, cost_functions, p; weights = nothing)
    value = zero(eltype(x))
    objectives = prob.objectives
    for (j, op) in enumerate(prob.operating_points)
        # Update the parameters
        op = new_op(op, p, x)
        # @show op[complete(prob.sys).inertia.J]
        ap = Union{Symbol, AnalysisPoint}
        cache = Dict{Tuple{ap,ap,Any},Any}()

        for (i, (fun, obj)) in enumerate(zip(cost_functions, objectives))
            if objective_type(obj) isa ConstraintObjective
                continue
            end
            vᵢ = fun(cache, op)
            if weights !== nothing
                vᵢ *= weights[j][i]
            end
            value += vᵢ
        end
    end
    value / length(prob.operating_points)
end

function evaluate_cons(prob::StructuredAutoTuningProblem, out, x, cost_functions, p; weights = nothing, slack_inds)
    objectives = prob.objectives
    constraint_number = 1
    for (j, op) in enumerate(prob.operating_points)
        # Update the parameters
        op = new_op(op, p, x)
        ap = Union{Symbol, AnalysisPoint}
        cache = Dict{Tuple{ap,ap,Any},Any}()

        for (i, (fun, obj)) in enumerate(zip(cost_functions, objectives))
            if !(objective_type(obj) isa ConstraintObjective)
                continue
            end
            vᵢ = fun(cache, op)
            if weights !== nothing
                vᵢ *= weights[j][i]
            end
            out[constraint_number] = vᵢ - x[slack_inds[constraint_number]]
            constraint_number += 1
        end
    end
    @assert constraint_number == length(out) + 1
    nothing
end


"""
    solve(
        prob::StructuredAutoTuningProblem,
        x0,
        alg = MPC.IpoptSolver(
            verbose         = true,
            exact_hessian   = false,
            acceptable_iter = 4,
            tol             = 1e-3,
            acceptable_tol  = 1e-2,
            max_iter        = 100,
        );
        verbose = true,
        kwargs...,
    )

Solve a structured parametric autotuning problem. The result can be plotted with `plot(result)`.

# Arguments:
- `prob`: An instance of StructuredAutoTuningProblem holding tunable parameters and tuning objectives.
- `x0`: A vector of initial guesses for the tunable parameters, in the same order as `prob.tunable_parameters`.
- `alg`: An optimization algorithm compatible with Optimization.jl.
- `verbose`: Print diagnostic information?
- `kwargs`: Are passed to `solve` on the optimization problem from Optimization.jl.
"""
function CommonSolve.solve(
    prob::StructuredAutoTuningProblem,
    x0,
    alg = MPC.IpoptSolver(
        verbose = true,
        exact_hessian = false,
        acceptable_iter = 4,
        tol = 1e-3,
        acceptable_tol = 1e-2,
        max_iter = 100,
    );
    verbose = true,
    probkwargs = (),
    outer_iters = 1, # Undocumented and experimental, set to 2 for an attempt at rebalancing the soft constraints in case there is large violation of some constraint.
    weights = nothing, # Undocumented and experimental, can be set to a vector of vectors of where the inner vector length matches the number of objectives and the outer length matches the number of operating points
    s1 = 100.0,
    s2 = 100.0,
    kwargs...,
)

    x0 = float.(x0) # To guard against integers
    @set! prob.operating_points = reduce(vcat, expand_uncertain_operating_points.(prob.operating_points))
    (; sys, w, t, objectives, tunable_parameters, operating_points) = prob

    # @show typeof(operating_points)

    p = first.(tunable_parameters)
    bounds = last.(tunable_parameters)
    # syss = structural_simplify(sys)
    # simprob = ODEProblem(syss, operating_points[1], (t[1], t[end]))
    example_op = deepcopy(prob.operating_points[1])
    for (p, r) in zip(p, bounds)
        example_op[p] = r[1] # Add tuning parameters to example op
    end
    cost_functions = [get_costfunction(o, sys, w, t, example_op) for o in objectives]
    num_op = length(operating_points)
    num_cons = count(obj -> objective_type(obj) isa ConstraintObjective, objectives) * num_op
    slack_inds = length(x0) .+ (1:num_cons)

    # sabs2(x) = sign(x)*x
    function costfun(x, p_throwaway)
        c1 = evaluate_cost(prob, x, cost_functions, p; weights)
        @views s = x[slack_inds] # slack variables
        c1 + s1*sum(abs, s) + s2*sum(abs2, s)
    end

    function cons(out, x, p_throwaway)
        evaluate_cons(prob, out, x, cost_functions, p; weights, slack_inds)
    end
    cons_jac_prototype = sparse([trues(num_cons, length(x0)) I(num_cons)]) # [vars slacks]
    autodiff = Optimization.AutoSparse(AutoFiniteDiff(); sparsity_detector=ADTypes.KnownJacobianSparsityDetector(cons_jac_prototype))
    # jacobian!(f!, y, jac, [prep,] backend, x, [contexts...])
    # y = zeros(num_cons)
    # function cons_j(jac, x, p)
    #     DI.jacobian!(
    #         (o,x)->cons(o, x, p), y, jac, autodiff, x)
    # end

    lb = [float.(first.(bounds)); fill(0.0, num_cons)] # Ints can cause obscure error messages
    ub = [float.(last.(bounds)); fill(Inf, num_cons)] # Add slack variable bounds
    ucons = fill(0.0, num_cons) # Constraints for which slack variables are added
    lcons = fill(-Inf, num_cons)
    slacks = fill(0.0, num_cons)
    x0s = [x0; slacks]
    # g(x) - s <= 0
    # s >= 0

    optfun = OptimizationFunction(costfun, autodiff; cons, cons_jac_prototype, hess=false, lag_h=false)
    optprob = OptimizationProblem(optfun, x0s, nothing; lb, ub, lcons, ucons, probkwargs...)
    local sol, opopts, objective_status
    for iter in 1:outer_iters
        sol = CommonSolve.solve(optprob, alg; kwargs...)
        opopts = [new_op(op, p, sol.u) for op in operating_points]
        objective_status = map(opopts) do op
            Base.nameof.(typeof.(objectives)) .=>
                    [cf(Dict(), op) for cf in cost_functions]
        end

        vals = [[sqrt(last(s)) for s in os] for os in objective_status]
        vvals = reduce(vcat, vals)
        if any(>(0.5), vvals)
            weights = vals
        else
            break
        end
    end

    if verbose
        @info "Result"
        for objective_status in objective_status
            for (o, v) in objective_status
                @printf("%28s: %4.4g\n", o, v)
            end
        end
    end
    StructuredAutoTuningResult(prob, optprob, sol, opopts, cost_functions, objective_status)
end

function optimized_probs(prob::StructuredAutoTuningResult)
    map(res.op) do op
        ODEProblem(structural_simplify(prob.sys), op, (prob.t[1], prob.t[end]))
    end
end

"""
    new_op(op, p, x)

Create a copy of `op` where `op[p]` is filled with `x`
"""
function new_op(op, p, x)
    op = copy(op)
    for i in eachindex(p)
        op[p[i]] = x[i]
    end
    op
end

# ==============================================================================
## Plot recipes ================================================================
# ==============================================================================

@recipe function plot(os::AbstractVector{<:TuningObjective})
    for o in os
        @series begin
            o
        end
    end
end

@recipe function plot(o::WeightedMagnitudeObjective, w = ControlSystemsBase._default_freq_vector(o.weight, Val(:bode)))
    W = o.weight
    ny, nu = size(W)
    s2i(i, j) = LinearIndices((nu, ny))[j, i]
    bm, _ = bode(W, w)
    for j = 1:nu, i = 1:ny
        @series begin
            xscale --> :log10
            seriestype --> :bodemag
            link := :none
            subplot --> min(s2i(i, j), prod(get(plotattributes, :layout, (nu, ny))))
            hover := string(nameof(typeof(o)))
            primary --> false
            linestyle --> :dash
            color --> FILL_COLOR
            fillalpha --> FILL_ALPHA
            seriesalpha --> SERIES_ALPHA
            fillrange --> 10 * bm[i, j, :] # NOTE: Should be Inf, this is a workaround for https://github.com/JuliaPlots/Plots.jl/issues/4075
            w, bm[i, j, :]
        end
    end
end

@recipe function plot(o::WeightedMagnitudeObjective, sys, op, t, w, extras)
    G = get_tf(o, sys, op)
    m, _ = bodev(G, w; unwrap=false)
    title --> string(o)
    @series begin
        primary := true
        label --> ""
        seriestype --> :bodemag
        link := :none
        legend --> false
        w, m
    end
    if extras
        @series begin
            o, w
        end
    end
end

@recipe function plot(o::AbstractStepObjective, tf = nothing)
    primary --> true
    linestyle --> :dash
    color --> TARGET_COLOR
    label --> string(nameof(typeof(o)))
    seriesalpha --> SERIES_ALPHA
    res = tf === nothing ? step(o.reference_model) : step(o.reference_model, tf)
    res
end

@recipe function plot(o::AbstractStepObjective, sys, op, t, w, extras)
    title --> string(o)
    G = get_tf(o, sys, op)
    res = step(G, t)
    legend --> :bottomright
    @series begin
        label --> (extras ? "Linearized sim" : "")
        res
    end
    # @series begin
    #     seriesalpha --> 0.5
    #     sol
    # end
    if extras
        @series begin
            o, t[end]
        end
    end
end

@recipe function plot(o::OvershootObjective, tf = nothing)
    @series begin
        seriestype := :hline
        primary --> false
        linestyle --> :dash
        color --> FILL_COLOR
        fillalpha --> FILL_ALPHA
        seriesalpha --> SERIES_ALPHA
        fillrange --> 2o.max_value 
        [o.max_value]
    end
end

@recipe function plot(o::SettlingTimeObjective, tf = 10 * o.time)
    primary --> false
    linestyle --> :dash
    color --> FILL_COLOR
    fillalpha --> FILL_ALPHA
    seriesalpha --> SERIES_ALPHA
    @series begin
        hover := "Settling time $(o.time), limit $(o.final_value-o.tolerance)"
        fillrange --> 0
        [o.time, tf], [o.final_value - o.tolerance, o.final_value - o.tolerance]
    end
    @series begin
        hover := "Settling time $(o.time), limit $(o.final_value+o.tolerance)"
        fillrange --> fill(2o.final_value, 2)
        [o.time, tf], [o.final_value + o.tolerance, o.final_value + o.tolerance]
    end
end

@recipe function plot(o::RiseTimeObjective, tf = 10 * o.time)
    @series begin
        # seriestype := :hline
        hover := "Rise time: $(o.min_value) after $(o.time)s"
        primary --> false
        linestyle --> :dash
        color --> FILL_COLOR
        fillalpha --> FILL_ALPHA
        seriesalpha --> SERIES_ALPHA
        fillrange --> 0
        [o.time, tf], [o.min_value, o.min_value]
    end
end

# 
@recipe function plot(o::Union{GainMarginObjective, PhaseMarginObjective}, sys, op, t, w, extras)
    title --> string(o)
    lsys = get_tf(o, sys, op)
    dm = diskmargin(lsys.sys, 0, w)
    sp = o isa GainMarginObjective ? 1 : 2
    @series begin
        if o isa GainMarginObjective
            phase --> false
            lower --> false
        else
            gain --> false
        end
        subplot --> sp
        dm
    end
    @series begin
        subplot --> sp
        o
    end
end

@recipe function plot(o::Union{GainMarginObjective, PhaseMarginObjective})
    if o isa GainMarginObjective
        phase --> false
        lower --> false
        subplot --> 1
    else
        gain --> false
        subplot --> 2
    end
    @series begin
        title --> string(o)
        seriestype := :hline
        link := :none
        primary --> false
        linestyle --> :dash
        color --> FILL_COLOR
        fillalpha --> FILL_ALPHA
        seriesalpha --> SERIES_ALPHA
        fillrange --> 1 / o.margin # We fill down to the inverse margin since this corresponds to the lower gain margin for `σ=0` as argument to diskmargin.
        [o.margin]
    end
end

@recipe function plot(o::SimulationObjective, sys, op, t, w, extras)
    p = filter(kv->ModelingToolkit.isparameter(kv[1]), op)
    u0 = filter(kv->!ModelingToolkit.isparameter(kv[1]), op)
    prob = remake(o.prob; u0, p)
    sol = solve(prob, o.solve_args...; o.solve_kwargs...)
    @series begin
        sol
    end
end

@recipe function plot(prob::StructuredAutoTuningProblem, opopts = prob.operating_points)
    @unpack sys, objectives, t, w = prob
    layout --> length(objectives)
    legend --> :bottomright
    titlefontsize --> 8
    opopts = reduce(vcat, expand_uncertain_operating_points.(opopts))
    for (j, op) in enumerate(opopts)
        for (i, o) in enumerate(prob.objectives)
            @series begin
                subplot --> i
                o, sys, op, t, w, (j == 1)
            end
        end
    end
end

@recipe function plot(res::StructuredAutoTuningResult, opopts = res.operating_points)
    @series begin
        res.prob, res.op
    end
end

# ==============================================================================
## Uncertain parameters ========================================================
# ==============================================================================
const MCM = RobustAndOptimalControl.MonteCarloMeasurements
expand_uncertain_operating_points(op::Dict) = MCM.particle_dict2dict_vec(op)
