using Bumper
function integrate_inputs!(o, u, u0, input_integrators)
    o === u && throw(ArgumentError("o and u must not be aliased"))
    o .= u
    for i in input_integrators
        o[i, 1] += u0[i]
    end
    for t in 1:size(u, 2)-1
        for i in input_integrators
            o[i, t+1] += o[i, t]
        end
    end
end

function integrate_inputs!(u, u0, input_integrators)
    u[:, 1] .+= u0
    for t in 1:size(u, 2)-1
        for i in input_integrators
            u[i, t+1] += u[i, t]
        end
    end
end

## Util functions ==============================================================
"""
    sum_diff(f, x1, x2, w = 1)

Compute `sum(f.(w .* (x1 .- x2)))` without allocating a temporary array. `w` may be a scalar or a vector.
"""
function sum_diff(f, x1, x2, w::Number = 1)
    r = zero(promote_type(eltype(x1), eltype(x2)))
    for (a, b) in zip(x1, x2)
        r += f(w*(a-b))
    end
    r
end

function sum_diff(f, x1, x2, w::AbstractVector)
    r = zero(promote_type(eltype(x1), eltype(x2)))
    for (a, b, wi) in zip(x1, x2, w)
        r += f(wi*(a-b))
    end
    r
end

## Costs =======================================================================
abstract type AbstractMPCCost end
@inline (cost::AbstractMPCCost)(args...) = evaluate(cost, args...)

abstract type AbstractStageCost <: AbstractMPCCost end
abstract type AbstractTrajectoryCost <: AbstractMPCCost end

"""
    Objective(costs...)
An `Objective` holds multiple cost objects.
"""
struct Objective{C <: Tuple} <: AbstractMPCCost
    costs::C
end

"""
    Objective(costs...)

A structure that represents the objective for [`GenericMPCProblem`](@ref). The input can be an arbitrary number of cost objects. See
- [`StageCost`](@ref)
- [`TerminalCost`](@ref)
- [`DifferenceCost`](@ref)
"""
Objective(costs...) = Objective((costs..., ))

"""
    LossFunction{O <: Objective, OI <: ObjectiveInput, V <: Variables, P, T, S}

This structure is used internally in the GenericMPCProblem to handle evaluation of the loss function. It is mutable so that `t` can be updated.
"""
mutable struct LossFunction{O<:Objective,OI<:ObjectiveInput,V<:Variables,P,T,SX,SU} <: Function
    objective::O
    objective_input::OI
    vars::V
    p::P
    t::T
    scale_x::SX
    scale_u::SU
end

function (lf::LossFunction{O,OI,V})(v::AbstractArray{T}, p = lf.p) where {O,OI,V,T}
    @unpack vars, scale_x, scale_u, objective, objective_input, t = lf
    @no_escape begin
        x_cache = isbits(T) ? @alloc(eltype(v), size(vars.x_cache)...) : vars.x_cache
        u_cache = isbits(T) ? @alloc(eltype(v), size(vars.u_cache)...) : vars.u_cache
        u_cache_int = isbits(T) ? @alloc(eltype(v), size(vars.u_cache_int)...) : vars.u_cache_int
        rvars = remake(vars; vars=v, x_cache, u_cache, u_cache_int)
        unscale!(rvars, scale_x, scale_u)
        val = sum(1:vars.n_robust) do ri
            x, u = get_xu(rvars, objective_input.u0, ri)
            oi = remake(objective_input; x, u)
            pind = get_parameter_index(p, ri)
            pp = get_mpc_parameters(pind)
            objective(oi, pp, t)
        end
        scale!(rvars, scale_x, scale_u)
    end
    val
end

"""
    StageCost

A cost-function object that represents the cost at each time-step in a [`GenericMPCProblem`](@ref).

Example:
```
p = (; Q1, Q2, Q3, QN)

running_cost = StageCost() do si, p, t
    Q1, Q2 = p.Q1, p.Q2
    e = si.x .- si.r
    u = si.u
    dot(e, Q1, e) + dot(u, Q2, u)
end
```
"""
struct StageCost{F} <: AbstractStageCost
    fun::F
end

struct TrajectoryCost{F} <: AbstractTrajectoryCost
    fun::F
end

"""
    TerminalCost{F}

A cost-function object that represents the terminal cost in a [`GenericMPCProblem`](@ref).

# Example:

```julia
p = (; Q1, Q2, Q3, QN)

terminal_cost = TerminalCost() do ti, p, t
    e = ti.x .- ti.r
    dot(e, p.QN, e) 
end
```
"""
struct TerminalCost{F} <: AbstractMPCCost
    fun::F
end

"""
    DifferenceCost(metric, getter)
    DifferenceCost(metric)

A cost-function object that represents a running cost of differences in a [`GenericMPCProblem`](@ref).
`metric: (Δz, p, t)->scalar` is a function that computes the cost of a difference ``Δz = z(k) - z(k-1)``, and `getter` is a function `(si, p, t)->z` that outputs ``z``. If `getter` is not provided, it defaults to output the control signal `(si, p, t)->si.u`.

# Example:
The example below illustrates how to penalize ``\\Delta u = u(k) - u(k-1)`` for a single-input system 
```julia
p = (; Q3)

difference_cost = DifferenceCost() do Δu, p, t
    dot(Δu, p.Q3, Δu) # Compute the penalty given a difference `Δu`
end
```

We may also penalize the difference of an arbitrary function of the state and inputs by passing an additional function to `DefferenceCost`. The example above is equivalent to the example above, but passes the explicit function `getter` that extracts the control signal. This function can extract any arbitrary value `z = f(x, u)`
```julia
getter = (si,p,t)->SVector(si.u[]) # Extract the signal to penalize differences for, in this case, the penalized signal `z = u`

difference_cost = DifferenceCost(getter) do e, p, t
    dot(e, p.Q3, e) # Compute the penalty given a difference `e = Δz`
end
```

# Extended help
It it common to penalize control-input differences in MPC controllers for multiple reasons, some of which include
- Reduce actuator wear and tear due to high-frequency actuation.
- Avoid excitation of higher-order and unmodeled dynamics such as structural modes, often occuring at higher frequencies.
- *Reduce* stationary errors without the presence of integral action.
- *Eliminate* stationary errors in the presence of input integration, see [Integral action](@ref) for more details.
"""
struct DifferenceCost{F,M} <: AbstractTrajectoryCost
    metric::M
    fun::F
end

get_control(si::StageInput, p, t) = si.u
DifferenceCost(metric) = DifferenceCost(metric, get_control)

## Evaluate ====================================================================

@generated function evaluate(obj::Objective{C}, oi::ObjectiveInput, p, t) where {N, C <: NTuple{N, Any}}
    quote # The function is generated to get around problems with type stability from heterogenous types in costs
        T = eltype(oi.x)
        s::T = zero(T)
        Base.Cartesian.@nexprs $N i -> s += evaluate(obj.costs[i], oi, p, t) 
        s
    end
end

function evaluate(cost::StageCost, si::StageInput, p, t)
    cost.fun(si, p, t)
end

function evaluate(cost::StageCost{F}, oi::ObjectiveInput, p, t) where F
    T = eltype(oi.x)
    s::T = zero(T)
    # sum(si->cost(si, p, t), stage_inputs(oi)) # Infers poorly
    for (i, si) in enumerate(stage_inputs(oi))
        treal = t + (i-1)*oi.discretization.Ts
        s += cost(si, p, treal)
    end
    s
end

function evaluate(cost::TerminalCost{F}, oi::ObjectiveInput, p, t) where F
    tf = t + oi.discretization.N*oi.discretization.Ts
    cost.fun(TerminalInput(oi), p, tf)
end

function evaluate(cost::DifferenceCost{F1,F2}, oi::ObjectiveInput{NX,NU}, p, t) where {F1,F2,NX,NU}
    s0 = StageInput(SVector{NX}(oi.x0), SVector{NU}(oi.u0), get_timeindex(oi.r, 1), 0)
    z0 = cost.fun(s0, p, t)
    c = zero(eltype(oi.u))
    if z0 isa StaticArray
        for (i, si) in enumerate(stage_inputs(oi))
            treal = t + (i-1)*oi.discretization.Ts
            z1 = cost.fun(si, p, treal)
            e = z1 .- z0
            c += cost.metric(e, p, treal)
            z0 = z1
        end
    else
        e = similar(z0, eltype(oi.u))
        for (i, si) in enumerate(stage_inputs(oi))
            treal = t + (i-1)*oi.discretization.Ts
            z1 = cost.fun(si, p, treal)
            @. e = z1 - z0
            c += cost.metric(e, p, treal)
            z0 = z1
        end
    end
    c
end