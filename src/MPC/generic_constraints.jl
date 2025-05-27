using Setfield
using Symbolics
using Bumper
## Constraints =================================================================

abstract type AbstractGenericMPCConstraint end
abstract type AbstractStageConstraint <: AbstractGenericMPCConstraint end
abstract type AbstractTrajectoryConstraint <: AbstractGenericMPCConstraint end
(cons::AbstractGenericMPCConstraint)(c, args...) = evaluate!(c, cons, args...)

"""

ConstraintEvaluator{C, OI <: ObjectiveInput, V <: Variables, P, T, S}

This structure is used internally in the GenericMPCProblem to handle evaluation of the constraint function. It is mutable so that `t` can be updated.
"""
mutable struct ConstraintEvaluator{C,OI<:ObjectiveInput,V<:Variables,P,T,S} <: Function
    constraint::C
    objective_input::OI
    vars::V
    p::P
    t::T
    scale_x::S
    scale_u::S
    robust_horizon::Int
end

function (ce::ConstraintEvaluator{C,OI,V})(c, x_::AbstractArray{T}, p = ce.p) where {C,OI,V,T}
    @unpack vars, scale_x, scale_u, constraint, objective_input, t, robust_horizon = ce
    @no_escape begin
        x_cache = isbits(T) ? @alloc(eltype(x_), size(vars.x_cache)...) : vars.x_cache
        u_cache = isbits(T) ? @alloc(eltype(x_), size(vars.u_cache)...) : vars.u_cache
        u_cache_int = isbits(T) ? @alloc(eltype(x_), size(vars.u_cache_int)...) : vars.u_cache_int
        vars = remake(vars; vars=x_, x_cache, u_cache, u_cache_int)
        unscale!(vars, scale_x, scale_u)
        nc = length(constraint)
        nu = vars.nu
        n_robust = vars.n_robust
        local u1 # control inputs for the first robust instantiation
        cinds = (1:nu*robust_horizon) .+ nc*n_robust
        for ri in 1:n_robust
            x, u = get_xu(vars, objective_input.u0, ri)
            if ri == 1
                u1 = @views u[:, 1:robust_horizon]
            end
            oi = remake(objective_input; x, u)
            constraint(@view(c[(1:nc) .+ (ri-1)*nc]), oi, get_parameter_index(p, ri), t)
            if ri > 1
                # Constrain the initial robust_horizon control inputs to be the same for all plant instantiations
                # the total number of additional constraints is nu*robust_horizon*(n_robust-1)
                ui = @views u[:, 1:robust_horizon]
                c[cinds] .= vec(u1) .- vec(ui)
                cinds = cinds .+ nu*robust_horizon    
            end
        end
        scale!(vars, scale_x, scale_u)
    end
    c
end

# The method below is to get around problems with RuntimeGeneratedFunctions and Nums
function (ce::ConstraintEvaluator{C,OI,V})(c, x_::AbstractArray{<:Num}, p = ce.p) where {C,OI,V}
    @unpack vars, scale_x, scale_u, constraint, objective_input, t, robust_horizon = ce
    @no_escape begin
        vars = remake(vars; vars=x_)
        unscale!(vars, scale_x, scale_u)
        nc = length(constraint)
        nu = vars.nu
        n_robust = vars.n_robust
        local u1 # control inputs for the first robust instantiation
        cinds = (1:nu*robust_horizon) .+ nc*n_robust
        for ri in 1:vars.n_robust
            x, u = get_xu(vars, objective_input.u0, ri)
            if ri == 1
                u1 = @views u[:, 1:robust_horizon]
            end
            oi = remake(objective_input; x=Matrix(x), u=Matrix(u))
            constraint(@view(c[(1:nc) .+ (ri-1)*nc]), oi, get_parameter_index(p, ri), t)
            if ri > 1
                # Constrain the initial robust_horizon control inputs to be the same for all plant instantiations
                # the total number of additional constraints is nu*robust_horizon*(n_robust-1)
                ui = @views u[:, 1:robust_horizon]
                c[cinds] .= vec(u1) .- vec(ui)
                cinds = cinds .+ nu*robust_horizon    
            end
        end
        scale!(vars, scale_x, scale_u)
    end
    c
end

struct InitialStateConstraint{DY,X,S,D} <: AbstractTrajectoryConstraint
    x0::X
    scale_x::S
    diff_inds::D
    alg_inds::D
    dynamics::DY # This is needed to make sure initial algebraic variables satisfy the dynamics, but not the initial condition. The initial condition for algebraic vars must be free for the initial control signal to be free
end

function InitialStateConstraint(x0, scale_x, dynamics)
    if dynamics.na > 0
        diff_inds = 1:dynamics.nx-dynamics.na
        alg_inds = dynamics.nx-dynamics.na+1:dynamics.nx
        InitialStateConstraint{typeof(dynamics), typeof(x0), typeof(scale_x), typeof(diff_inds)}(x0, scale_x, diff_inds, alg_inds, dynamics)
    else
        # For ODEs, we store nothing as dynamics for dispatch purposes
        diff_inds = 1:dynamics.nx
        alg_inds = 1:-1
        InitialStateConstraint{Nothing, typeof(x0), typeof(scale_x), typeof(diff_inds)}(x0, scale_x, diff_inds, alg_inds, nothing)
    end
end

function _check_promote_bounds(lb, ub)
    lb = replace(lb, NaN => -Inf)
    ub = replace(ub, NaN => Inf)
    length(lb) == length(ub) || throw(ArgumentError("lower and upper bounds must have the same length"))
    all(l ≤ u for (l, u) in zip(lb, ub)) || throw(ArgumentError("lower bound must be less than or equal to upper bound"))
    promote(lb, ub)
end

"""
    TerminalStateConstraint(f, lcons, ucons)

Create a constraint that holds for the terminal set `x(N+1)` in an [`GenericMPCProblem`](@ref).
NOTE: to implement simple bounds constraints on the terminal state, it is more efficient to use [`BoundsConstraint`](@ref).

# Arguments:
- `f`: A function (ti, p, t)->v that computes the constrained output, where `ti` is an object of type [`TerminalInput`](@ref).
- `lcons`: A vector of lower bounds for `v`.
- `ucons`: A vector of upper bounds for `v`, set equal to `lcons` to indicate equality constraints.

# Example
This example shows how to force the terminal state to equal a particular reference point `r`
```julia
terminal_constraint = TerminalStateConstraint(r, r) do ti, p, t
    ti.x
end
```
To make the terminal set a box ``l ≤ x ≤ u``, use
```julia
terminal_constraint = TerminalStateConstraint(l, u) do ti, p, t
    ti.x
end
```
"""
struct TerminalStateConstraint{F, C} <: AbstractTrajectoryConstraint
    fun::F
    lcons::C
    ucons::C
    function TerminalStateConstraint(fun, lcons, ucons)
        lb, ub = _check_promote_bounds(lcons, ucons)
        new{typeof(fun), typeof(lb)}(fun, lb, ub)
    end
end

struct StageConstraint{F, C} <: AbstractStageConstraint
    fun::F
    lcons::C
    ucons::C
    N::Int
end

struct DifferenceConstraint{M, F, C} <: AbstractTrajectoryConstraint
    metric::M
    fun::F
    lcons::C
    ucons::C
    N::Int
end

"""
    StageConstraint(f, lcons, ucons, N)

Create a constraint that holds for each stage of a [`GenericMPCProblem`](@ref). The constraint may be any nonlinear function of states and inputs.
NOTE: to implement simple bounds constraints on the states or control inputs, it is more efficient to use [`BoundsConstraint`](@ref).

# Arguments:
- `f`: A function (si, p, t)->v that computes the constrained output, where `si` is an object of type [`StageInput`](@ref).
- `lcons`: A vector of lower bounds for `v`.
- `ucons`: A vector of upper bounds for `v`, set equal to `lcons` to indicate equality constraints.
- `N`: The optimization horizon.

# Example:
This example creates a constraints that bounds the square of a single input ``1 ≤ u^2 ≤ 3`` and the sum of the state components ``-4 ≤ \\sum x_i ≤ 4``. Note that we create `v` as a static array for maximum performance.
```julia
control_constraint = StageConstraint([1, -4], [3, 4], N) do si, p, t
    SA[
        si.u[1]^2
        sum(si.x)
    ]
end
```
"""
function StageConstraint(f, lcons, ucons, N=-1)
    lb, ub = _check_promote_bounds(lcons, ucons)
    StageConstraint{typeof(f), typeof(lb)}(f, lb, ub, N=-1)
end

"""
    BoundsConstraint(xmin, xmax, umin, umax, xNmin, xNmax, dumin, dumax)
    BoundsConstraint(; xmin, xmax, umin, umax, xNmin=xmin, xNmax=xmax, dumin=-Inf, dumax=Inf)

Upper and lower bounds for state and control inputs. This constraint is typically more efficient than [`StageConstraint`](@ref) for simple bounds constraints.
`dumin` and `dumax` are the bounds on the change in control inputs from one stage to the next.

Separate bounds may be provided for the terminal state `xN`, if none are given, the terminal state is assumed to have the same bounds as the rest of the state trajectory.
"""
struct BoundsConstraint{C} <: AbstractStageConstraint
    xmin::C
    xmax::C
    umin::C
    umax::C
    xNmin::C
    xNmax::C
    dumin::C
    dumax::C
end
function BoundsConstraint(; xmin, xmax, umin, umax, xNmin=xmin, xNmax=xmax, dumin = fill(float(eltype(umin))(-Inf), length(umin)), dumax = fill(float(eltype(umax))(Inf), length(umax)))
    lbx, ubx = _check_promote_bounds(xmin, xmax)
    lbu, ubu = _check_promote_bounds(umin, umax)
    lbxN, ubxN = _check_promote_bounds(xNmin, xNmax)
    lbdu, ubdu = _check_promote_bounds(dumin, dumax)
    BoundsConstraint(promote(lbx, ubx, lbu, ubu, lbxN, ubxN, lbdu, ubdu)...)
end

struct TrajectoryConstraint{F,C} <: AbstractTrajectoryConstraint
    fun::F
    lcons::C
    ucons::C
end

struct CompositeMPCConstraint{C <: Tuple} <: AbstractGenericMPCConstraint
    constraints::C
end
function CompositeMPCConstraint(N::Int, Ts, scale_x, scale_u, threads, a, args...)
    # Initialize all fields of constraints
    cons0 = (a, args...)
    cons = ntuple(length(cons0)) do i
        ci = cons0[i]
        if hasfield(typeof(ci), :N)
            ci.N == -1 || ci.N == N || error("Inconsistent N for $ci")
            @set! ci.N = N
        end
        if hasfield(typeof(ci), :Ts)
            ci.Ts == -1 || ci.Ts == Ts || error("Inconsistent Ts, got $Ts and $(ci.Ts)")
            @set! ci.Ts = Ts
        end
        if hasfield(typeof(ci), :scale_x)
            @set! ci.scale_x = scale_x
        end
        if hasfield(typeof(ci), :scale_u)
            @set! ci.scale_u = scale_u
        end
        if hasfield(typeof(ci), :threads)
            @set! ci.threads = threads
        end
        ci
    end
    CompositeMPCConstraint(cons)
end


Base.length(c::InitialStateConstraint) = length(c.x0)
Base.length(c::TerminalStateConstraint) = length(c.lcons)
Base.length(c::StageConstraint) = length(c.lcons)*c.N
Base.length(c::DifferenceConstraint) = length(c.lcons)*c.N
Base.length(c::BoundsConstraint) = 0 # This constraint is handled separately by the solver
Base.length(c::CompositeMPCConstraint) = sum(length, c.constraints)

## Bounds ======================================================================

get_bounds(c::AbstractGenericMPCConstraint) = c.lcons, c.ucons # Default fallback
get_bounds(c::InitialStateConstraint{Nothing}) = c.x0 ./ c.scale_x, c.x0 ./ c.scale_x 
function get_bounds(c::InitialStateConstraint{<:FunctionSystem})
    # For DAEs, we have equality constraints corresponding to the algebraic equations
    b = zeros(length(c))
    b[c.diff_inds] .= c.x0[c.diff_inds] ./ c.scale_x[c.diff_inds]
    b, b
end
get_bounds(c::StageConstraint) = repeat(c.lcons, c.N), repeat(c.ucons, c.N)
get_bounds(c::DifferenceConstraint) = repeat(c.lcons, c.N), repeat(c.ucons, c.N)
get_bounds(c::BoundsConstraint{C}) where C = eltype(C)[],eltype(C)[]
function get_bounds(@nospecialize constraints::CompositeMPCConstraint)
    lu = map(constraints.constraints) do ci
        l, u = get_bounds(ci)
        @assert length(l) == length(u) == length(ci) "Inconsistent bound lengths for $ci"
        l, u
    end
    lcons = reduce(vcat, first.(lu))
    ucons = reduce(vcat, last.(lu))
    @assert length(lcons) == length(ucons) == length(constraints)
    lcons, ucons
end

update_bounds!(lcons, ucons, c::AbstractGenericMPCConstraint, oi::ObjectiveInput) = nothing # Default fallback
function update_bounds!(lcons, ucons, c::InitialStateConstraint{Nothing}, oi::ObjectiveInput)
    x0, scale_x = oi.x0, c.scale_x
    # c.x0  .= x0 # note: not updating x0 in the constraint object so that we can use it to solve MPC problems repeatedly from the same initial point
    lcons .= x0 ./ scale_x
    ucons .= x0 ./ scale_x 
    nothing
end

# DAE
function update_bounds!(lcons, ucons, c::InitialStateConstraint{<:FunctionSystem}, oi::ObjectiveInput)
    x0, scale_x, di = oi.x0, c.scale_x, c.diff_inds
    @views lcons[di] .= x0[di] ./ scale_x[di]
    @views ucons[di] .= x0[di] ./ scale_x[di] 
    nothing
end

function update_bounds!(lcons, ucons, constraints::CompositeMPCConstraint, oi::ObjectiveInput)
    last = 0
    for ci in constraints.constraints
        ci isa BoundsConstraint && continue # This constraint is handled separately by the solver
        inds = (1:length(ci)) .+ last
        @views update_bounds!(lcons[inds], ucons[inds], ci, oi)
        last = inds[end]
    end
end

function get_variable_bounds(constraints::CompositeMPCConstraint, objective_input::ObjectiveInput, scale_x, scale_u, n_robust)
    nx, Nx = size(objective_input.x)
    nu, Nu = size(objective_input.u)
    for c in constraints.constraints
        if c isa BoundsConstraint
            length(c.xmin) == length(c.xmax) == nx || throw(ArgumentError("Inconsistent length of variable bounds. xmin and xmax must have length nx=$nx"))
            length(c.umin) == length(c.umax) == nu || throw(ArgumentError("Inconsistent length of variable bounds. umin and umax must have length nu=$nu"))
            lbx = [fill(-Inf, nx); repeat(c.xmin ./ scale_x, Nx-2); c.xNmin ./ scale_x;]
            lbu = repeat(c.umin ./ scale_u, Nu)
            ubx = [fill(Inf, nx); repeat(c.xmax ./ scale_x, Nx-2); c.xNmax ./ scale_x;]
            ubu = repeat(c.umax ./ scale_u, Nu)

            lbx = repeat(lbx, n_robust)
            lbu = repeat(lbu, n_robust)
            ubx = repeat(ubx, n_robust)
            ubu = repeat(ubu, n_robust)

            lb = [lbx; lbu]
            ub = [ubx; ubu]
            # TODO: The first nx entries are for the initial-state constraint, which is currently handled using a generic constraint, optimize this!
            return lb, ub
        end
    end
    return nothing, nothing
end

## Evaluate ====================================================================

function evaluate!(c, x0::InitialStateConstraint{Nothing}, oi::ObjectiveInput, p0, t)
    p = get_system_parameters(p0)
    c .= get_timeindex(oi.x, 1) ./ x0.scale_x
end

function evaluate!(c, x0::InitialStateConstraint{<:FunctionSystem}, oi::ObjectiveInput, p0, t)
    p = get_system_parameters(p0)
    diff_inds, alg_inds = x0.diff_inds, x0.alg_inds
    x = get_timeindex(oi.x, 1)
    u = get_timeindex(oi.u, 1)
    @views c[diff_inds] .= x[diff_inds]
    xp = x0.dynamics(x, u, p, t)
    @views c[alg_inds] .= xp[alg_inds] # note: for some problems, the problem won't converge when using xp instead of x for unless the mu stragegy is set to adaptive. using xp is the right thing though, otherwise initial control signal may be constrained by arbitrary user-provided algebraic initial conditions
    c ./= x0.scale_x
end

function evaluate!(c, con::TerminalStateConstraint, oi::ObjectiveInput, p0, t)
    p = get_mpc_parameters(p0)
    tf = t + oi.discretization.N*oi.discretization.Ts
    c .= con.fun(TerminalInput(oi), p, tf)
end

function evaluate!(c, con::StageConstraint, si::StageInput, p0, t)
    p = get_mpc_parameters(p0)
    c .= con.fun(si, p, t)
end

function evaluate!(c, con::StageConstraint, oi::ObjectiveInput, p0, t)
    p = get_mpc_parameters(p0)
    nc = length(con.lcons)
    inds = 1:nc
    for (i, si) in enumerate(stage_inputs(oi))
        treal = t + (i-1)*oi.discretization.Ts
        ci = @view c[inds]
        evaluate!(ci, con, si, p, treal)
        inds = inds .+ nc
    end
end

function evaluate!(c, con::DifferenceConstraint, oi::ObjectiveInput, p0, t)
    p = get_mpc_parameters(p0)
    s0 = StageInput(oi.x0, oi.u0, get_timeindex(oi.r, 1), 0)
    z0 = con.fun(s0, p, t)
    nc = length(con.lcons)
    inds = 1:nc
    for (i, si) in enumerate(stage_inputs(oi))
        treal = t + (i-1)*oi.discretization.Ts
        z1 = con.fun(si, p, treal)
        e = z1 .- z0
        ci = @view c[inds]
        ci .= con.metric(e, p, treal)
        z0 = z1
        inds = inds .+ nc
    end
    c
end

evaluate!(c, ::BoundsConstraint, args...) = nothing # This constraint is handled separately by the solver

@generated function evaluate!(c, cons::CompositeMPCConstraint{C}, oi::ObjectiveInput, p, t) where {N, C <: NTuple{N, Any}}
    quote # The function is generated to get around problems with type stability from heterogenous types in cons.constraints
        last = 0
        Base.Cartesian.@nexprs $N i -> begin
            con_i = cons.constraints[i]
            if !(con_i isa BoundsConstraint) # This constraint is handled separately by the solver
                inds = (1:length(con_i)) .+ last
                @views evaluate!(c[inds], con_i, oi, p, t)
                last = inds[end]
            end
        end
        c
    end
end

