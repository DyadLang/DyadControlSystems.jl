using SparseArrays
import ControlSystemsBase: state_names, input_names, output_names
import ModelingToolkit: linearize

issystem(::Any) = false

# This is the MPC version of linearize
const VecOrNum = Union{AbstractVector, Number}
"""
    A, B = linearize(f, x, u, p, t)
    A, B = linearize(f, op::OperatingPoint, p, t)

Linearize dynamics `f` around operating point (x,u,p,t) using ForwardDiff.
"""
function linearize(f, xi::VecOrNum, ui::VecOrNum, p, t)
    A = ForwardDiff.jacobian(x -> f(x, ui, p, t), xi)
    B = ForwardDiff.jacobian(u -> f(xi, u, p, t), ui)
    A, B
end

## LTISystem ===================================================================
"""
    linearize(P::LTISystem)

`linearize` on an LTISystem is the identity function.
"""
linearize(P::LTISystem, args...; kwargs...) = P

issystem(::LTISystem) = true

(sys::AbstractStateSpace)(x,u,p,t) = sys.A*x + sys.B*u # Technically a pirated method, but exceedingly unlikely to cause any problems

# Spell out all the arguments to not overload linearize that returns all matrices
linearize(sys::AbstractStateSpace, xi::VecOrNum, ui::VecOrNum, p, t) = sys.A, sys.B

## ODESystem ===================================================================
using ModelingToolkit: ODESystem

function ControlSystemsBase.input_names(P::ODESystem)
    return string.(ModelingToolkit.inputs(P))
end

function ControlSystemsBase.output_names(P::ODESystem)
    return string.(ModelingToolkit.outputs(P))
end

function ControlSystemsBase.state_names(P::ODESystem)
    return string.(ModelingToolkit.unknowns(P))
end

issystem(::ODESystem) = true

## FunctionSystem ==============================================================

"""
    FunctionSystem{TE <: ControlSystemsBase.TimeEvolution, F, G}

A structure representing the dynamical system
```math
\begin{aligned}
x′ &= f(x,u,p,t)\\
y  &= g(x,u,p,t)
\end{aligned}
```

To build a FunctionSystem for a DAE on mass-matrix form, use an `ODEFunction` as `f`
```
f = ODEFunction(dae_dyn, mass_matrix = M)
```

To obtain the simplified system and default values for the initial condition and parameters, see [`simplified_system`](@ref).

# Fields:
- `dynamics::F`
- `measurement::G`
- `timeevol::TE`: ControlSystemsBase.TimeEvolution
- `x`: states
- `u`: controlled inputs
- `y`: measured outputs
- `w`: disturbance inputs
- `z`: performance outputs
- `p`: parameters
"""
Base.@kwdef struct FunctionSystem{TE <: ControlSystemsBase.TimeEvolution, F, G, X, U, Y, W, Z, P, I, M}
    dynamics::F
    measurement::G
    timeevol::TE
    x::X
    u::U
    y::Y
    w::W = nothing
    z::Z = x isa Symbol ? (1:1) : (1:length(x)) # Default to all states
    p::P = DiffEqBase.NullParameters()
    input_integrators::I = 0:-1
    meta::M = nothing
end
FunctionSystem(d,m,t,x,u,y,w,z,p) = FunctionSystem(d,m,t,x,u,y,w,z,p,0:-1,nothing)


"""
    FunctionSystem(f, g;           kwargs...)
    FunctionSystem(f, g, Ts::Real; kwargs...)

Constructor for `FunctionSystem`.

# Arguments:
- `f`: Discrete dynamics with signature (x,u,p,t)
- `g`: Measurement function with signature (x,u,p,t)
- `Ts`: If the sample time `Ts` is provided, the system represents a discrete-time system, otherwise the dynamics is assumed to be continuous.
- `kwargs`: Signal names
"""
FunctionSystem(f, g, Ts::Real; kwargs...) = FunctionSystem(; dynamics=f, measurement=g,timeevol=Discrete(Ts), kwargs...)
FunctionSystem(f, g; kwargs...) = FunctionSystem(; dynamics=f, measurement=g,timeevol=Continuous(), kwargs...)

function FunctionSystem(G::AbstractStateSpace{<: Discrete})
    G isa NamedStateSpace || (G = named_ss(G))
    dynamics, measurement = let A = to_static(G.A), B = to_static(G.B), C = to_static(G.C), D = to_static(G.D)
        function dynamics(x, u, p, t)
            return A * x + B * u
        end
        function measurement(x, u, p, t)
            return C * x + D * u
        end
        dynamics, measurement
    end
    return FunctionSystem(dynamics, measurement, G.Ts; G.x, G.u, G.y)
end

(sys::FunctionSystem{F})(args...) where F = getfield(sys, :dynamics)(args...)

function Base.getproperty(sys::FunctionSystem, s::Symbol)
    s ∈ fieldnames(typeof(sys)) && return getfield(sys, s)
    if s === :Ts
        return getfield(getfield(sys, :timeevol), :Ts)
    elseif s === :nx
        x = getfield(sys, :x)
        return x isa Symbol ? 1 : length(sys.x)::Int
    elseif s === :nu
        u = sys.u
        return u isa Symbol ? 1 : length(u)::Int
    elseif s === :ny
        y = sys.y
        return y isa Symbol ? 1 : length(y)::Int
    elseif s === :nw
        w = sys.w
        return w isa Symbol ? 1 : length(w)::Int
    elseif s === :nz
        z = sys.z
        return z isa Symbol ? 1 : length(z)::Int
    elseif getfield(sys, :x) isa Symbol ? s === getfield(sys, :x) : s ∈ sys.x
        return s
    elseif s === :na
        dyn = getfield(sys, :dynamics)
        hasproperty(dyn, :mass_matrix) || return 0
        M = dyn.mass_matrix
        if dyn isa ODEFunction && M !== nothing
            nx = size(M, 1)
            for i = nx:-1:1
                if !iszero(M[i, i])
                    return nx-i
                end
            end
        else
            return 0
        end
    else
        throw(ArgumentError("$(typeof(sys)) has no property named $s"))
    end
end

"""
    simplified_system(funcsys::FunctionSystem)

Obtain the result from `structural_simplify` that was obtained after input-output processing. This system is often of a lower order than the original system. To obtain the default initial condition and parameters of the simplified system, call
```julia
ssys = simplified_system(funcsys)
defs = ModelingToolkit.defaults(ssys)
x0, p0 = ModelingToolkit.get_u0_p(ssys, defs, defs)
```
"""
simplified_system(sys::FunctionSystem) = sys.meta.simplified_system

function Base.propertynames(sys::FunctionSystem{Continuous}, private::Bool = false)
    return (fieldnames(typeof(sys))..., :nx, :ny, :nw, :nz, :na)
end

function Base.propertynames(sys::FunctionSystem{<:Discrete}, private::Bool = false)
    return (fieldnames(typeof(sys))..., :Ts, :nx, :ny, :nw, :nz, :na)
end

Base.Broadcast.broadcastable(f::FunctionSystem) = Ref(f)



function ControlSystemsBase.input_names(P::FunctionSystem)
    return vcat(string.(P.u))
end

function ControlSystemsBase.output_names(P::FunctionSystem)
    return vcat(string.(P.y))
end

function ControlSystemsBase.state_names(P::FunctionSystem)
    return vcat(string.(P.x))
end

ControlSystemsBase.isdiscrete(::FunctionSystem{TE}) where TE = TE <: Discrete

issystem(::FunctionSystem) = true

linearize(fs::FunctionSystem, x::VecOrNum, u::VecOrNum, args...) = linearize(fs.dynamics, x, u, args...) # The x required for ambiguity

"""
    rk4(f::FunctionSystem, Ts::Real)

Apply `rk4` on a `FunctionSystem`, return a discrete-time `FunctionSystem`.
"""
function rk4(f::FunctionSystem, Ts::Real; kwargs...)
    f.timeevol isa ControlSystemsBase.Continuous || error("The system is already discrete.")
    f.na == 0 || error("rk4 is not implemented for systems with algebraic variables.")
    dynamics = rk4(f.dynamics, Ts; kwargs...)
    FunctionSystem(dynamics, f.measurement, Ts; f.x, f.u, f.y, f.w, f.z, f.p)
end

"""
    add_input_integrators(f::FunctionSystem, ii = 1:f.nu)

Modify the dynamics of `f` to add integrators in series with each input indicated by `ii`.
The resulting system will have `length(ii)` additional states, appended after the original state vector, that corresponds to the control input, and the control inputs lsited in `ii` will instead refer to *input differences*, i.e., the difference between the current input and the previous input.

Adding input integration is a common method to endow MPC controllers with integral action.
"""
function add_input_integrators(f::FunctionSystem{<:Discrete}, ii = 1:f.nu)
    function dynamics(x, u, p, t)
        T = promote_type(eltype(x), eltype(u))
        uint = Vector{T}(undef, length(u)) # TODO: there are some allocations in this function
        uint .= u
        for (j, i) in enumerate(ii)
            uint[i] += x[f.nx+j]
        end
        @views x1 = f.dynamics(x[1:f.nx], uint, p, t)
        [x1; uint[ii]]
    end

    FunctionSystem(dynamics, f.measurement, f.Ts; x=[f.x; f.u], f.u, f.y, f.w, f.z, f.p)
end


function add_input_integrators(f::FunctionSystem{Continuous}, ii = 1:f.nu)
    ii = SVector(ii...)
    xinds = SVector((1:f.nx)...)
    uinds = SVector(((1:length(ii)) .+ f.nx)...)
    dynamics = let ii = ii, xinds = xinds, uinds = uinds
        @views function dynamics(x, u, p, t)
            uint = x[f.nx+1:end]
            x1 = f.dynamics(x[xinds], uint, p, t)
            [x1; u[ii]]
        end
    end

    FunctionSystem(dynamics, f.measurement; x=[f.x; f.u], f.u, f.y, f.w, f.z, f.p)
end