using Markdown, UnPack
using StaticArrays
import SciMLBase.remake
abstract type AbstractSignal end
abstract type CompositeSignal end
using MonteCarloMeasurements

"""
    MPCParameters(p)
    MPCParameters(p, p_mpc)

A struct containing two sets of parameters, `p` belong to the dynamics of the system being controlled, and `p_mpc` belong to cost and constraint functions. If `p_mpc` is not supplied, it defaults to be the same as `p`.

For robust MPC formulations using the `robust_horizon`, a vector of `MPCParameters` must be used to store the parameters for each uncertain realizaiton. 
"""
struct MPCParameters{P, PM}
    p::P
    p_mpc::PM
end

MPCParameters(p) = MPCParameters(p, p)

function MCMMPCParameters(p, p_mpc=p)
    N1 = nparticles(p)
    N2 = nparticles(p_mpc)

    N1 == 1 || N1 == N2 || error("Inconsistent number of parameters")
    N = max(N1, N2)

    if N1 > 1
        pu = [MonteCarloMeasurements.vecindex(p, i) for i = 1:N]
    else
        pu = [p for i = 1:N]
    end

    if N2 > 1
        pu_mpc = [MonteCarloMeasurements.vecindex(p_mpc, i) for i = 1:N]
    else
        pu_mpc = [p_mpc for i = 1:N]
    end

    MPCParameters.(pu, pu_mpc)
end


function get_parameter_index(p::MPCParameters, i::Int)
    i == 1 || throw(ArgumentError("When using robust MPC, a vector of parameters must be supplied."))
    p
end

function get_parameter_index(p::Vector{<:MPCParameters}, i::Int)
    p[i]
end

function get_parameter_index(p, i::Int)
    i == 1 || throw(ArgumentError("When using robust MPC, parameters must be supplied in the form of a vector of `MPCParameters`."))
    p
end

function get_system_parameters(p0::MPCParameters)
    p0.p
end

function get_mpc_parameters(p0::MPCParameters)
    p0.p_mpc
end

get_system_parameters(p) = p
get_mpc_parameters(p) = p

function _check_valid_traj_sizes(x, u)
    N = size(u, 2)
    size(x, 2) == N+1 || throw(DimensionMismatch("Invalid sizes of x and u. x is expected to be one time step longer than u"))
    N
end



_signal_map =
md"""
```
  ┌───────────────┬──────────────────────┐
  │               │                      │
  │    ┌─────┐    │     ┌─────┐          │
w─┴───►│     │    └────►│     ├─────►v   │
       │     │ u        │     │          │
r─────►│ MPC ├──┬──────►│  P  ├─────►z   │
       │     │  │       │     │          │
 ┌────►│     │  │ d────►│     ├──┬──►    │
 │     └─────┘  │       └─────┘  │y      │
 │              │                │       │
 │   ┌───────┐  │                │       │
 │   │       │◄─┘                │       │
 │   │       │                   │       │
 └───┤  OBS  │◄──────────────────┘       │
     │       │                           │
     │       │◄──────────────────────────┘
     └───────┘
```
All signals relevant in the design of an MPC controller are specified in the block-diagram above. The user is tasked with designing the MPC controller as well as the observer.

The following signals are shown in the block diagram
- ``w`` is a *known* disturbance, i.e., its value is known to the controller through a measurement or otherwise.
- ``r`` is a reference value for the controlled output $z$.
- ``\hat x`` is an estimate of the state of the plant $P$.
- ``u`` is the control signal.
- ``v`` is a set of *constrained* outputs. This set may include direct feedthrough of inputs from $u$.
- ``z`` is a set of controlled outputs, i.e., outputs that will be penalized in the cost function.
- ``y`` is the measured output, i.e., outputs that are available for feedback to the observer. $z$ and $y$ may overlap.
- ``d`` is an *unknown* disturbance, i.e., a disturbance of which there is no measurement or knowledge.
"""


get_timeindex(x, i) = x
get_timeindex(x::AbstractMatrix, i) = @view x[:, i]
get_timeindex(x::AbstractMatrix, i, ::Val{n}) where n = SVector{n}(@view(x[:, i]))
terminal_state(sig::AbstractArray) = @view sig[:, end]
terminal_reference(sig::AbstractArray) = @view sig[:, end]
terminal_reference(sig::SVector, N::Int=1) = sig # SVectors don't support the end indexing above :/ 
terminal_state(sig::SVector, N::Int=1) = sig

terminal_state(sig::AbstractArray, N) = @view sig[:, N]
terminal_reference(sig::AbstractArray, N) = @view sig[:, N]

terminal_state(sig::AbstractVector, N) = sig
terminal_reference(sig::AbstractVector, N) = sig


"""
    ObserverInput <: CompositeSignal
    ObserverInput(u, y, r, w)
    ObserverInput(; u, y, r, w)

Contains all signals that are required to update an observer.

$_signal_map
"""
Base.@kwdef struct ObserverInput{VU<:AbstractVector,VY<:AbstractVector,VR,VW} <: CompositeSignal
    u::VU
    y::VY
    r::VR = []
    w::VW = []
end

"""
    PlantInput <: CompositeSignal

Contains all signals that are required to simulate the plant (outside of the controller).

$_signal_map
"""
struct PlantInput{VX,VU,VW,VD} <: CompositeSignal
    x::VX
    u::VU
    w::VW
    d::VD
end

"""
    ControllerInput <: CompositeSignal

Contains all signals that are required to run the MPC controller one iteration.

$_signal_map
"""
Base.@kwdef struct ControllerInput{VX,VR,VW,VU} <: CompositeSignal
    x::VX
    r::VR
    w::VW = []
    u0::VU = []
end


"""
    ControllerOutput <: CompositeSignal
    
Contains all signals resulting from one step of the MPC controller. Also contains `sol` which is the solution object from the optimization problem. `sol.retcode` indicates whether the optimization was successful or not.
- For [`GenericMPCProblem`](@ref) `sol.retcode` indicates whether the optimization was successful or not.
- For [`LQMPCProblem`](@ref) and [`QMPCProblem`](@ref) `sol.info.status` indicates whether the optimization was successful or not.

$_signal_map
"""
struct ControllerOutput{VU,VX,S} <: CompositeSignal
    x::VX
    u::VU
    sol::S
end

struct ObjectiveInput{NX,NU,D,X,U,R,X0,U0} <: CompositeSignal
    x::X
    u::U
    r::R
    x0::X0
    u0::U0
    discretization::D
    ObjectiveInput{NX,NU}(x, u, r, x0, u0, discretization) where {NX,NU} = new{NX,NU,typeof(discretization),typeof(x),typeof(u),typeof(r),typeof(x0),typeof(u0)}(x, u, r, x0, u0, discretization)
end

@inline static_nx(::ObjectiveInput{NX}) where NX = NX
@inline static_nu(::ObjectiveInput{<:Any,NU}) where NU = NU

"""
    ObjectiveInput(x, u, r[, x0, u0])

Create an ObjectiveInput structure containing trajectories for `x,u` and `r`. Initial state `x0` and initial control `u0` will be taken from `x` and `u`.

# Arguments:
- `x`: The state trajectory as a matrix of size `(nx, N+1)`
- `u`: The control trajectory as a matrix of size `(nu, N)`
- `r`: A reference vector or matrix of size `(nr, N+1)`, `nr` does not have to equal `nx`.
"""
function ObjectiveInput(x,u,r)
    ObjectiveInput{size(x,1), size(u,1)}(x,u,r,x[:,1],u[:,1],nothing)
end

function remake(oi::ObjectiveInput{NX,NU}; x=missing, u=missing, r=missing, x0=missing, u0=missing, discretization=missing) where {NX,NU}
    x_  = ismissing(x)  ? oi.x  : x
    u_  = ismissing(u)  ? oi.u  : u
    r_  = ismissing(r)  ? oi.r  : r
    x0_ = ismissing(x0) ? oi.x0 : x0
    u0_ = ismissing(u0) ? oi.u0 : u0
    discretization_ = ismissing(discretization) ? oi.discretization : discretization
    ObjectiveInput{NX,NU}(x_, u_, r_, x0_, u0_, discretization_)
end

function Base.getproperty(oi::ObjectiveInput, s::Symbol)
    s ∈ fieldnames(typeof(oi)) && return getfield(oi, s)
    if s === :nx
        return size(oi.x, 1)
    elseif s === :nu
        return size(oi.u, 1)
    elseif s === :nr
        return size(oi.r, 1)
    elseif s === :N
        return size(oi.u, 2)
    else
        throw(ArgumentError("$(typeof(oi)) has no property named $s"))
    end
end

"""
    StageInput{X, U, R}

Structure that holds the input to [`StageCost`](@ref) functions.

# Fields:
- `x::X`
- `u::U`
- `r::R`
- `i::Int`: The time index of the stage. This index typically ranges from `1` to `N`, but [`DifferenceCost`](@ref) will receive indices starting at `i=0`.
"""
struct StageInput{X,U,R} <: CompositeSignal
    x::X
    u::U
    r::R
    i::Int
end

"""
    TerminalInput{X, R}

Structure that holds the input to [`TerminalCost`](@ref) functions.

# Fields:
- `x::X`
- `r::R`
"""
struct TerminalInput{X,R} <: CompositeSignal
    x::X
    r::R
end


"""
    Variables{D, T}

Structure that holds all optimization variables for [`GenericMPCProblem`](@ref).

# Fields:
- `vars::Vector{T}`: the memory layout is given by `[vec(x); vec(u)]`. See [`get_xu`](@ref) to extract `x` and `u`.
- `nx::Int`
- `nu::Int`
"""
struct Variables{D, T, MT <: AbstractMatrix{}}
    vars::Vector{T}
    nx::Int
    nu::Int
    N::Int
    disc::D
    n_robust::Int
    x_cache::MT
    u_cache::MT
    u_cache_int::MT
end

function Variables{D, T}(vars, nx, nu, N, disc, n_robust) where {D, T}
    xinds = all_x_indices(disc, N, nx, nu, n_robust, 1)
    x_cache = zeros(T, nx, length(xinds) ÷ nx)
    u_cache = zeros(T, nu, N)
    u_cache_int = zeros(T, nu, N)
    Variables{D, T, Matrix{T}}(vars, nx, nu, N, disc, n_robust, x_cache, u_cache, u_cache_int)
end

Base.length(vars::Variables) = length(vars.vars)

function remake(v::Variables{D}; vars, x_cache=v.x_cache, u_cache=v.u_cache, u_cache_int=v.u_cache_int) where {D}
    # NOTE: this remake will cause copies in the Variables constructor each time step. The problem is that the original vars is Float64 but we remake it with Duals. This causes convert to be called on the cache arrays.
    Variables{D,eltype(vars),typeof(x_cache)}(vars, v.nx, v.nu, v.N, v.disc, v.n_robust, x_cache, u_cache, u_cache_int)
end

"""
    x, u = get_xu(vars::Variables, u0=nothing, ri=1)
    x, u = get_xu(prob::GenericMPCProblem, ri=1)

Extract `x` and `u` matrices for robust index `ri`. This function handles input integration for [`FunctionSystem`](@ref)s with specified `input_integrators`, this requires the specification of `u0` if Variables are passed. 
"""
function get_xu(vars::Variables{D}, u0=nothing, ri=1) where D
    @unpack nx, nu, n_robust, u_cache, u_cache_int = vars
    N = horizon(vars)
    xinds = all_x_indices(vars.disc, N, nx, nu, vars.n_robust, ri)
    uinds = all_u_indices(vars.disc, N, nx, nu, vars.n_robust, ri)
    x = @views reshape(vars.vars[xinds], nx, :) # NOTE: updated in-place later so do not return cache array
    u = @views reshape(vars.vars[uinds], nu, :) # NOTE: updated in-place later so do not return cache array

    input_integrators = vars.disc.dyn.input_integrators
    if !isempty(input_integrators) && u0 !== nothing 
        # integrate_inputs!(u_cache_int, u, u0, input_integrators)
        # return x, @views reshape(vec(u_cache_int)[1:length(uinds)], nu, :) # The gymastics with reshape and view here is unfortunate, but necessary to avoid allocations due to unstable return type
        o = similar(u) # NOTE: use Bump allocator? Make sure to assign into u in that case and don't return o. Also make sure this does not screw with vars.vars of which u is a view. u_cache_int was created to fix this, see if it can be used again now (it's already bump allocated in the caller of this function
        integrate_inputs!(o, u, u0, input_integrators)
        u = o
    end

    x, u
end

terminal_state(oi::ObjectiveInput) = terminal_state(oi.x, size(oi.x, 2))
terminal_reference(oi::ObjectiveInput) = terminal_reference(oi.r, size(oi.x, 2))
horizon(oi::ObjectiveInput) = size(oi.u, 2)


TerminalInput(oi::ObjectiveInput) = TerminalInput(terminal_state(oi), terminal_reference(oi))



all_x_indices(disc, oi::ObjectiveInput, n_robust, ri=1) = all_x_indices(disc, oi.N, oi.nx, oi.nu, n_robust, ri)
all_u_indices(disc, oi::ObjectiveInput, n_robust, ri=1) = all_u_indices(disc, oi.N, oi.nx, oi.nu, n_robust, ri)


using ResumableFunctions
@resumable function x_indices(N::Int, nx::Int, nu::Int)::UnitRange{Int}
    inds = 1:nx
    for i = 1:N+1
        @yield inds
        inds = inds .+ nx
    end
end

@resumable function u_indices(N::Int, nx::Int, nu::Int)::UnitRange{Int}
    inds = (1:nu) .+ (N+1)*nx
    for i = 1:N
        @yield inds
        inds = inds .+ nu
    end
end