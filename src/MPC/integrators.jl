import DiffEqBase as DE
import SciMLBase as SciML
import OrdinaryDiffEq as ODE
using StaticArrays
import DyadControlSystems as JSC
import ControlSystems
using ForwardDiff
using ForwardDiff: JacobianConfig, jacobian, jacobian!, Dual, Chunk, Tag

struct ControlInputWrapper{F, U}
    dynamics::F
    u::U
end

function set_u!(w::ControlInputWrapper, u)
    # copyto!(get_tmp(w.u, u), u)
    copyto!(w.u, u)
end

function get_u(w::ControlInputWrapper, u)
    # get_tmp(w.u, u)
    w.u
end

"""
    control_input_wrapper(dynamics::F, nu)

Take a function `dynamics` that maps `(x,u,p,t)->x⁺` and return a closure over a cache vector named `u` of length `nu`. The closure has the signature `(x,p,t)->x⁺` and requires the closed-over `u` to be written to prior to calling the function.
"""
function control_input_wrapper(dynamics::F, nu) where F
    u = nu isa Integer ? zeros(nu) : nu
    ControlInputWrapper(dynamics, u)
end

function control_input_wrapper(dynamics::F, nu) where F <: ODEFunction
    ODEFunction(control_input_wrapper(dynamics.f, nu); dynamics.mass_matrix)
end

function (w::ControlInputWrapper)(x,p,t)
    w.dynamics(x, get_u(w, x), p, t)
end
function (w::ControlInputWrapper{<:ODEFunction})(x,p,t)
    w.dynamics.f(x, get_u(w, x), p, t)
end

function (w::ControlInputWrapper{<:ODEFunction{<:Any, <:ControlInputWrapper}})(x,p,t)
    w.dynamics.f.dynamics(x, get_u(w, x), p, t)
end

# Integrators
struct MPCIntegrator{I <: DE.DEIntegrator, R <: Real, DIX <: DE.DEIntegrator, DIU, CFX, CFU}
    int::I
    diffxint::DIX # Used to linearize using ForwardDiff 
    diffuint::DIU
    cfgx::CFX
    cfgu::CFU
    Ts::R
end

set_u!(i::MPCIntegrator, u) = set_u!(i.int.f.f, u)

function Base.getproperty(i::MPCIntegrator, s::Symbol)
    s ∈ fieldnames(typeof(i)) && return getfield(i, s)
    int = getfield(i, :int)
    getproperty(int, s) # Forward to the integrator
end


"""
    int = MPCIntegrator(dynamics, problem_constructor, alg::SciML.DEAlgorithm; Ts::Real, nx, nu, kwargs...)

Discretize a dynamics function on the form `(x,u,p,t)->ẋ` using DE-integrator `alg`. The resulting object `int` behaves like a discrete-time dynamics function `(x,u,p,t)->x⁺`

# Arguments:
- `dynamics`: The continuous-time dynamics. 
- `problem_constructor`: One of `ODEProblem, DAEProblem` etc.
- `alg`: Any `DEAlgorithm`, e.g., `Tsit5()`.
- `Ts`: The fixed sample time between control updates. The algorithm may take smaller steps internally.
- `nx`: The state (`x`) dimension.
- `nu`: The input (`u`) dimension.
- `kwargs`: Are sent to the integrator initialization function [`init`](https://diffeq.sciml.ai/stable/basics/integrator/#Initialization-and-Stepping).

# Example:
This example creates two identical versions of a discretized dynamics, one using the [`rk4`](@ref) function and one using `MPCIntegrator`. For the `MPCIntegrator`, we set `dt` and `adaptive=false` in order to get equivalent results.
```
using DyadControlSystems.MPC
using OrdinaryDiffEq

"Continuous-time dynamics of a quadruple tank system."
function quadtank(h,u,p=nothing,t=nothing)
    kc = 0.5
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a1, a3, a2, a4 = 0.03, 0.03, 0.03, 0.03
    γ1, γ2 = 0.3, 0.3

    ssqrt(x) = √(max(x, zero(x)) + 1e-3)
    # ssqrt(x) = sqrt(x)
    xd = @inbounds SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end

nx = 4 # Number of states
nu = 2 # Number of inputs
Ts = 2 # Sample time
x0 = SA[1.0,1,1,1]
u0 = SVector(zeros(nu)...)

discrete_dynamics_rk = MPC.rk4(quadtank, Ts)
discrete_dynamics    = MPC.MPCIntegrator(quadtank, ODEProblem, RK4(); Ts, nx, nu, dt=Ts, adaptive=false, saveat_end=false)

x1_rk = discrete_dynamics_rk(x0, u0, 0, 0)
x1    = discrete_dynamics(x0, u0, 0, 0)
@assert norm(x1 - x1_rk) < 1e-12
```
"""
function MPCIntegrator(dynamics, problem_constructor, alg::SciML.DEAlgorithm; Ts::Real, p=nothing, nx, nu, kwargs...)


    wrapped_dynamics  = control_input_wrapper(dynamics, nu)
    wrapped_dynamicsx = control_input_wrapper(dynamics, nu)
    wrapped_dynamicsu = control_input_wrapper(dynamics, [ForwardDiff.Dual{Tag{typeof(Bjac!)}, Float64, nu}(0) for _ in 1:nu])
    static = !(dynamics isa ODEFunction && dynamics.mass_matrix != I)
    if static
        x0   = SVector(zeros(nx)...)
        x0d  = SVector([ForwardDiff.Dual{Tag{typeof(Ajac!)}, Float64, nx}(0) for _ in 1:nx]...)
        x0d2 = SVector([ForwardDiff.Dual{Tag{typeof(Bjac!)}, Float64, nu}(0) for _ in 1:nx]...)
    else
        x0   = zeros(nx)
        x0d  = [ForwardDiff.Dual{Tag{typeof(Ajac!)}, Float64, nx}(0) for _ in 1:nx]
        x0d2 = [ForwardDiff.Dual{Tag{typeof(Bjac!)}, Float64, nu}(0) for _ in 1:nx]
    end
    prob = problem_constructor(wrapped_dynamics, x0, (0.0, Inf), p)
    int  = DE.init(prob, alg; save_everystep = false, force_dtmin=false, maxiters=typemax(Int), kwargs...) # Integrator state cannot be reset unless it is initialized with save_everystep=false
    # We also set maxiters to Inf since we are running our own loop, this allows us to benchmark the integrator step which otherwise maxes out.
    
    probd = problem_constructor(wrapped_dynamicsx, x0d, (0.0, Inf), p)
    intdx = DE.init(probd, alg; save_everystep = false, force_dtmin=true, maxiters=typemax(Int), kwargs...)

    probd2 = problem_constructor(wrapped_dynamicsu, x0d2, (0.0, Inf), p)
    intdu  = DE.init(probd2, alg; save_everystep = false, force_dtmin=true, maxiters=typemax(Int), kwargs...)

    cfgx = JacobianConfig(nothing, SVector(x0...))
    cfgu = JacobianConfig(nothing, SVector(zeros(nu)...))

    MPCIntegrator(int, intdx, intdu, cfgx, cfgu, Ts)
end


function (i::MPCIntegrator)(x, u, p=nothing, t=nothing)
    int = i.int
    set_u!(i, u) # Set the control input
    # reinit!(int, x)
    t === nothing ? DE.set_u!(int, x) : DE.set_ut!(int, x, t)
    int.u     = x # https://github.com/SciML/OrdinaryDiffEq.jl/issues/1638
    int.uprev = x
    old_p = int.p
    int.p = p # || error("MPCIntegrator does not support modifying p")
    initialize!(int, int.cache)
    DE.step!(int, i.Ts, true)
    u_modified!(int, true) # control input changes in the transition so the derivative in the last point is not valid.
    int.p = old_p
    int.u
end

const MPCIntOrFuncSys = Union{MPCIntegrator, JSC.FunctionSystem{<:ControlSystemsBase.Discrete, <:MPCIntegrator}}
unwrap_MPCInt(x) = x
unwrap_MPCInt(x::JSC.FunctionSystem) = x.dynamics

@views function rollout!(i::MPCIntOrFuncSys, x::AbstractMatrix, u, p=i.p)
    i = unwrap_MPCInt(i)
    nu, N = size(u)
    int   = i.int
    Ts    = i.Ts
    Tf    = N*Ts
    old_p = int.p
    int.p = p
    wrapped_dynamics = i.int.f.f
    @assert wrapped_dynamics isa ControlInputWrapper
    set_u!(wrapped_dynamics, u[:, 1])
    x0 = copy(x[:, 1])
    DE.set_ut!(int, x0, 0)
    # DE.reinit(int, x0)
    int.u     = x0 # https://github.com/SciML/OrdinaryDiffEq.jl/issues/1638
    int.uprev = x0
    initialize!(int, int.cache)
    for ind in axes(u, 2)
        set_u!(wrapped_dynamics, u[:, ind])
        u_modified!(int, true) # control input changes in the transition so the derivative in the last point is not valid.
        DE.step!(int, Ts, true)
        x[:, ind+1] .= int.u
    end
    int.p = old_p
    x, u
end

function JSC.linearize(i::MPCIntOrFuncSys, x0::AbstractVector, u0::Union{Number, AbstractVector}, p, t, args...)
    i = unwrap_MPCInt(i)
    A = zeros(length(x0), length(x0))
    B = zeros(length(x0), length(u0))
    Ajac!(A,i,x0,u0,p,t,args...)
    Bjac!(B,i,x0,u0,p,t,args...)
    A,B
end

function Ajac!(A::AbstractMatrix, i::MPCIntOrFuncSys, x0::AbstractVector, u0::Union{Number, AbstractVector}, p, t, args...; kwargs...)
    i = unwrap_MPCInt(i)
    # TODO: handle p
    intx = i.diffxint
    wrapped_dynamicsx = intx.f.f
    @unpack cfgx = i
    set_u!(wrapped_dynamicsx, u0)
    function jacfunA(x::AbstractVector{<:Dual{<:Any, T, N}}) where {T,N}
        x2 = reinterpret(Dual{Tag{typeof(Ajac!)}, T, N}, x)
        # reinit!(intx, x2)
        DE.set_ut!(intx, x2, t)
        intx.u     = x2 # https://github.com/SciML/OrdinaryDiffEq.jl/issues/1638
        intx.uprev = x2
        initialize!(intx, intx.cache)
        u_modified!(intx, true) 
        DE.step!(intx, i.Ts, true)
        reinterpret(eltype(x), intx.u)
    end
    jacobian!(A, jacfunA, x0, cfgx)
end

function Bjac!(B::AbstractMatrix, i::MPCIntOrFuncSys, x0::AbstractVector, u0::Union{Number, AbstractVector}, p, t, args...; kwargs...)
    i = unwrap_MPCInt(i)
    intu = i.diffuint
    wrapped_dynamicsu = intu.f.f
    @unpack cfgu = i
    DE.set_ut!(intu, x0, t)
    function jacfunB(u::AbstractVector{<:Dual{<:Any, T, N}}) where {T,N}
        u2 = reinterpret(Dual{Tag{typeof(Bjac!)}, T, N}, u)
        set_u!(wrapped_dynamicsu, u2)
        intu.u     = x0 # https://github.com/SciML/OrdinaryDiffEq.jl/issues/1638
        intu.uprev = x0
        initialize!(intu, intu.cache)
        u_modified!(intu, true) # This is required
        DE.step!(intu, i.Ts, true)
        reinterpret(eltype(u), intu.u)
    end
    jacobian!(B, jacfunB, u0, cfgu)
end

ForwardDiff.:≺(::Type{ForwardDiff.Tag{typeof(Ajac!)}}, ::Type{<:ForwardDiff.Tag}) = true
ForwardDiff.:≺(::Type{ForwardDiff.Tag{typeof(Bjac!)}}, ::Type{<:ForwardDiff.Tag}) = true


