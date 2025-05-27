using LowLevelParticleFilters
using LowLevelParticleFilters: AbstractFilter, state, dynamics, predict!, correct!
using UnPack


"""
    OperatingPoint(x, u, y)
    OperatingPoint()

Structure representing an operating point around which a system is linearized.
If no arguments are supplied, an empty operating point is created. 

# Arguments:
- `x`: State
- `u`: Control input
- `y`: Output
"""
struct OperatingPoint{X,U,Y}
    x::X
    u::U
    y::Y
end
OperatingPoint() = OperatingPoint(0,0,0)
linearize(f, op::OperatingPoint, p, t) = linearize(f, op.x, op.u, p, t)


mpc_observer_predict!(o, u, r, p, t) = LowLevelParticleFilters.predict!(o, u, p, t)
mpc_observer_correct!(o, u, y, r, p, t) = LowLevelParticleFilters.correct!(o, u, y, p, t)

function reset_observer!(sys, x0)
    LowLevelParticleFilters.reset!(sys)
    state(sys) .= x0
end

"""
    StateFeedback{F <: Function, x} <: AbstractFilter
    StateFeedback(discrete_dynamics, x0, nu, ny)
    StateFeedback(sys::FunctionSystem, x0)

An observer that uses the dynamics model without any measurement feedback. This observer can be used as an oracle that has full knowledge of the state. Note, this is often an unrealistic assumption in real-world contexts and open-loop observers can not account for load disturbances. Use of this observer in a closed-loop context creates a false closed loop. 
"""
struct StateFeedback{F, X} <: AbstractFilter
    dyn::F
    x::X
    nu::Int
    ny::Int
    StateFeedback(dyn, x, nu, ny) = new{typeof(dyn), typeof(x)}(dyn, copy(x), nu, ny)
end

function StateFeedback(sys::FunctionSystem{TE}, x) where TE
    TE <: Discrete || throw(ArgumentError("Observers for MPC must use discrete-time dynamics. See https://help.juliahub.com/DyadControlSystemss/dev/mpc/#Discretization for more info."))
    size(x, 1) == sys.nx || throw(ArgumentError("Initial state must have length sys.nx."))
    StateFeedback(sys.dynamics, x, sys.nu, sys.ny)
end

Base.@deprecate OpenLoopObserver StateFeedback

LowLevelParticleFilters.reset!(observer::StateFeedback) = nothing
LowLevelParticleFilters.state(observer::StateFeedback) = observer.x
LowLevelParticleFilters.dynamics(observer::StateFeedback) = observer.dyn
LowLevelParticleFilters.sample_measurement(observer::StateFeedback, x, u, t) = x
LowLevelParticleFilters.measurement(observer::StateFeedback) = (x,u,p,t) -> x

"""
    x = predict!(observer::AbstractFilter, u, p, t)

Predict the state forward in time, updates the internal state of the observer.
"""
function LowLevelParticleFilters.predict!(observer::StateFeedback, u, p, t)
    dyn = dynamics(observer)
    x = state(observer)
    xp = @views dyn(x, u[:, 1], p, t)
    x .= xp
end

"""
    x = correct!(observer::AbstractFilter, u, y, p, t)

Correct the state estimate using a measurement `y`, updates the internal state of the observer.
"""
function LowLevelParticleFilters.correct!(observer::StateFeedback, u, y, p, t)
    0,0 # The oracle doesn't need any measurement, it's all knowing and always right
end


function Base.getproperty(o::StateFeedback, s::Symbol)
    s ∈ fieldnames(typeof(o)) && return getfield(o, s)
    if s === :nx
        return length(o.x)
    else
        throw(ArgumentError("StateFeedback has no property named $s"))
    end
end



## Fixed-gain observer
"""
    FixedGainObserver{F <: Function, x} <: AbstractFilter
    FixedGainObserver(sys::AbstractStateSpace, x0, K)

A linear observer, similar to a Kalman filer, but with a fixed measurement feedback gain.
The gain can be designed using, e.g., pole placement or solving a Riccati equation.
For a robust observer, consider using `glover_mcfarlane` followed by [`inverse_lqr`](@ref).
"""
struct FixedGainObserver{F, X, G} <: AbstractFilter
    sys::F 
    x::X
    K::G
    FixedGainObserver(sys, x, K) = new{typeof(sys), typeof(x), typeof(K)}(sys, copy(x), copy(K))
end

LowLevelParticleFilters.reset!(observer::FixedGainObserver) = nothing
LowLevelParticleFilters.state(observer::FixedGainObserver) = observer.x
LowLevelParticleFilters.dynamics(observer::FixedGainObserver) = observer.sys
function LowLevelParticleFilters.sample_measurement(observer::FixedGainObserver, x, u, p, t)
    yh = observer.sys.C*x + observer.sys.D*u
    @. yh += 0.5 * abs(yh) * rand()
end
LowLevelParticleFilters.measurement(observer::FixedGainObserver) = (x,u,p,t) -> observer.sys.C*x + observer.sys.D*u


function LowLevelParticleFilters.predict!(observer::FixedGainObserver, u, p, t)
    sys = dynamics(observer)
    x = state(observer)
    x .= @views sys.A*x .+ sys.B*u[:, 1]
end


function LowLevelParticleFilters.correct!(observer::FixedGainObserver, u, y, p, t)
    sys = dynamics(observer)
    x = state(observer)
    yh = sys.C*x
    if !iszero(sys.D)
        yh .+= sys.D*u[:, 1]
    end
    e = y - yh
    x .+= observer.K*e
    0,e
end


function Base.getproperty(o::FixedGainObserver, s::Symbol)
    s ∈ fieldnames(typeof(o)) && return getfield(o, s)
    if s === :nx
        return length(o.x)
    elseif s === :dyn
        return o.sys
    else
        throw(ArgumentError("FixedGainObserver has no property named $s"))
    end
end


## DiscreteSystems
"""
    Observer(o::AbstractFilter; name, ny = o.ny, nu = o.nu, nx = o.nx)

Create `DiscreteSystem` representing a state observer.

# Arguments:
- `o`: Any `AbstractFilter` available from DyadControlSystems or LowLevelParticleFilters
- `name`: A symbol indicating the name of the system
- `ny`: Number of measured signals
- `nu`: Number of control inputs
- `nx`: Number of states
"""
function Observer(o::AbstractFilter; name,
    ny = o.ny, # allow the user to provide those for observers that do not store this information
    nu = o.nu,
    nx = o.nx)
    # @variables x[1:nx](t)=0 u[1:nu](t)=0 [input=true] y[1:ny](t)=0 [output=true] 
    @variables u_u(t)[1:nu]=0 [input=true] u_y(t)[1:ny]=0 [input=true] y_x(t)[1:nx]=0 [output=true] 
    @parameters p
    y_x = collect(y_x)
    u_u = collect(u_u)
    u_y = collect(u_y)
    s = update!(o, u_u, u_y, p, t)
    eqs = y_x .~ s
    DiscreteSystem(eqs, t, [u_y; u_u; y_x], [], name=name)
end



## Operating point adjustment for nonlinear observers with linear prediction models
"""
    OperatingPointWrapper{F <: Function, x} <: AbstractFilter
    OperatingPointWrapper(sys::AbstractStateSpace, x0, K)

An OperatingPointWrapper contains an observer and an [`OperatingPoint`](@ref) and serves to translate between absolute coordinates and Δ-coordinates. This is useful in scenarios where a nonlinear observer that operates in absolute coordinates is used together with a linear prediction model, which operates in Δ-coordinates.
"""
struct OperatingPointWrapper{O,OP} <: AbstractFilter
    observer::O
    op::OP
end

LowLevelParticleFilters.reset!(o::OperatingPointWrapper) = LowLevelParticleFilters.reset!(o.observer)
# NOTE: the definition below may cause problems if the function `state` is called in LLPF, introduce DyadControlSystems adjusted_state()?
LowLevelParticleFilters.state(o::OperatingPointWrapper) = o.observer.x - o.op.x
# function LowLevelParticleFilters.dynamics(o::OperatingPointWrapper)
#     (x,u,p,t) -> dynamics(o.observer)(x+o.op.x,u+o.op.u,p,t) - o.op.x
# end
# LowLevelParticleFilters.measurement(o::OperatingPointWrapper) = (x,u,p,t) -> o.observer.measurement(x+o.op.x,u+o.op.u,p,t) - o.op.y

function LowLevelParticleFilters.sample_measurement(o::OperatingPointWrapper, args...)
    sample_measurement(o.observer) #.- o.op.y
end

function LowLevelParticleFilters.predict!(o::OperatingPointWrapper, u, p, t)
    predict!(o.observer, u .+ o.op.u, p, t) .- o.op.x
end


function LowLevelParticleFilters.correct!(o::OperatingPointWrapper, u, y, p, t)
    correct!(o.observer, u.+o.op.u, y.+o.op.y, p, t)
end


function Base.getproperty(o::OperatingPointWrapper, s::Symbol)
    s ∈ fieldnames(typeof(o)) && return getfield(o, s)
    getproperty(getfield(o, :observer), s)
end

## Convenience constructors
"""
    UnscentedKalmanFilter(sys::FunctionSystem, R1, R2, d0=MvNormal(Matrix(R1)); p = SciMLBase.NullParameters())

Convencience constructor for systems of type [`FunctionSystem`](@ref).
"""
function LowLevelParticleFilters.UnscentedKalmanFilter(sys::FunctionSystem, args...; kwargs...)
    UnscentedKalmanFilter{false,false,false,false}(
        sys.dynamics,
        sys.measurement,
        args...;
        sys.nu,
        sys.ny,
        kwargs...
    )
end
