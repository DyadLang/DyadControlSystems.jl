mutable struct Stateful{C, X}
    controller::C
    x::X
end

"""
    Stateful(controller)
    Stateful(controller, x0)

Wrap a discrete-time controller with signature `x⁺, y = controller(x, u, p, t)` to make it stateful, i.e., remember its own state. The wrapped controller has the signature `y = controller(u, p, t)`, i.e., the state `x` is removed from both input arguments and return values.
"""
Stateful(controller) = Stateful(controller, default_initial_state(controller))

function (s::Stateful)(u, p, t)
    s.x, y = s.controller(s.x, u, p, t)
    y
end

abstract type AbstractSlidingModeController end


"""
    SuperTwistingSMC{K, S}
    SuperTwistingSMC(k, σ, Ts, k2=1.1)

A "SuperTwisting" Sliding Mode Controller. Create a standard (SISO) sliding-mode controller with switching function `s = σ(u)`, where `s` is the sliding variable computed by `σ`. The controller can be called using the syntax `x⁺, y smc(x, u, p, t)` where `x` is the _controller_ state, `u` the state of the system (the input to the controller), `p` the parameters of σ (if any), and `t` the time. The new controller state `x⁺` as well as the control signal `y` (the output of the controller) are returned.

# Fields:
- `k::K`: Gain
- `σ::S`: Switching function ``σ(u, p, t)``
- `Ts`: Sample time
- `k2::K`: Second tuning gain (defaults to 1.1 and does typically not need tuning).

# Example
```julia
function σ(x, p, t)
    q, qd = x
    e = q - qr(t)
    ė = qd - qdr(t)
    ė + e
end
smc_supertwist = SuperTwistingSMC(50, σ, 0.01) |> Stateful 
```
Wrap the controller in a [`Stateful`](@ref) to make it stateful, i.e., remember its own state.
"""
struct SuperTwistingSMC{K, S, F} <: AbstractSlidingModeController
    k::K
    σ::S
    Ts::F
    k2::K
    function SuperTwistingSMC(k, σ, Ts, k2=1.1)
        k, k2 = promote(k, k2)
        new{typeof(k), typeof(σ), typeof(Ts)}(k, σ, Ts, k2)
    end
end

default_initial_state(::SuperTwistingSMC{K}) where K = zero(K)

"""
    (smc::SuperTwistingSMC)(x, u, p, t)

Evaluate the sliding mode controller `smc` at the controller state `x`, input `u` (plant state), parameters `p`, and time `t`.

# Arguments:
- `x`: The state of the controller
- `u`: The input to the controller, this will be forwarded to the switching function ``σ(u, p, t)``. The controller input will in typical applications be the (estimated) state of the system to be controlled.
- `p`: The parameters of the controller, this will be forwarded to the switching function ``σ(u, p, t)``.
- `t`: Time.
"""
function (smc::SuperTwistingSMC)(x, u, p, t)
    # Reference: A QUICK INTRODUCTION TO SLIDING MODE CONTROL AND ITS APPLICATIONS
    (; σ, k, k2, Ts) = smc
    s = σ(u, p, t)
    y  = -√(k*abs(s))*sign(s)
    xd = -k2*k*sign(s)
    x += Ts*xd # Fwd Euler integration
    x, y + x
end

"""
    SlidingModeController{S, U}
    SlidingModeController(σ, u)

Create a standard (SISO) sliding-mode controller with switching function `σ` and control law `u(s, t)`, where `s` is the sliding variable computed by `σ`. The controller can be called using the syntax `x⁺, y smc(x, u, p, t)` where `x` is the controller state, `u` the state of the plant (the input to the controller), `p` the parameters of σ (if any), and `t` the time. The new controller state `x⁺` as well as the control signal `y` (the output of the controller) are returned.

# Example:
```julia
function σ(x, p, t)
    q, qd = x
    e = q - qr(t)
    ė = qd - qdr(t)
    ė + e
end
sat(x) = x/(abs(x) + 0.01) # smooth saturation function
smc_standard = SlidingModeController(σ, (s,t) -> -20*sign(s))   |> Stateful
smc_smooth1  = SlidingModeController(σ, (s,t) -> -20*tanh(10s)) |> Stateful
smc_smooth2  = SlidingModeController(σ, (s,t) -> -20*sat(s))    |> Stateful
```
Wrap the controller in [`Stateful`](@ref) to make it stateful, i.e., remember its own state. This version of SMC controller does not have a state, and it's thus safe to wrap it in `Stateful` by default.
"""
struct SlidingModeController{S, U} <: AbstractSlidingModeController
    σ::S
    u::U
end

default_initial_state(::SlidingModeController) = 0

function (smc::SlidingModeController)(x, u, p, t)
    σ, ufun = smc.σ, smc.u
    s = σ(u, p, t)
    y = ufun(s, t)
    0, y + x
end



"""
    LinearSlidingModeController(; k, ks, sys::AbstractStateSpace{<:Discrete}, err, desired_poles)

Create a linear sliding-mode controller with switching function of system `sys`.

This controller chooses a sliding surface of dimension `nx-nu` and a control law on the form
```math
u = -(GB)^{-1}(GAx - Gx + kT_s s + k_s T_s \\operatorname{sign}(s))
```
where `G` is chosen such that the poles of the reduced-order closed-loop system (of size `nx-nu`) are placed at `desired_poles`. The vector of discrete-time desired poles is thus to be of length `nx-nu`.

The controller can be called using the syntax `x⁺, y smc(x, u, p, t)` where `x` is the controller state, `u` the state of the plant (the input to the controller), `p` the parameters of σ (if any), and `t` the time. The new controller state `x⁺` as well as the control signal `y` (the output of the controller) are returned.

Wrap the controller in [`Stateful`](@ref) to make it stateful, i.e., remember its own state. This version of SMC controller does not have a state, and it's thus safe to wrap it in `Stateful` by default.

# Arguments:
- `k`: Proportional feedback gain on the sliding surface.
- `ks`: Discontinuous feedback gain on the sliding surface.
- `sys`: System model
- `err`: Error function `err(x, p, t) -> e`, typically `e = x(t) - xr(t)`. The default if no function is provided is `err(x, p, t) -> x`, i.e., for regulation around the origin.

# Extended help
The width of the quasi-sliding mode band is 2Δ = 2ks*Ts / (1-k*Ts)

In steady state, the trajectory will move within the small band given by
{ x | s(x) < ks*Ts }

Ref: "High Precision Motion Control Based on a Discrete-time Sliding Mode Approach"
"""
struct LinearSlidingModeController{K,ΛT,SYS,E,TT,SYSR} <: AbstractSlidingModeController
    k::K
    Λ::ΛT
    Ts::K
    ks::K
    sys::SYS
    err::E
    T::TT
    sysr::SYSR
end
function LinearSlidingModeController(; k, ks, sys::AbstractStateSpace{<:Discrete}, err = (x, p, t) -> x, desired_poles)
    Ts = sys.Ts
    k, ks = promote(k, ks, Ts)
    1 > k*Ts || throw(ArgumentError("k*Ts must be less than 1."))
    T = _get_T(sys.B)
    sys = similarity_transform(sys, T)
    (; nx, nu) = sys
    nx1 = nx-nu
    @assert norm(sys.B[1:nx1, :]) < 1e-10
    length(desired_poles) == nx1 || error("The number of desired poles should be nx-nu = $nx1")
    if any(abs(p) > 1 for p in desired_poles)
        @warn("Unstable discrete-time poles specified, transforming continuous-time poles to discrete-time poles using ZoH assumption.")
        desired_poles = @. exp(desired_poles*Ts)
    end
    sysr = ss(sys.A[1:nx1,1:nx1], sys.A[1:nx1,nx1+1:end], I, 0, sys.timeevol)
    K = place(sys.A[1:nx1,1:nx1], sys.A[1:nx1,nx1+1:end], desired_poles)

    Λ = [K I(nu)]
    LinearSlidingModeController(k, Λ, Ts, ks, sys, err, T, sysr)
end

default_initial_state(::LinearSlidingModeController) = 0

function (smc::LinearSlidingModeController)(x, u, p, t)
    (; sys, Λ, k, Ts, ks, T) = smc
    (; A, B) = sys # This system has been transformed by T

    err = T*smc.err(u, p, t) # We have to transform the error and the plant state u, these are still in the original coordinates
    s = (Λ*err)[]
    u = T*u
    y = if sys.nu == 1
        -(Λ*B)[]\((Λ*A*u - Λ*u)[] + k*Ts*s + ks*Ts*sign(s))
    else
        -(Λ*B)\(Λ*A*u - Λ*u + k*Ts*s + ks*Ts*sign(s))
    end
    0, y
end


function _get_T(B)
    Tl, _ = RobustAndOptimalControl._coordinatetransformqr(B)
    inv(Tl)
end