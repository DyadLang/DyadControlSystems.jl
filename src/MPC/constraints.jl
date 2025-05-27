abstract type AbstractMPCConstraints end
abstract type AbstractLinearMPCConstraints end

struct MPCConstraints{TU, TX} <: AbstractLinearMPCConstraints
    umin::TU
    umax::TU
    xmin::TX
    xmax::TX
    soft::Bool
end

"""
    LinearMPCConstraints(vmin, vmax, Cv, Dv, soft_indices)
    LinearMPCConstraints(; vmin, vmax, Cv, Dv, soft_indices)

Applicable to [`LinearMPCModel`](@ref), this structure allows you to constrain any linear combination of the states and inputs. The constrained output ``v_{min} ≤ v ≤ v_{max}`` is defined as
```math
v = C_v x + D_v u
```
- `soft_indices`: A vector of integer indices indicating which components of `v` are soft constraints. It's recommended to make components of ``v`` that depend on the state be soft constraints.
"""
Base.@kwdef struct LinearMPCConstraints{TX, M <: AbstractMatrix, VI <: AbstractVector{Int}} <: AbstractLinearMPCConstraints
    vmin::TX
    vmax::TX
    Cv::M
    Dv::M
    soft_indices::VI
end

"""
    MPCConstraints(; umin, umax, xmin = nothing, xmax = nothing, soft = true)

A structure representing the constraints of an MPC problem. See also [`LinearMPCConstraints`](@ref) for a more advanced interface to linear constraints.

# Arguments:
- `umin`: Lower bound for control signals.
- `umax`: Upper bound for control signals.
- `xmin`: Lower bound for constrained output signals.
- `xmax`: Upper bound for constrained output signals.
- `soft`: Indicate if constrained outputs are using soft constraints (recommended)
"""
function MPCConstraints(; umin, umax, xmin=nothing, xmax=nothing, soft=true)
    MPCConstraints(umin, umax, xmin, xmax, soft)
end

"""
    Cv, Dv, vmin, vmax, soft_indices = setup_output_constraints(nx, nu, constraints, op, v)

prepate linear constraints by calculating constraint jacobians and adjusting for operating point op.
"""
function setup_output_constraints(nx, nu, constraints::MPCConstraints, op, v)
    @unpack umin, umax, xmin, xmax, soft = constraints
    xor(xmin === nothing, xmax === nothing) && throw(ArgumentError("Both xmin and xmax must be supplied"))
    u_active = findall(isfinite.(umin) .| isfinite.(umax))
    if xmin === nothing
        x_active = 1:0 # Empty range
    else
        x_active = findall(isfinite.(xmin) .| isfinite.(xmax))
    end
    nxv, nuv = length(x_active), length(u_active)
    if xmin === nothing
        Cv = spzeros(nuv, nx) # No constrained outputs
    else
        if v == I
            Cv0 = _make_matrix(x_active, nx)
        else
            Cv0 = _make_matrix(v, nx)
        end
        if length(x_active) == size(Cv0, 1) 
            Cv = sparse([Cv0; zeros(nuv, nx)])
        else
            error("Number of output constraints does not match number of constrained plant outputs")
        end
    end
    Dv = sparse([zeros(nxv, nu); _make_matrix(u_active, nu)])
    soft_indices = soft ? (1:length(x_active)) : (1:0) # This is used to denote the soft constraints which require slack variables
    umin = umin[u_active]
    umax = umax[u_active]
    
    # Constraints are now only output constraints
    if xmin === nothing
        vmin = umin 
        vmax = umax
    else
        vmin = [xmin[x_active]; umin[u_active]] 
        vmax = [xmax[x_active]; umax[u_active]]
    end

    if op !== nothing
        big_op = vec(sum(Cv*op.x, dims=2) + sum(Dv*op.u, dims=2)) # The sum is added in case op is a scalar (OperatingPoint() produces an op with scalar zeros) 
        vmin = vmin .- big_op
        vmax = vmax .- big_op
    end

    Cv, Dv, vmin, vmax, soft_indices
end

function setup_output_constraints(nx, nu, constraints::BoundsConstraint, op, v)
    @unpack umin, umax, xmin, xmax, xNmin, xNmax, dumin, dumax = constraints
    xNmin == xmin || throw(ArgumentError("xNmin must equal xmin in Linear MPC"))
    xNmax == xmax || throw(ArgumentError("xNmax must equal xmax in Linear MPC"))
    all(isinf, dumin) || throw(ArgumentError("Linear MPC does not support dumin constraints (bounds on control-input change)"))
    all(isinf, dumax) || throw(ArgumentError("Linear MPC does not support dumax constraints (bounds on control-input change)"))
    all(!isfinite, xmin) && (xmin = nothing)
    all(!isfinite, xmax) && (xmax = nothing)
    lc = MPCConstraints(; umin, umax, xmin, xmax)
    setup_output_constraints(nx, nu, lc, op, v)
end

function setup_output_constraints(nx, nu, constraints::LinearMPCConstraints, op, v)
    @unpack vmin, vmax, Cv, Dv, soft_indices = constraints
    
    if op !== nothing
        big_op = vec(sum(Cv*op.x, dims=2) + sum(Dv*op.u, dims=2)) # The sum is added in case op is a scalar (OperatingPoint() produces an op with scalar zeros) 
        vmin = vmin .- big_op
        vmax = vmax .- big_op
    end

    Cv, Dv, vmin, vmax, soft_indices
end



## =============================================================

"""
    NonlinearMPCConstraints(fun, min, max, soft_indices)
    NonlinearMPCConstraints(; umin, umax, xmin=nothing, xmax=nothing)

A struct holding constraint information for nonlinear MPC.

If the signature `NonlinearMPCConstraints(; umin, umax, xmin=nothing, xmax=nothing)` is used, the `fun` and `soft_indices` are created automatically.

# Arguments:
- `fun`: A function `(x,u,p,t)->constrained_outputs`
- `min`: The lower bound for `constrained_outputs`
- `max`: The upper bound for `constrained_outputs`
- `soft_indices::Vector{Int}`: A vector of indices that indicates which `constrained_outputs` are soft constraints. Slack variables will be added for each soft constraint and this increases the computational complexity. It is recommended to use soft constraints for states and functions of the states, but typically not for intputs.
"""
struct NonlinearMPCConstraints{F, V<:AbstractVector, VI <: AbstractVector{Int}} <: AbstractMPCConstraints
    fun::F
    min::V
    max::V
    soft_indices::VI
end

function NonlinearMPCConstraints(; umin, umax, xmin=nothing, xmax=nothing, soft=true)
    lincon = MPCConstraints(; umin, umax, xmin, xmax, soft)
    if xmin === xmax === nothing
        nx = 1 # garbage value, we will not use Cv in this case
    else
        nx = length(xmax)
    end
    nu = length(umax)
    Cv, Dv, vmin, vmax, soft_indices = setup_output_constraints(nx, nu, lincon, OperatingPoint(), I(nx))

    if xmin === xmax === nothing
        constrained_outputs = (x,u,p,t)->Dv*u
    else
        constrained_outputs = (x,u,p,t)-> Cv*x + Dv*u # TODO: optimize
    end
    constraints = NonlinearMPCConstraints(constrained_outputs, vmin, vmax, soft_indices)
end

function is_feasible(c::NonlinearMPCConstraints, x, u, p=nothing, t=0)
    v = c.fun(x,u,p,t)
    all(eachindex(v)) do i
        c.min[i] ≤ v[i] ≤ c.max[i]
    end
end