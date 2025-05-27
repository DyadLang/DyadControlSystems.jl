#=
Equation references in this file are to "H-Infinity Loop-Shaped Model Predictive Control with Heat Pump Application"
Bortoff, Schwerdtner, Danielson and Di Cairano, 2019
=#
##

import LowLevelParticleFilters: parameters


function _make_matrix(z::AbstractMatrix, nx::Int)
    size(z,2) == nx || throw(ArgumentError("inconsistent sizes, expected a matrix where the number of columns is `nx`"))
    z
end

"""
    _make_matrix(z, nx::Int)

If `z` is a matrix, checks the dimensions and returns `z`
If `z` is a vector of integers, construct a matrix `C` such that `C*x` selects components `z` from `x`.
`nx` is the expected column size of the matrix.
"""
function _make_matrix(z::AbstractVector{<:Integer}, nx::Int)
    maximum(z) <= nx && minimum(z) >= 1 || throw(ArgumentError("invalid index vector $z, valid indices are 1:$nx"))
    nz = length(z)
    [i == z[j] for j = 1:nz, i = 1:nx]
end

"""
    _make_matrix_t(soft_inds::AbstractVector{<:Integer}, nv::Int, nx::Int)

return a sparse nx×nz boolean matrix (nz = length(soft_inds)).

# Arguments:
- `soft_inds`: Indicates constraints that need slack variables
- `nv`: Total num constraints
- `nx`: num states
"""
function _make_matrix_t(soft_inds::AbstractVector{<:Integer}, nv::Int, nx::Int)
    isempty(soft_inds) && return zeros(nv, 0)
    maximum(soft_inds) <= nv && minimum(soft_inds) >= 1 || throw(ArgumentError("invalid index vector $soft_inds, valid indices are 1:$nv"))
    nz = length(soft_inds)
    [i == soft_inds[j] for i = 1:nv, j = 1:nz]
    # S = spzeros(nv, nx)
    # for i in eachindex(z)
    #     S[i, z[i]] = 1
    # end
    # S
end

"""
    hasintegrator(x::AbstractStateSpace)

Determine whether or not a linear system has an integrator by checking the real value 
of the slowest real pole. The intended usage of this function is to determine if weight functions `W1,W2` have integral action.
"""
function hasintegrator(G::AbstractStateSpace)
    G.nx == 0 && return false
    any(poles(G)) do p
        abs(imag(p)) < 1e-6 && real(p) > -1e-3
    end
end
hasintegrator(x) = false




## LinearPredictionModel =======================================================
abstract type LinearPredictionModel{TE <: ControlSystemsBase.Discrete} <: AbstractStateSpace{TE} end

issystem(::LinearPredictionModel) = true
struct LinearMPCModel{TE <: ControlSystemsBase.Discrete, S <: LTISystem, O <: AbstractFilter, OP <: OperatingPoint,CzT,CvT,DvT,CT} <: LinearPredictionModel{TE}
    sys::S
    Cz::CzT
    Cv::CvT
    Dv::DvT
    observer::O
    timeevol::TE
    strictly_proper::Bool
    op::OP
    vmin
    vmax
    soft_indices
    constraints::CT
end

function Base.getproperty(sys::LinearMPCModel, s::Symbol)
    s ∈ fieldnames(typeof(sys))     && return getfield(sys, s)
    s ∈ propertynames(getfield(sys, :sys)) && return getproperty(getfield(sys, :sys), s)
    s === :nx                       && return size(sys.A, 1)
    s === :ny                       && return size(sys.C, 1)
    s === :nu                       && return size(sys.B, 2)
    s === :nz                       && return size(getfield(sys, :Cz), 1)
    s === :nv                       && return size(getfield(sys, :Cv), 1)
    s === :w2sinds                  && return 1:0
    s === :xsinds                   && return 1:0
    s === :w1inds                   && return 1:0
    s === :Ts                       && return sys.timeevol.Ts
    s === :W1_integrator            && return false
    s === :nxw1                     && return 0
    s === :nxw2                     && return 0
    throw(ArgumentError("$(typeof(sys)) has no property named $s"))
end

Base.propertynames(f::LinearMPCModel, private::Bool = false) = (fieldnames(typeof(f))..., :C, :D, :nx, :nu, :ny, :nz, :nv, :nxw1, :nxw2, :w2sinds, :xsinds, :w1inds, :Gs, :info, :Ts)

function parameters(f)
    hasproperty(f, :p) ? getproperty(f, :p) : []
end

for f in [:state_names, :output_names, :input_names]
    @eval $f(sys::LinearMPCModel{<:Any, <:NamedStateSpace}) = $f(sys.sys)
end


"""
    LinearMPCModel(G, observer; constraints::AbstractLinearMPCConstraints, op::OperatingPoint = OperatingPoint(), strictly_proper = false, z = I(G.nx), x0)

A model structure for use with linear MPC controllers. This structure serves as both a prediction model and an observer.

# Arguments:
- `G`: A linear system created using, e.g., the function [`ss`](@ref).
- `observer`: Any supported observer object, such as a `KalmanFilter`.
- `constraints`: An instance of [`MPCConstraints`](@ref) or [`LinearMPCConstrints`](@ref)
- `op`: An instance of [`OperatingPoint`](@ref)
- `strictly_proper`: Indicate whether or not the MPC controller is to be considered a strictly proper system, i.e., if there is a one sample delay before a measurement has an effect on the control signal. This is typically required if the computational time of the MPC controller is close to the sample time of the system. 
- `z`: Either a vector of state indices indicating controlled variables, or a matrix `nz × nx` that multiplies the state vector to yield the controlled variables.
- `v`: Either a vector of state indices indicating constrained outputs, or a matrix `nv × nx` that multiplies the state vector to yield the constrained outputs. This option has no effect if [`LinearMPCConstraints`](@ref) are used. 
- `x0`: The initial state of the internal observer.
"""
function LinearMPCModel(G, observer; constraints::AbstractLinearMPCConstraints, op::OperatingPoint = OperatingPoint(zeros(G.nx), zeros(G.nu), zeros(G.ny)), strictly_proper=false, z = I(G.nx), v = I(G.nx), x0)
    
    Cz = _make_matrix(z, G.nx)
    Cv, Dv, vmin, vmax, soft_indices = setup_output_constraints(G.nx, G.nu, constraints, op, v)

    state(observer) .= x0 .- (ndims(op.x) == 2 ? op.x[:,1] : op.x)
    LinearMPCModel(G, Cz, Cv, Dv, observer, G.timeevol, strictly_proper, op, vmin, vmax, soft_indices, constraints)
end

LowLevelParticleFilters.state(observer::LinearMPCModel) = state(observer.observer)

mpc_observer_predict!(observer::LinearMPCModel, args...) = mpc_observer_predict!(observer.observer, args...)

@views function mpc_observer_correct!(sys::LinearMPCModel, u, y, r, args...)
    @unpack sys, op, observer = sys
    x = state(observer)
    ys = y .- (ndims(op.y) == 2 ? op.y[:,1] : op.y)
    mpc_observer_correct!(observer, u, ys, r, args...)
end

function reset_observer!(sys::LinearMPCModel, x0::AbstractVector)
    x = sys.op.x
    @views state(sys.observer) .= x0 .- (ndims(x) == 2 ? x[:,1] : x)
end

# Special handling required if the observer is a wrapper
function reset_observer!(o::LinearMPCModel{<:ControlSystemsBase.Discrete, <:ControlSystemsBase.LTISystem, W}, x0::AbstractVector) where W <: OperatingPointWrapper
    state(o.observer.observer) .= x0
end

## RobustMPCModel ==============================================================
struct RobustMPCModel{TE <: ControlSystemsBase.Discrete, X, HT, KT, OP <: OperatingPoint, TSYS, TA, TB} <: LinearPredictionModel{TE}
    sys::TSYS
    A::TA
    B::TB
    Br
    Cz
    Cv
    Dv
    H::HT
    K::KT
    timeevol::TE
    gmf
    Q1
    Q2
    W1_integrator::Bool
    W1
    x::X
    strictly_proper::Bool
    op::OP
    vmin
    vmax
    soft_indices
end

function Base.getproperty(sys::RobustMPCModel, s::Symbol)
    s ∈ fieldnames(typeof(sys)) && return getfield(sys, s)
    if s === :C
        return getfield(sys, :Cz)
    elseif s === :D
        return zeros(sys.ny, sys.nu)
    elseif s === :nx
        return size(getfield(sys, :A), 1)
    elseif s === :nu
        return size(getfield(sys, :B), 2)
    elseif s === :ny
        return size(getfield(sys, :Cz), 1)
    elseif s === :nv
        return size(getfield(sys, :Cv), 1)
    elseif s === :nz
        return size(getfield(sys, :Cz), 1)
    elseif s === :nxw1
        W1 = sys.W1
        return W1 isa AbstractStateSpace ? W1.nx : 0
    elseif s === :nxw2
        W2 = sys.gmf[3].W2
        return W2 isa AbstractStateSpace ? W2.nx : 0
    elseif s === :w2sinds
        return 1:sys.nxw2
    elseif s === :xsinds
        return 1:sys.Gs.nx
    elseif s === :w1inds
        (1:sys.nxw1) .+ sys.Gs.nx
    elseif s === :xqinds
        (1:sys.sys.nx) .+ (sys.Gs.nx + sys.nxw1)
    elseif s === :Gs
        return sys.gmf[3].Gs
    elseif s === :info
        return sys.gmf[3]
    elseif s === :W2
        return sys.gmf[3].W2
    elseif s === :Ts
        return sys.timeevol.Ts
    else
        throw(ArgumentError("$(typeof(sys)) has no property named $s"))
    end
end

Base.propertynames(f::RobustMPCModel, private::Bool = false) = (fieldnames(typeof(f))..., :C, :D, :nx, :nu, :ny, :nv, :nxw1, :nxw2, :w2sinds, :xsinds, :w1inds, :xqinds, :Gs, :info, :Ts)




"""
    RobustMPCModel(G; W1, W2 = I(G.ny), constraints::AbstractLinearMPCConstraints, x0, strictly_proper = true, op::OperatingPoint = OperatingPoint(), K)

A model structure for use with linear MPC controllers. This structure serves as both a prediction model and an observer.
Internally, the Glover-McFarlane method is used to find a robustly stabilizing controller for the shaped plant \$G_s = W_2 G W_1\$, see [`glover_mcfarlane`](@ref) and examples in the documentation for additional details.

Note, this model has automatically generated penalty matrices `Q1, Q2` built in, and there is thus no need to supply them to the constructor of [`LQMPCProblem`](@ref).

# Arguments:
- `G`: A linear system, created using, e.g., the function [`ss`](@ref).
- `W1`: A precompensator for loop shaping. Set `W1` to a LTI controller with an integrator to achieve integral action in the MPC controller.
- `W2`: A post compensator for loop shaping.
- `K`: Is an observer gain matrix of size `(G.nx, G.ny)`, or an observer object for the plant `G`, i.e., a `KalmanFilter`.
- `constraints`: An instace of [`MPCConstraints`](@ref) or [`LinearMPCConstrints`](@ref)
- `x0`: The initial state.
- `strictly_proper`: Indicate whether or not the MPC controller is to be considered a strictly proper system, i.e., if there is a one sample delay before a measurement has an effect on the control signal. This is typically required if the computational time of the MPC controller is close to the sample time of the system. 
- `op`: An instance of [`OperatingPoint`](@ref).
- `v`: Either a vector of state indices indicating constrained outputs, or a matrix `nv × nx` that multiplies the state vector to yield the constrained outputs. This option has no effect if [`LinearMPCConstraints`](@ref) are used. 
"""
function RobustMPCModel(G; W1, W2=I(G.ny), K, constraints::AbstractLinearMPCConstraints, x0, strictly_proper=true, op::OperatingPoint = OperatingPoint(), v=I(G.nx))
    # see reference at top of file for equations
    W1 isa Number && (W1 = ss(W1*I(G.nu), G.timeevol))
    W1 isa AbstractMatrix && (W1 = ss(W1, G.timeevol))
    W2 isa Number && (W2 = ss(W2*I(G.ny), G.timeevol))
    W2 isa AbstractMatrix && (W2 = ss(W2, G.timeevol))
    W1 isa TransferFunction && (W1 = ss(W1))
    W2 isa TransferFunction && (W2 = ss(W2))
    gmf = _, γ, info = glover_mcfarlane(G; W1, W2, strictly_proper)

    K = get_observer_gain(K)

    γ > 4 && @warn "The Glover-McFarlane tuning resulted in a large γ of $γ which implies poor robustness (γ > 4). Consider modifying the loop-shaping weights W1, W2. For more help, see `?glover_mcfarlane`"

    Gs = info.Gs

    W1_integrator = W1 isa AbstractStateSpace && hasintegrator(W1)
    if W1_integrator
        hasintegrator(W2) && error("both W1 and W2 have a large static gain, could not determine a suitable realization for the observer.")
    end

    method = GMF(gmf)
    Q1,Q2 = inverse_lqr(method)
    Q2 = Symmetric(Matrix(1.0*Q2)) 
    
    # H∞ Observer gain
    H  = -info.Hkf # We need negative feedback

    A,B,C,D = ssdata(G)
    As,Bs,Cs,Ds = ssdata(Gs)
    Aw,Bw,Cw,Dw = ssdata(W1)
    
    # @unpack Cv, Dv, soft_indices = constraints
    Cv, Dv, vmin, vmax, soft_indices = setup_output_constraints(G.nx, G.nu, constraints, op, v)

    # TODO: Implement simplified model for known special cases
    # TODO: Augment with disturbance observer here, possibly accept covariance matrices or a δ tuning parameter to make the interface nicer

    nw = W1.nx
    n = G.nx
    ns = Gs.nx
    Ap = [
        As zeros(ns, nw+n)
        zeros(nw, ns)  Aw zeros(nw, n)
        zeros(n, ns) B*Cw A
    ] # eq 23a
    Bp = [Bs; Bw; B*Dw]
    Cz2 = [I(ns) zeros(ns, nw+n)]           # eq 23b
    Cv2 = [zeros(size(Dv, 1), ns) Dv*Cw Cv] # Eq 23c

    Dv2 = Dv*Dw # The Dv*Dw is not included in the paper but is required in case W1 is a P/PI controller, i.e., has a nonzero D-term. 
    nr = size(C, 1) # TODO: called nz elsewhere, this was changed in MPC2 and may need more consideration
    Br = zeros(size(Ap, 1), nr)
    if hasintegrator(W2)
        Br[1:W2.nx, :] = -W2.B # reference subtracted from the input to weight W2, fig 2 and eq 12a
    end

    nxw1 = W1 isa AbstractStateSpace ? W1.nx : 0
    x0 = [
        zeros(W2.nx)
        x0 .- op.x # x part of xs
        zeros(nxw1) # W1 in xs
        zeros(nxw1) # W1 in xw
        x0 .- op.x
    ]
    @assert length(x0) == size(Ap, 1)
    # op = OperatingPoint(SVector([zeros(nw); op.x; zeros(2nw); op.x]...), SVector(op.u...), SVector(op.y...)) # extend op.x with added states
    RobustMPCModel(G, Ap, Bp, Br, Cz2, Cv2, Dv2, H, K, G.timeevol, gmf, Q1, Q2, W1_integrator, W1, copy(x0), strictly_proper, op, vmin, vmax, soft_indices)
end

get_observer_gain(K::AbstractArray) = K
get_observer_gain(K::LowLevelParticleFilters.AbstractKalmanFilter) = ControlSystemsBase.lqr(ControlSystemsBase.Discrete, Matrix(K.A'), Matrix(K.C'), Matrix(K.R1), Matrix(K.R2))'
function get_observer_gain(K)
    hasproperty(K, :K) || error("The observer of type $(nameof(typeof(K))) provided to RobustMPCModel is not recognized.")
    return K.K
end

# Observer interface
# NOTE: The observer realization depends on what strategy is used for W1 and W2
function LowLevelParticleFilters.state(observer::LinearPredictionModel)
    observer.x
end
LowLevelParticleFilters.dynamics(observer::LinearPredictionModel) = observer

function reset_observer!(sys::RobustMPCModel, x0)
    @unpack op, W2, nxw1, nx = sys
    # nx = (sys.nx - 2nxw1 - W2.nx) ÷ 2
    opx = op.x#[(1:nx) .+ W2.nx]
    state(sys) .= [
        zeros(W2.nx)
        x0 .- opx # x part of xs
        zeros(2nxw1) # W1 in xs
                     # W1 in xw
        x0 .- opx  # G 
                    # TODO: q
    ]

end

function mpc_observer_predict!(observer::RobustMPCModel, u, r, p, t)
    # see reference at top of file for equations
    x = state(observer)
    # We do not subtract the operating point in this case since the raw u returned by the optimizer is already in Δ-coordinates
    # u = u .- observer.op.u
    if observer.W1_integrator # Eqs 10 a-c
        # For this realization, the reference enters the measurement equation so nothing extra to do
        x .= @views observer.A*x .+ observer.B*u[:, 1]
    else # Eqs 11-12
        error("mpc_observer_predict! does not yet support designs with W2 integrators")
        # Note, not sure if to include r here or just when computing the control signal in get_u")
        ri = r[:,1] .- observer.op.y
        x .= @views observer.A*x .+ observer.B*u[:, 1] .+ observer.Br*ri
    end
end

@views function mpc_observer_correct!(observer::RobustMPCModel, u, y, r, p, t)
    # see reference at top of file for equations
    @unpack gmf, sys, Cz, W1_integrator, op, H, K = observer
    x = state(observer)
    Gs = gmf[3].Gs

    ns = size(Cz, 1) # length of xs

    # ==========================================
    # State components
    # xs computed by H∞ observer
    # xw computed by the weight
    # x  computed by standard observer
    # ==========================================
    
    # Adjust signals to deviations around operating point
    if W1_integrator
        # Eq 10a,  Reference enters on the measurement
        length(y) == length(r) || error("y and r must have the same length when using the RobustMPCModel")
        ys = y .- r # Both y and r should be adjusted with op.y, but since ys is the difference between them, the adjustment is not needed
    else
        # Eq 11a
        # While it looks like y is supposed to be filtered through W2 to get shaped output ys, we handle the reference in the predict step through the Br term (eq 12a and below)
        # We thus only adjust for the operating point.
        ys = y .- op.y
    end
    
    ## Update shaped plant state
    ysh = Gs.C*x[1:ns]
    if !iszero(Gs.D)
        u = u #.- op.u # u already in Δ coordinates
        ysh .+= Gs.D*u[:, 1]
    end
    es = ys - ysh
    x[1:ns] .+= H*es
    

    ## Update non-shaped plant state
    xqinds = observer.xqinds
    yxq = sys.C*x[xqinds]
    if !iszero(sys.D)
        u = u #.- op.u # u already in Δ coordinates
        yxq .+= sys.D*u[:, 1]
    end
    exq = ys - yxq
    x[xqinds] .+= K*exq
    

    0,es
end
