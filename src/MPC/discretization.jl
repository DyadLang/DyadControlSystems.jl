abstract type DiscretizationConstraint <: AbstractTrajectoryConstraint end

const _disc_no_init = "Discretization structure has not been initialized, please provide N, Ts and scale_x upon creation if used in this way."

function Base.show(io::IO, disc::DiscretizationConstraint)
    println(io, Base.nameof(typeof(disc)))
    hasproperty(disc, :N) && println(io, "\tN: $(disc.N)")
    hasproperty(disc, :Ts) && println(io, "\tTs: $(disc.Ts)")
    hasproperty(disc, :scale_x) && println(io, "\tscale_x: $(disc.scale_x)")
    hasproperty(disc, :n_colloc) && println(io, "\tn_colloc: $(disc.n_colloc)")
    hasproperty(disc, :threads) && println(io, "\tthreads: $(disc.threads)")
end

function Base.getproperty(d::DiscretizationConstraint, s::Symbol)
    s ∈ fieldnames(typeof(d)) && return getfield(d, s)
    getproperty(getfield(d, :dyn), s)
end


"""
    MultipleShooting{F <: FunctionSystem}

Multiple-shooting dynamics constraint. This discretization method assumes zero-order-hold inputs and only supports ODE dynamics. 

# Fields:
- `dyn::F`: The discrete-time dynamics
- `scale_x`: Numerical scaling of state variables
- `threads::Bool = false`: Use threaded evaluation of the dynamics. For small dynamics, the overhead of threads is too large to be worthwhile.
"""
Base.@kwdef struct MultipleShooting{F <: FunctionSystem{<:Discrete}, S} <: DiscretizationConstraint
    dyn::F
    N::Int
    scale_x::S = ones(dyn.nx)
    threads::Bool = false
end


"""
    MultipleShooting(dyn, threads = false; scale_x)

Multiple-shooting dynamics constraint. This discretization method assumes zero-order-hold inputs and only supports ODE dynamics.
For DAE dynamics, or to use first-order-hold inputs, see [`Trapezoidal`](@ref) or [`CollocationFinE`](@ref).

# Arguments:
- `dyn::F`: The discrete-time dynamics
- `scale_x`: Numerical scaling of state variables
- `threads::Bool = false`: Use threaded evaluation of the dynamics. For small dynamics, the overhead of threads is too large to be worthwhile.
"""
function MultipleShooting(dyn, threads=false; N=-1, kwargs...)
    dyn isa FunctionSystem{<:Discrete} || throw(ArgumentError("Dynamics must be discrete-time"))
    MultipleShooting(; dyn, N, threads, kwargs...)
end

"""
    Variables{MultipleShooting}(x::AbstractMatrix{T}, u::AbstractMatrix{T})

Constructor of variables for MultipleShooting discretization.
"""
function Variables(x::AbstractMatrix{T}, u::AbstractMatrix{T}, disc::D, n_robust) where {D <: MultipleShooting, T}
    N = _check_valid_traj_sizes(x, u)
    nx = size(x, 1)
    nu = size(u, 1)
    #D === MultipleShooting || error("Discretization $D not yet supported")
    vars = [repeat(vec(x), n_robust); repeat(vec(u), n_robust)]
    Variables{D, T}(vars, nx, nu, N, disc, n_robust)
end

all_x_indices(::MultipleShooting, N, nx, nu, n_robust, ri=1) = (1:(N+1)*nx) .+ (ri-1)*(N+1)*nx
all_u_indices(::MultipleShooting, N, nx, nu, n_robust, ri=1) = (1:N*nu) .+ (n_robust*(N+1)*nx + (ri-1)*(N*nu))

variable_factor(::Type{<:MultipleShooting}) = 1
Base.length(c::MultipleShooting) = c.N*c.dyn.nx

horizon(vars::Variables) = vars.N

function get_bounds(c::DiscretizationConstraint)
    c = zeros(length(c))
    c, c
end

function evaluate!(c, ms::MultipleShooting{D}, oi::ObjectiveInput, p0, t) where D
    p = get_system_parameters(p0)
    dyn = ms.dyn
    N = horizon(oi)
    N > 0 || error(_disc_no_init)
    nx = dyn.nx
    nu = dyn.nu
    scale_x = ms.scale_x
    x, u = oi.x, oi.u
    Ts::Float64 = Float64(dyn.Ts)
    if ms.threads
        let nx=nx, nu=nu, x=x, u=u, N=N, dyn=dyn, c=c, p=p, t=t
            @views Threads.@threads for i = 1:N
                c[(1:nx) .+ (i-1)*nx] .= (x[:, i+1] .- dyn(x[:, i], u[:, i], p, t + i*Ts)) ./ scale_x
            end
        end
    else
        cinds = 1:nx
        @views for i = 1:N
            c[cinds] .= (x[:, i+1] .- dyn(x[:, i], u[:, i], p, t + i*Ts)) ./ scale_x
            cinds = cinds .+ nx
        end
    end
    c
end


## =============================================================================
"""
    Trapezoidal{is_dae, F <: FunctionSystem}

Trapezoidal integration dynamics constraint. This discretization method assumes first-order hold control inputs.
"""
struct Trapezoidal{DAE, F <: FunctionSystem{Continuous}, S} <: DiscretizationConstraint
    dyn::F
    N::Int
    Ts::Float64
    scale_x::S
    threads::Bool
end

Trapezoidal(dyn, N, Ts, scale_x, threads) = Trapezoidal(; dyn, N, Ts, scale_x, threads)


"""
    Trapezoidal(; dyn, Ts, scale_x = ones(dyn.nx), threads = false)

Trapezoidal integration dynamics constraint. This discretization method assumes first-order hold control inputs and supports ODE or DAE dynamics.

# Arguments:
- `dyn::F`: The continuous-time dynamics
- `scale_x`: Numerical scaling of state variables
- `threads::Bool = false`: Use threaded evaluation of the dynamics. For small dynamics, the overhead of threads is too large to be worthwhile.
"""
function Trapezoidal(; dyn, N=-1, Ts=-1.0, scale_x = ones(dyn.nx), threads = false)
    inner_dyn = dyn.dynamics
    is_dae = (inner_dyn isa ODEFunction) && inner_dyn.mass_matrix !== nothing && inner_dyn.mass_matrix != I
    Trapezoidal{is_dae, typeof(dyn), typeof(scale_x)}(dyn, N, Ts, scale_x, threads)
end

Trapezoidal(dyn, threads=false; kwargs...) = Trapezoidal(; dyn, threads, kwargs...)

"""
    Variables{Trapezoidal}(x::AbstractMatrix{T}, u::AbstractMatrix{T})

Constructor of variables for Trapezoidal discretization.
"""
function Variables(x::AbstractMatrix{T}, u::AbstractMatrix{T}, disc::D, n_robust) where {D <: Trapezoidal, T}
    N = _check_valid_traj_sizes(x, u)
    nx = size(x, 1)
    nu = size(u, 1)
    #D === Trapezoidal || error("Discretization $D not yet supported")
    vars = [repeat(vec(x), n_robust); repeat(vec(u), n_robust)]
    Variables{D, T}(vars, nx, nu, N, disc, n_robust)
end

all_x_indices(::Trapezoidal, N, nx, nu, n_robust, ri=1) = (1:(N+1)*nx) .+ (ri-1)*(N+1)*nx
all_u_indices(::Trapezoidal, N, nx, nu, n_robust, ri=1) = (1:N*nu) .+ (n_robust*(N+1)*nx + (ri-1)*(N*nu))

variable_factor(::Type{<:Trapezoidal}) = 1
Base.length(c::Trapezoidal) = c.N*c.dyn.nx

# ODE
function evaluate!(c, trapz::Trapezoidal{false}, oi::ObjectiveInput, p0, t)
    p = get_system_parameters(p0)
    dyn = trapz.dyn
    N = horizon(oi)
    Ts = (0.5 * trapz.Ts)::Float64
    Ts > 0 || error(_disc_no_init)
    N > 0 || error(_disc_no_init)
    nx = dyn.nx
    nu = dyn.nu
    scale_x = trapz.scale_x
    x, u = oi.x, oi.u
    cinds = 1:nx
    f0 = dyn(x[:, 1], u[:, 1], p, t)
    @views for i = 1:N
        f1 = dyn(x[:, i+1], u[:, min(i+1, N)], p, t + i*Ts)
        c[cinds] .= (x[:, i] .+ Ts .* (f1 .+ f0) .- x[:, i+1] ) ./ scale_x
        cinds = cinds .+ nx
        f0 = f1
    end
    c
end

# DAE
function evaluate!(c, trapz::Trapezoidal{true}, oi::ObjectiveInput, p0, t)
    p = get_system_parameters(p0)
    dyn = trapz.dyn
    N = horizon(oi)
    Ts = (0.5 * trapz.Ts)::Float64
    Ts > 0 || error(_disc_no_init)
    N > 0 || error(_disc_no_init)
    scale_x = trapz.scale_x
    x, u = oi.x, oi.u
    @unpack nx, nu, na = dyn
    ns = nx-na
    f0 = dyn(x[:, 1], u[:, 1], p, t)
    @views for i = 1:N
        inds_diff   = (1:ns) .+ (i-1)*nx
        inds_alg    = (ns+1:nx) .+ (i-1)*nx
        f1 = dyn(x[:, i+1], u[:, min(i+1, N)], p, t + i*Ts)
        c[inds_diff] .= (x[1:ns, i] .+ Ts .* (f1[1:ns] .+ f0[1:ns]) .- x[1:ns, i+1] ) ./ scale_x[1:ns]
        for j = 1:na
            c[inds_alg[j]] = f1[ns+j] / scale_x[ns+j]
        end

        f0 = f1
    end
    c
end

## =============================================================================

"""
    CollocationFinE{is_dae, F <: FunctionSystem}

Orhogonal Collocation on Finite Elements dynamics constraint. This discretization method can use zero or first-order hold control inputs.

# Fields:
- `dyn::F`: The continuous-time dynamics
- `Ts`: Sample time
- `threads::Bool = false`: Use threaded evaluation of the dynamics. For small dynamics, the overhead of threads is too large to be worthwhile.
- `hold_order::Int = 0` : Order of hold for control inputs. 0 for zero-order hold (piecewise constant inputs), 1 for first-order hold (piecewise affine inputs).
""" 
Base.@kwdef struct CollocationFinE{DAE, F <: FunctionSystem, DAT, S} <: DiscretizationConstraint
    dyn::F
    N::Int = -1
    Ts::Float64 = -1
    Der_A::DAT
    lfc::Vector{Float64}
    taupoints::Vector{Float64}
    u_cache::Vector{Float64}
    threads::Bool = false
    n_colloc::Int = 5
    roots_c::String = "Legendre"
    scale_x::S = ones(dyn.nx)
    hold_order::Int = 0
end

"""
    Variables{CollocationFinE}(x::AbstractMatrix{T}, u::AbstractMatrix{T})

Constructor of variables for CollocationFinE discretization.
"""
function Variables(x::AbstractMatrix{T}, u::AbstractMatrix{T}, disc::D, n_robust) where {D <: CollocationFinE, T}
    #N = _check_valid_traj_sizes(x, u)
    N = size(u, 2)
    nx = size(x, 1)
    nu = size(u, 1)
    vars = [repeat(vec(x), n_robust); repeat(vec(u), n_robust)]
    Variables{D, T}(vars, nx, nu, N, disc, n_robust)
end


CollocationFinE(dyn, N::Int, Ts, Der_A, lfc, taupoints, u_cache, threads, n_colloc, roots_c, scale_x, hold_order) = CollocationFinE(dyn, threads; N, n_colloc, roots_c, scale_x, Ts, hold_order)

"""
    CollocationFinE(dyn, threads=false; n_colloc = 5, roots_c = "Legendre", scale_x=ones(dyn.nx), hold_order = 0)

Orhogonal Collocation on Finite Elements dynamics constraint. This discretization method can use zero or first-order hold control inputs and supports ODE or DAE dynamics.

# Arguments:
- `dyn::F`: The continuous-time dynamics
- `threads::Bool = false`: Use threaded evaluation of the dynamics. For small dynamics, the overhead of threads is too large to be worthwhile.
- `n_colloc::Int = 5`: Number of collocation points per finite element
- `roots_c::String = "Legendre"`: Type of collocation points. Currently supports "Legendre" and "Radau" points.
- `hold_order::Int = 0` : Order of hold for control inputs. 0 for zero-order hold (piecewise constant inputs), 1 for first-order hold (piecewise affine inputs).
"""
function CollocationFinE(dyn, threads=false; N = -1, n_colloc = 5, roots_c = "Legendre", scale_x=ones(dyn.nx), Ts=-1, hold_order = 0)
    1 <= n_colloc <= 9 || error("Currently only supporting degree < 9")
    # Location of 4th Order Legendre roots on the time scale
    # b_total = [0,0.06943184420297369,0.3300094782075724,0.6699905217924279,0.9305681557970239] # These may be needed later to send the correct t into dynamics function

    function legendre_point_vector(N)
        legendre_points =[[0,0.500000],
                       [0,0.21132486540518713, 0.7886751345948129],
                       [0,0.11270166537925824,0.500000,0.8872983346207419],
                       [0,0.06943184420297369,0.3300094782075724,0.6699905217924279,0.9305681557970239],
                       [0,0.04691007703066803,0.23076534494715872,0.500000,0.7692346550528459,0.9530899229693315],
                       [0.0,0.033765242898424,0.16939530676686707,0.38069040695841083,0.6193095930415667,0.8306046932331836,0.9662347571015469],
                       [0.0,0.025446043828620736,0.1292344072003027,0.2970774243113074,0.49999999999995565,0.702922575688846,0.8707655927994734,0.9745539561714928],
                       [0.0,0.019855071751231912,0.10166676129318651,0.23723379504183797,0.40828267875218943,0.5917173212477325,0.7627662049583206,0.8983332387066989,0.9801449282488013],
                       [0.0,0.01591988024618698,0.08198444633668228,0.19331428364970127,0.3378732882981136,0.4999999999999494,0.6621267117018799,0.8066857163506065,0.9180155536629273,0.984080119753947]]
        return legendre_points[N]
        # TODO: Equivalent to [0; (FastGaussQuadrature.gausslegendre(N) .+ 1) ./ 2)]
    end

    function radau_point_vector(N)
        radau_points =[[0,1.000000],
                       [0,0.333333,1.000000],
                       [0.0, 0.15505102572168217, 0.6449489742783178, 1.0],
                       [0.0, 0.08858795951270398, 0.4094668644407347, 0.787659461760847, 1.0],
                       [0.0, 0.05710419611451767, 0.2768430136381238, 0.5835904323689168, 0.8602401356562195, 1.0],
                       [0.0, 0.03980985705146878, 0.19801341787360816, 0.43797481024738616, 0.695464273353636, 0.9014649142011736, 1.0],
                       [0.0, 0.029316427159784886, 0.1480785996684843, 0.3369846902811543, 0.5586715187715501, 0.7692338620300545, 0.9269456713197411, 1.0],
                       [0.0, 0.0224793864387125, 0.11467905316090421, 0.26578982278458946, 0.45284637366944464, 0.6473752828868303, 0.8197593082631076, 0.9437374394630779, 1.0],
                       [0.0, 0.017779915147363434, 0.09132360789979393, 0.21430847939563075, 0.3719321645832723, 0.5451866848034267, 0.7131752428555695, 0.8556337429578544, 0.9553660447100302, 1.0]]
        return radau_points[N]
    end
    
    deg       = n_colloc - 1
    if roots_c == "Radau"
        taupoints = radau_point_vector(deg)
    elseif roots_c == "Legendre"
        taupoints = legendre_point_vector(deg)
    else
        error("Currently only supporting roots_c = \"Legendre\" or roots_c = \"Radau\"")
    end

    
    # Continuity vector
    lfc = vec(zeros(n_colloc,1))
    for k = 1:n_colloc
        lfc[k]= 1;
       for i = 1:n_colloc
           if i != k
               lfc[k] *= (1.0 - taupoints[i])/(taupoints[k] - taupoints[i]) 
           end
       end
   end

    # Generate Derivative matrix
    function derivative_Lagrange(j,x,z)
        y = 0
        n = length(x)
            for l=1:n
                if l!=j
                    k = 1/(x[j]-x[l]);
                    for m=1:n
                        if m!=j && m!=l
                            k = k*(z-x[m])/(x[j]-x[m]);
                        end
                    end
                    y = y + k;
                end
            end
        return y
    end

    Der_Ac = zeros(n_colloc, deg)
    for i = 1:deg+1
        for j = 2:deg+1
            Der_Ac[i,j-1] = derivative_Lagrange(i,taupoints,taupoints[j])
        end
    end
    Der_A = SMatrix{n_colloc,deg}(Der_Ac./Ts)

    u_cache = vec(zeros(dyn.nu))

    inner_dyn = dyn.dynamics
    is_dae = (inner_dyn isa ODEFunction) && inner_dyn.mass_matrix !== nothing && inner_dyn.mass_matrix != I
    CollocationFinE{is_dae, typeof(dyn), typeof(Der_A), typeof(scale_x)}(dyn, N, Ts, Der_A, lfc, taupoints, u_cache, threads, n_colloc, roots_c, scale_x, hold_order)
end

Base.length(c::CollocationFinE) = (c.N*c.n_colloc)*c.dyn.nx

all_x_indices(disc::CollocationFinE, N, nx, nu, n_robust, ri=1) = (1:(N*disc.n_colloc+1)*nx) .+ (ri-1)*(N*disc.n_colloc+1)*nx
all_u_indices(disc::CollocationFinE, N, nx, nu, n_robust, ri=1) = (1:N*nu) .+ (n_robust*(N*disc.n_colloc+1)*nx + (ri-1)*N*nu)


function naive_but_nonallocating_mul!(C, A, B)
    @inbounds @fastmath for m ∈ axes(A,1), n ∈ axes(B,2)
        Cmn = zero(eltype(C))
        for k ∈ axes(A,2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end

naive_but_nonallocating_mul!(C::AbstractMatrix{Any}, A, B) = mul!(C,A,B)


# ODE
@views function evaluate!(cv, cfe::CollocationFinE{false}, oi::ObjectiveInput, p0, t)
    p = get_system_parameters(p0)
    @unpack dyn, Ts, N, lfc, n_colloc, Der_A, taupoints, u_cache, scale_x, hold_order = cfe
    @unpack nx, nu = dyn
    Ts > 0 || error(_disc_no_init)
    N > 0 || error(_disc_no_init)
    x, u = oi.x, oi.u

    n_c = n_colloc
    for i = 1:N # TODO: make this loop optionally threaded
        colinds_x   = (1:n_c)        .+ (i-1)*n_c
        colinds_c   = (1:n_c-1)      .+ (i-1)*n_c
        inds_c      = (1:nx)         .+ (i-1)*n_c*nx
        all_inds_c  = (1:(n_c-1)*nx) .+ (i-1)*n_c*nx
        naive_but_nonallocating_mul!(reshape(cv[all_inds_c], nx, n_c-1), x[:,colinds_x], Der_A) # cv now contains ẋ
        # mul!(c[:,colinds_c], x[:,colinds_x], Der_A)
        
        for k in eachindex(colinds_c)
            if hold_order == 1 && i > 1
                u_cache    .= u[:,i-1] .+ (u[:,i].-u[:,i-1]).*taupoints[mod(i-1,n_c)+1]
                cv[inds_c] .-= dyn(x[:,colinds_c[k]+1],u_cache, p, t + i*Ts) # cv now contains ẋ - f(x,u)
            else    
                cv[inds_c] .-= dyn(x[:,colinds_c[k]+1],u[:,i], p, t + i*Ts) 
            end
            inds_c = inds_c .+ nx
        end

        # Continuity conditions
        next_x = colinds_x[end] + 1
        mul!(cv[inds_c], x[:, colinds_x], lfc)
        cv[inds_c] .-= x[:, next_x]
    end
    inds = 1:nx
    for i = 1:length(cv) ÷ length(scale_x)
        cv[inds] ./= scale_x
        inds = inds .+ nx
    end
    nothing
end

# DAE
@views function evaluate!(cv, cfe::CollocationFinE{true}, oi::ObjectiveInput, p0, t)
    p = get_system_parameters(p0)
    @unpack dyn, Ts, N, lfc, n_colloc, Der_A, taupoints, u_cache, scale_x, hold_order = cfe
    @unpack nx, nu, na = dyn
    Ts > 0 || error(_disc_no_init)
    N > 0 || error(_disc_no_init)
    x, u = oi.x, oi.u
    n_c = n_colloc
    for i = 1:N # TODO: make this loop optionally threaded
        colinds_x   = (1:n_c)        .+ (i-1)*n_c
        # colinds_c   = (1:n_c-1)      .+ (i-1)*n_c
        inds_c      = (1:nx-na)   .+ (i-1)*n_c*nx
        inds_alg    = (nx-na+1:nx).+ (i-1)*n_c*nx
        all_inds_c  = (1:(n_c-1)*nx) .+ (i-1)*n_c*nx
        inds_ode    = (1:nx-na)
        naive_but_nonallocating_mul!(reshape(cv[all_inds_c], nx, n_c-1), x[:,colinds_x], Der_A)
        # mul!(c[:,colinds_c], x[:,colinds_x], Der_A)
        
        for k in eachindex(colinds_x)
            if hold_order == 1 & i > 1
                u_cache      .= u[:,i-1] .+ (u[:,i].-u[:,i-1]).*taupoints[mod(i,n_c)]
                temp_dyn      = dyn(x[:,colinds_x[k]],u_cache, p, t + i*Ts) 
            else    
                temp_dyn      = dyn(x[:,colinds_x[k]],u[:,i], p, t + i*Ts)
            end
            # temp_dyn      = dyn(x[:,colinds_x[k]],u[:,i], p, t + i*Ts)
            if k > 1
                cv[inds_c]  .-= temp_dyn[inds_ode]
                inds_c        = inds_c .+ nx
            end
            cv[inds_alg] .=  temp_dyn[nx-na+1:end]
            inds_alg      = inds_alg .+ nx
        end

         # Continuity conditions
         next_x = colinds_x[end] + 1
         mul!(cv[inds_c], x[inds_ode,colinds_x], lfc)
         cv[inds_c] .-= x[inds_ode, next_x]  
    end
    inds = 1:nx
    for i = 1:length(cv) ÷ length(scale_x)
        cv[inds] ./= scale_x
        inds = inds .+ nx
    end
    nothing
end


##
## Iterators ===================================================================
struct SCI{D, T <: ObjectiveInput{<:Any,<:Any, D}}
    oi::T
end

function Base.iterate(sci::SCI{<:Union{<:MultipleShooting, <:Trapezoidal}}, i=1)
    oi = sci.oi
    i > horizon(oi) && return nothing
    StageInput(  
                get_timeindex(oi.x, i, Val(static_nx(oi))),
                get_timeindex(oi.u, i, Val(static_nu(oi))),
                get_timeindex(oi.r, i),
                i,
    ), i+1
end

function Base.iterate(sci::SCI{<:CollocationFinE}, i=1)
    oi = sci.oi
    N = horizon(oi)
    i > N && return nothing
    nc = size(oi.x, 2) == N+1 ? 1 : sci.oi.discretization.n_colloc # To handle the MPC history which does not have collocation vars
    StageInput(  
                get_timeindex(oi.x, nc*(i-1)+1, Val(static_nx(oi))),
                get_timeindex(oi.u, i, Val(static_nu(oi))),
                get_timeindex(oi.r, i),
                i,
    ), i+1
end

"""
    stage_inputs(oi::ObjectiveInput)

Iterate over the trajectories in `oi` in the form of [`StageInput`](@ref) objects.
"""
stage_inputs(oi::ObjectiveInput) = SCI(oi)

# using ResumableFunctions
# @resumable function x_indices(N::Int, nx::Int, nu::Int)::UnitRange{Int}
#     inds = 1:nx
#     n_tot = nx + nu
#     for i = 1:N+1
#         @yield inds
#         inds = inds .+ n_tot
#     end
# end

# @resumable function u_indices(N::Int, nx::Int, nu::Int)::UnitRange{Int}
#     inds = 1:nu
#     n_tot = nx + nu
#     for i = 1:N
#         @yield inds
#         inds = inds .+ n_tot
#     end
# end

# High precision Collocation Root Finder
#using FastGaussQuadrature
#function collocation_rootfinder(deg::Int,root_c::String)
#   if deg < 1
#       error("Root finder only works for positive degree")
#   end
#    if root_c == "Legendre"
#        if deg == 1
#            taupoints = [0.0;0.5]
#        else
#            α = 0.0
#            β = 0.0
#            taupoints = [0.0;0.5.*(FastGaussQuadrature.gaussjacobi(deg,α,β)[1].+1)]
#        end
#    elseif root_c == "Radau"
#        if deg == 1
#            taupoints = [0.0;1.0]
#        else
#            α = 1.0
#            β = 0.0
#            taupoints = [0.0;0.5.*(FastGaussQuadrature.gaussjacobi(deg-1,α,β)[1].+1);1.0]
#        end
#    else
#        error("roots supported for Legendre or Radau Quadratures only")
#    end
#    return taupoints
#end