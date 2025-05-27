using ForwardDiff
using OSQP
using SparseArrays
using LinearAlgebra, Statistics
using RecipesBase
import ControlSystems
using ControlSystems: AbstractStateSpace, ssdata, Discrete
using SeeToDee
using UnPack
import CommonSolve: solve
using TimerOutputs
const to = TimerOutput()
using LowLevelParticleFilters: measurement, sample_measurement, parameters

"""
    abstract type AbstractMPCSolver

An MPCSolver defines the following interface
- `workspace = init_solver_workspace(solver; P, q, A, l, u)` where the arrays are defined according to https://osqp.org/docs/solver/index.html
- `update_constraints!(solver, l, u)`
- `update_constraints!(solver, A)`
- `optimize!(solver; verbose = true)`
"""
abstract type AbstractMPCSolver end

"""
    rms(x)

Root-mean square error of `x`. If `x` is a matrix, the rms value will be calculated along rows, i.e.,
```
sqrt.(mean(abs2, x, dims = 2))
```
"""
rms(x::AbstractVector) = sqrt(mean(abs2, x))
mse(x::AbstractVector) = sse(x) / length(x)
rms(x::AbstractMatrix) = sqrt.(mean(abs2, x, dims = 2))[:]
oneifzero(x) = x == 0 ? 1 : x

"""
    modelfit(y, yh)

Calculate model-fit percentage (normalized RMS error)
```math
100 \\dfrac{1-rms(y - yh)}{rms(y - mean(y))}
```
"""
modelfit(y, yh) = 100 * (1 .- rms(y .- yh) ./ oneifzero.(rms(y .- mean(y, dims = 2))))
modelfit(y::T, yh::T) where {T<:AbstractVector} =
    100 * (1 .- rms(y .- yh) ./ oneifzero.(rms(y .- mean(y))))


"""
    lqr_cost(x, u, Q1, Q2, Q3 = nothing)

Calculate the LQR cost for trajectory `x,u`.
If `Q3 > 0`, the initial control value is assumed to be zero and the first control value will be penalized by `dot(u[:,1]-0, Q3, u[:,1]-0)`.
"""
function lqr_cost(x,u,Q1,Q2,Q3=nothing)
    c = dot(x,Q1,x) + dot(u,Q2,u)
    if Q3 !== nothing
        Δu = diff(u, dims=2)
        c += dot(Δu, Q3, Δu) + @views(dot(u[:,1], Q3, u[:,1]))
    end
    c
end


speye(N) = spdiagm(ones(N))

## =============================================================================
include("problem_constructors_nonlinear.jl")
## =============================================================================


lqr_cost(x,u,prob::AbstractMPCProblem) = lqr_cost(x,u,prob.Q1,prob.Q2,prob.Q3)

rk4(args...; kwargs...) = SeeToDee.Rk4(args...; kwargs...)

function fwdeuler(f::F, Ts0; supersample::Integer = 1) where {F}
    supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
    # Runge-Kutta 4 method
    Ts = Ts0 / supersample # to preserve type stability in case Ts0 is an integer
    let Ts = Ts
        function (x, u, p, t)
            T = typeof(x)
            f1 = f(x, u, p, t)
            add = Ts * f1
            for i in 2:supersample
                f1 = f(y, u, p, t)
                add = Ts * f1
                # This gymnastics with changing the name to y is to ensure type stability when x + add is not the same type as x. The compiler is smart enough to figure out the type of y
                y += add
            end
            return y
        end
    end
end

"""
    f_discrete, l_discrete = rk4(f, l, Ts)

Discretize dynamics `f` and loss function `l`using RK4 with sample time `Ts`. 
The returned function is on the form `(xₖ,uₖ,p,t)-> (xₖ₊₁, loss)`.
Both `f` and `l` take the arguments `(x, u, p, t)`.
"""
function rk4(f::F, l::LT, Ts) where {F, LT}
    # Runge-Kutta 4 method
    function (x, u, p, t)
        f1, L1 = f(x, u, p, t), l(x, u, p, t)
        f2, L2 = f(x + Ts / 2 * f1, u, p, t + Ts / 2), l(x + Ts / 2 * f1, u, p, t + Ts / 2)
        f3, L3 = f(x + Ts / 2 * f2, u, p, t + Ts / 2), l(x + Ts / 2 * f2, u, p, t + Ts / 2)
        f4, L4 = f(x + Ts * f3, u, p, t + Ts), l(x + Ts * f3, u, p, t + Ts)
        x += Ts / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        L  = Ts / 6 * (L1 + 2 * L2 + 2 * L3 + L4)
        return x, L
    end
end

"""
    x, u = rollout(f, x0::AbstractVector, u, p, t=1)

Simulate discrete system `f` from initial condition `x0` and input array `u`.
"""
function rollout(f, x0::AbstractVector, u, p=parameters(f), t=1)
    T = promote_type(eltype(x0), eltype(u))
    x = zeros(T, length(x0), size(u, 2)+1)
    x[:, 1] .= x0
    rollout!(f, x, u, p, t)
end

"""
x, u = rollout!(f, x, u, p, t=1; Ts)

Simulate discrete system `f` from initial condition `x[:, 1]` and input array `u`.
Modifies `x,u`.
- `Ts` denotes the sample time, if `f` is a `FunctionSystem`, `Ts` defaults to `f.Ts`. If `Ts` is not provided and `f` is not a `FunctionSystem`, `Ts` defaults to 1.
"""
function rollout!(f, x, u, p=parameters(f), t=1; Ts = nothing)
    f isa FunctionSystem{Continuous} && error("Cannot perform rollout with continuous-time dynamics. Discretize the dynamics prior to calling rollout, see, e.g., `rk4` or `MPCIntegrator`.")
    @boundscheck size(u,2) == size(x, 2)-1 || throw(ArgumentError("x should be 1 element longer (in second dimension) than u."))
    if Ts === nothing
        Ts = hasproperty(f, :Ts) ? f.Ts : 1.0
    end
    @views for i = 1:size(x, 2)-1
        @inbounds x[:, i+1] = f(x[:, i], u[:, i], p, (i-1)*Ts+t)
    end
    x, u
end

# The code below can reduce allocations ever so slightly but doesn't improve runtime performance
# function rollout!(::Val{NX}, ::Val{NU}, f::F, x::AbstractMatrix{T}, u, p=nothing) where {NX,NU,T,F}
#     @boundscheck size(u,2) == size(x, 2)-1 || throw(ArgumentError("x should be 1 element longer (in second dimension) than u."))
#     @boundscheck size(u,1) == NU && size(x,1) == NX || error("Static sizes don't match dynamic sizes")
#     X = reinterpret(SVector{NX, T}, vec(x))
#     U = reinterpret(SVector{NU, T}, vec(u))
#     rollout!(f, X, U, p)
#     x, u
# end

# function rollout!(f::F, x::AbstractVector{<:SVector}, u::AbstractVector{<:SVector}, p=nothing) where F
#     for i = 1:length(x)-1
#         @inbounds x[i+1] = f(x[i], u[i], p, i) # TODO: i * Ts like above
#     end
#     x, u
# end

function rollout!(sys::AbstractStateSpace, x, u, p=parameters(sys), t=1)
    A,B = ssdata(sys)
    @views for i = 1:size(x, 2)-1
        mul!(x[:, i+1], A, x[:, i])
        mul!(x[:, i+1], B, u[:, i], 1, 1)
    end
    x, u
end

"""
    rollout!(prob::AbstractMPCProblem, p=parameters(prob))

Convenince method for `rollout!(prob.dyn, prob.x, prob.u)`.
"""
rollout!(prob::AbstractMPCProblem, p=parameters(prob), t=1) = rollout!(prob.dynamics, prob.x, prob.u, p, t)

"""
    lqr_rollout!(prob, x0::AbstractVector; kwargs...)

Trajectory rollout from initial state `x0`
"""
function lqr_rollout!(prob, x0::AbstractVector; kwargs...)
    prob.x[:, 1] .= x0
    lqr_rollout!(prob; kwargs...)
end

"""
    lqr_rollout!(prob, x = prob.x, u = prob.u, xr = prob.xr; update_L = true)

Trajectory rollout using an LQR controller.

# Arguments:
- `xr`: Reference state. May be a vector or a matrix with time in second dimension.
- `update_L`: if false, the state feedback is calculated using an infinite horizon cost function and the dynamics is linearized around the final reference state.
"""
function lqr_rollout!(prob::QMPCProblem, x::AbstractMatrix=prob.x, u=prob.u, xr=prob.xr; update_L = true)
    ref(xr::AbstractVector, i) = xr
    ref(xr::AbstractMatrix, i) = @views xr[:, i]
    f = prob.dynamics
    if !update_L
        xri = xr[:, end]
        QN, A, B = calc_QN_AB(prob, xri)
        L = (B'*QN*B + R)\(B'QN*A)
    end
    @views for i = 1:size(x, 2)-1
        xri = ref(xr, i)
        if update_L
            QN, A, B = calc_QN_AB(prob, xri)
            L = (B'*QN*B + prob.Q2)\(B'QN*A)
        end
        u[:, i] = L*(xri - x[:, i])
        x[:, i+1] = f(x[:, i], u[:, i], nothing, i)
    end
    x, u
end



"""
    update_constraints!(prob::QMPCProblem, x, u, p=parameters(prob); update_dynamics = true)

Update the problem and internal solver with new constraint vectors. Constraints are used to set the initial state as well as lower and upper bounds on states and control inputs. Currently, only the initial state is updated by this function.

If `update_dynamics = true`, the constraint matrix `A` will also be updated using new linearized dynamics.
"""
function update_constraints!(prob::QMPCProblem, x, u, x0, p=parameters(prob); update_dynamics=true)
    # Update initial state
    # x0 = @views x[:, 1]
    nx, Nx = size(x)
    @assert Nx == prob.N + 1
    @unpack N, nu, ns, A, xr = prob
    nx == prob.nx || throw(ArgumentError("Inconsistent size of x"))
    @. prob.lb[1:nx] = -x0
    @. prob.ub[1:nx] = -x0
    update_constraints!(prob.solver, prob.lb, prob.ub)

    update_dynamics || return 

    # Update dynamics
    f = prob.dynamics
    # size Ax = (N+1) * n × (N+1) * n 
    # size Bu = (N+1) * n × N * m
    # size Aeq = (N+1) * n × (N+1) * n + N * m
    Aeq = @view A[1:(N+1)*nx, :]
    Ad = zeros(nx, nx)
    Bd = zeros(nx, nu)

    Adinds = CartesianIndices((nx,nx)) .+ CartesianIndex(nx, 0) # first constraint is x0, then comes dynamics constraints 
    Bdinds = CartesianIndices((nx,nu)) .+ CartesianIndex(nx, nx)
    @timeit to "fill A" @views for i = 1:N
        @views xi = x[:, i]
        @views ui = u[:, i]

        Ajac!(Ad, f, xi, ui, p, i; chunk = prob.chunkA)
        Bjac!(Bd, f, xi, ui, p, i; chunk = prob.chunkB)

        Aeq[Adinds] .= Ad
        Aeq[Bdinds] .= Bd

        Adinds = Adinds .+ CartesianIndex(nx, nx+nu+2ns)
        Bdinds = Bdinds .+ CartesianIndex(nx, nx+nu+2ns)
    end
    @timeit to "update A" update_constraints!(prob.solver, A) # The filling of A takes about 20% of the time compared to this update
end

function update_constraints_sqp!(prob::QMPCProblem, x, u, x0, s, p=parameters(prob); update_dynamics=true)
    f = prob.dynamics
    # Update initial state
    nx, Nx = size(x)
    @assert Nx == prob.N + 1
    @unpack A, N, nu, ns, nv, A, xr, constraints = prob
    n_tot = (nx + nu + 2ns)
    c = constraints
    # x0 = @views x[:, 1]
    @. prob.lb[1:nx] = x[:,1] - x0
    @. prob.ub[1:nx] = x[:,1] - x0

    slack = prob.ns > 0

    # evaluate constraint function and put on rhs
    inds = (1:nx) .+ nx # start at x[2]
    @views for i = 1:N
        prob.lb[inds] .= x[:,i+1] .- f(x[:,i], u[:,i], p, i) # constraint violation
        prob.ub[inds] .= prob.lb[inds]
        inds = inds .+ nx
    end

    # Update output inequality constraints to map to Δ variables
    n_eq_constraints = (N+1)*nx # Equality constraints for dynamics only
    inds = (1:nv) .+ n_eq_constraints
    # sinds1 = (1:ns) .+ (nx + nu)
    # sinds2 = (1:ns) .+ (nx + nu + ns)
    @views for i = 1:N
        vi = c.fun(x[:,i], u[:,i], p, i) # Constrained output
        # s1,s2 are first and second slacks
        prob.ub[inds] = c.max - vi
        prob.lb[inds .+ nv] = c.min - vi
        # Add slack variables. Not all constraints have slacks, only the soft constraints
        (ns > 0) && for (si, j) in enumerate(c.soft_indices)
            s1 = s[1:ns, i] # s has 2ns rows, corresponding to upper and lower bound constraints
            s2 = s[ns+1:end, i]
            prob.ub[inds[j]] += s1[si] # Signs are reverese for upper and lower bounds
            prob.lb[inds[j] + nv] -= s2[si]
        end
        inds = inds .+ 2nv # bounds are repeated twice to handle slack vars
    end
    @assert inds[1]-1 == length(prob.ub) - 2ns*N

    @timeit to "update l u" update_constraints!(prob.solver, prob.lb, prob.ub)

    # update_dynamics || return 

    # Update dynamics
    # size Ax = (N+1) * n × (N+1) * n 
    # size Bu = (N+1) * n × N * m
    # size Aeq = (N+1) * n × (N+1) * n + N * m
    Aeq = @view A[1:n_eq_constraints, :]
    Ad = zeros(nx, nx)
    Bd = zeros(nx, nu)

    Adinds = CartesianIndices((nx,nx)) .+ CartesianIndex(nx, 0) # first constraint is x0, then comes dynamics constraints 
    Bdinds = CartesianIndices((nx,nu)) .+ CartesianIndex(nx, nx)
    @timeit to "fill A" @views for i = 1:N
        @views xi = x[:, i]
        @views ui = u[:, i]

        Ajac!(Ad, f, xi, ui, p, i; chunk = prob.chunkA)
        Bjac!(Bd, f, xi, ui, p, i; chunk = prob.chunkB)

        # TODO: Support update Cv and Dv

        Aeq[Adinds] .= Ad
        Aeq[Bdinds] .= Bd

        Adinds = Adinds .+ CartesianIndex(nx, n_tot)
        Bdinds = Bdinds .+ CartesianIndex(nx, n_tot)
    end
    @timeit to "update A" update_constraints!(prob.solver, A) # The filling of A takes about 20% of the time compared to this update
end


function Ajac!(A::AbstractMatrix, f, xi::AbstractVector, ui::Union{Number, AbstractVector}, p, t, args...; chunk, kwargs...)
    jacA = x -> f(x, ui, p, t)
    cfgA = ForwardDiff.JacobianConfig(jacA, xi, chunk)
    ForwardDiff.jacobian!(A, jacA, xi, cfgA, Val{false}())
end

function Bjac!(B::AbstractMatrix, f, xi::AbstractVector, ui::Union{Number, AbstractVector}, p, t, args...; chunk, kwargs...)
    jacB = u -> f(xi, u, p, t)
    cfgB = ForwardDiff.JacobianConfig(jacB, ui, chunk)
    ForwardDiff.jacobian!(B, jacB, ui, cfgB, Val{false}())
end

calc_QN_AB(prob, args...) = calc_QN_AB(prob.Q1, prob.Q2, prob.Q3, prob.dynamics, args...)


"""
    QN, A0, B0 = calc_QN_AB(Q1, Q2, Q3, dyn, xr, ur, p = nothing)
"""
function calc_QN_AB(Q1, Q2, Q3, dyn, xr, ur = 1e-16*ones(size(Q2,1)), p=nothing) # TODO: do not accept p=nothing
    # TODO: make sure ur is properly sent into this function
    nx, nu = size(Q1, 1), size(Q2, 1)
    A0, B0 = linearize(dyn, 1e-16*ones(nx)+xr[:,end], ur[:,end], p, 0) # TODO: compute sparsity pattern with SymbolicIR?
    S = try
        if Q3 === nothing || iszero(Q3)
            K, V = safe_lqr(Discrete, A0, B0, Matrix(Q1), Matrix(Q2))
            V
            # ControlSystemsBase.are(Discrete, A0, B0, Matrix(Q1), Matrix(Q2))
        else
            dare3(A0, B0, Matrix(Q1 + 1e-13I), Matrix(Q2), Matrix(Q3))
        end
    catch e
        @error "Failed to solve Riccati equation with arguments" A0 B0 Q1 Q2 Q3 xr ur
        rethrow()
    end
    QN = sparse(S)
    QN, A0, B0
end


function update_xr!(prob::QMPCProblem, xr, ur, u0; force=false)
    Q1, Q2, Q3 = prob.Q1, prob.Q2, prob.Q3
    N = prob.N
    if !force && prob.xr === xr
        return
    end
    prob.xr .= xr
    QN, A0, B0 = calc_QN_AB(prob, xr)
    prob.QN .= QN
    update_q!(prob.q, Q1, Q3, QN, N, xr, ur, u0)
    update_q!(prob.solver, prob.q)
    update_hessian!(prob, prob.QN)
end

function update_xr_sqp!(prob::QMPCProblem, xr, ur, xlin, ulin, p=nothing; force=false)
    Q1, Q2, Q3 = prob.Q1, prob.Q2, prob.Q3
    N = prob.N
    prob.xr .= xr
    prob.ur .= ur
    @views xri = size(xr, 2) == 1 ? xr : xr[:,prob.N+1]
    @views uri = size(ur, 2) == 1 ? ur : ur[:,prob.N]
    # QN, A0, B0 = calc_QN_AB(prob, xri - xlin[:,prob.N+1], uri - ulin[:,prob.N])
    QN, A0, B0 = calc_QN_AB(prob, xlin[:,prob.N+1], ulin[:,prob.N], p)
    prob.QN .= QN
    update_q_sqp!(prob.q, Q1, Q2, Q3, QN, N, xr, ur, xlin, ulin)
    update_q!(prob.solver, prob.q)
end


function update_hessian!(prob::QMPCProblem, QN)
    nx = prob.nx
    prob.P[end-nx+1:end, end-nx+1:end] .= QN
    @timeit to "update_hessian" update_hessian!(prob.solver, prob.P)
end

update_q!(prob::QMPCProblem, xr, ur, u0) = update_q!(prob.q, prob.Q1, prob.Q3, prob.QN, prob.N, xr, ur, u0)

@views function update_q!(q, Q1, Q3, QN, N, xr, ur, u0; init=false)
    nx = size(Q1, 1)
    nu = length(u0)
    ns = (length(q) - (N+1)*nx - N*nu) ÷ N # total number of slack vars (typically 2nx for state constraints) 
    n_tot = nx + nu + ns
    if xr isa AbstractVector # single goal point
        Q1r = -(Q1 * xr) # r should have been expanded to size of x by Cz'r, hence xr
        si = 1
        for i = 1:N
            copyto!(q, si, Q1r, 1, nx) # update q at all inds corresponding to x vars
            si += nx+nu+ns
        end
        mul!(q[end-nx+1:end], QN, -xr) # xN is last variable
    else # a full trajectory
        size(xr,1) == nx || error("The size of the reference trajectory was incorrect, expected size (nx, ≥N+1) and got $(size(xr)). r should at this point have been expandeed by Cz.")
        size(xr,2) ≥ N+1 || error("The size of the reference trajectory was incorrect, expected size (nx, ≥N+1) and got $(size(xr))")
        
        inds = 1:nx
        for i = 1:N
            mul!(q[inds], Q1, xr[:, i], -1, 0) # update q at all inds corresponding to x vars
            inds = inds .+ n_tot
        end
    
        mul!(q[end-nx+1:end], QN, xr[:, N+1], -1, 0)
    end
    # TODO: iszero(ur) || error("Non-zero ur not yet supported.")
    if Q3 !== nothing && !init # The very first run should have no Δu penalty
        mul!(q[(1:nu) .+ nx], Q3, -u0) # First u 
    end
    nothing
end

@views function update_q_sqp!(q, Q1, Q2, Q3, QN, N, xr, ur, xlin, ulin; init=false)
    # r should have been expanded to size of x before calling by means of Cz'r
    nx = size(Q1, 1)
    nu = size(ulin, 1)
    nr = size(xr,2)
    ns = (length(q) - (N+1)*nx - N*nu) ÷ N # total number of slack vars (typically 2nx for state constraints) 
    n_tot = nx+nu+ns
    size(xr,1) == nx || error("The size of the reference trajectory was incorrect, expected size (nx, ≥N+1) and got $(size(xr)). Has r been expanded to size of x before calling by means of Cz'r?")
    nr >= N+1 || nr == 1 || error("The size of the reference trajectory was incorrect, expected size (nx, ≥N+1) or (nx,) and got $(size(xr))")
    xri = size(xr, 2) == 1 ? xr : xr[:, 1:N+1]
    uri = size(ur, 2) == 1 ? ur : ur[:, 1:N]
    xlin .= xlin .- xri
    ulin .= ulin .- uri

    inds = 1:nx
    for i = 1:N
        # The minus sign in update_q! is baked int xlin here
        mul!(q[inds], Q1, xlin[:, i]) # update q at all inds corresponding to x vars
        inds = inds .+ n_tot
    end

    mul!(q[end-nx+1:end], QN, xlin[:, N+1], 1, 0) # The minus sign in update_q! is baked int xlin here
    
    inds = (1:nu) .+ nx
    for i = 1:N
        # The minus sign in update_q! is baked int ulin here
        mul!(q[inds], Q2, ulin[:, i]) # update q at all inds corresponding to u vars
        inds = inds .+ n_tot
    end

    if Q3 !== nothing && !init # The very first run should have no Δu penalty
        iszero(Q3) || error("The SQP procedure does not yet support Q3")
        # TODO: ulin is already advanced and does not represent u in the previous time step
    end
    nothing
end

function advance_xr!(prob, ci::Union{ControllerInput, Nothing} = nothing) # TODO: update Q3 term if required
    need_xr_advance(prob) || return
    @unpack q, nx, nu, N, Q3, xr, ur = prob
    u0 = prob.u[:, 1]
    if Q3 !== nothing
        @views mul!(q[(1:nu) .+ nx], Q3, -u0) # First u 
    end
    if ndims(prob.xr) < 2
        update_q!(prob.solver, q)
        return # we only advance reference trajectories, not goal points 
    end
    copyto!(xr, 1, xr, nx+1, length(xr)-nx) # advance xr
    copyto!(ur, 1, ur, nu+1, length(ur)-nu) # advance ur

    if ci !== nothing
        if ci.r isa AbstractVector
            length(ci.r) == size(xr, 1) || throw(DimensionMismatch("The length of the updated reference vector `r` must match the size of the reference trajectory stored in the problem."))
            xr[:, end] .= ci.r
        elseif ci.r isa AbstractMatrix
            size(ci.r) == size(xr) || throw(DimensionMismatch("The size of the updated reference trajectory matrix `r` must match the size of the reference trajectory stored in the problem."))
            xr .= ci.r
        end
    end
    
    
    update_q!(prob, xr, ur, u0) 
    # The strategy of updating q like below does not work if the reference traj is longer than the prediction horizon, q terms corresponding to reference points beyond N are never updated. That's why we currently update the full q vector (it's not expensive)
    # copyto!(q, 1, q, nx+1, (N-1)*nx) # advance q for x
    # copyto!(q, (N+1)*nx+1, q, (N+1)*nx+nu+1, (N-1)*nu) # advance q for u
    update_q!(prob.solver, q)
    # no need to update the hessian or QN here since the last point of the reference trajectory hasen't changed
    # QN, A0, B0 = calc_QN_AB(prob, prob.xr)
    # prob.QN .= QN
    # update_hessian!(prob, QN) # This updates the center term QN of the hessian
end

"linear problems need no advancements since they'll be updated by the SQP procedure"
need_xr_advance(prob) = true # isa(prob, LinearMPCProblem) || size(prob.xr, 2) > 1

"""
    optimize!(prob::QMPCProblem, x0, p, t; verbose = true)

Solve a single instance of the optimal-control problem in the MPC controller.

# Arguments:
- `x0`: Initial state
- `t`: Initial time
"""
function optimize!(prob::AbstractMPCProblem, x0, p, t; verbose = true, kwargs...)
    @views prob.x[:, 1] .= x0[:, 1]
    # update_constraints!(prob, prob.x, prob.u, x0; update_dynamics=false) # This is done in optimize!
    # TODO: we do not update xr here as we do for linear MPC, since xr is the linearization trajectory and not always the same as the reference. Figure out a more consistent way to handle this.
    # update_constraints!(prob, prob.x, prob.u, x0; update_dynamics=false) # This is done in optimize!

    co = optimize!(prob.solver, prob, p, t; verbose, kwargs...)
    prob.u .= co.u
    co
end

function optimize!(prob::AbstractMPCProblem, controllerinput::ControllerInput, args...;  kwargs...)
    optimize!(prob, controllerinput.x, args...; kwargs...)
end

abstract type AbstractMPCHistory end

"""
    NonlinearMPCHistory{T}

Call `X,E,R,U,Y,UE = reduce(hcat, history)` to get matrices.

# Fields:
- `prob::QMPCProblem`
- `X::Vector{T}`
- `E::Vector{T}`
- `R::Vector{T}`
- `U::Vector{T}`
- `Y::Vector{T}`
- `UE::Vector{T}`
"""
struct NonlinearMPCHistory{T} <: AbstractMPCHistory
    prob::AbstractMPCProblem
    X::Vector{T}
    E::Vector{T}
    R::Vector{T}
    U::Vector{T}
    Y::Vector{T}
    UE::Vector{T}
end

function NonlinearMPCHistory(prob::AbstractMPCProblem)
    # output arrays
    X = Vector{Float64}[]
    E = Vector{Float64}[]
    UE = Vector{Float64}[]
    R = Vector{Float64}[]
    U = Vector{Float64}[]
    Y = Vector{Float64}[]

    NonlinearMPCHistory(prob,X,E,R,U,Y,UE)
end


function Base.getproperty(hist::AbstractMPCHistory, s::Symbol)
    s ∈ fieldnames(typeof(hist)) && return getfield(hist, s)
    getproperty(getfield(hist, :prob), s)
end

Base.length(h::AbstractMPCHistory) = length(h.U)


function Base.push!(hist::NonlinearMPCHistory, x, u, xr, ur, y=[])
    prob = hist.prob
    @unpack nx, nu, N = prob
    x1 = x isa AbstractVector ? x : x[:, 1] # https://github.com/JuliaArrays/StaticArrays.jl/issues/1081
    u1 = u[:, 1]
    push!(hist.U, u1)
    push!(hist.X, x1)
    # TODO: update this to use xr-z
    if length(xr) == length(x1)
        e = x1 - xr # to measure error from reference
        ue = u1 - ur
    elseif length(xr) == length(y)
        e = y - xr # to measure error from reference
        ue = u1 - ur
    else
        e = zeros(size(prob.nx, 1))
        ue = zeros(prob.nu)
    end
    push!(hist.E, e)
    push!(hist.UE, ue)
    push!(hist.R, xr)
    push!(hist.Y, y)
end

advance!(hist::AbstractMPCHistory) = advance!(hist.prob)

function advance!(prob::QMPCProblem, ci::Union{ControllerInput, Nothing} = nothing)
    @unpack nx, nu, N, x, u = prob
    @timeit to "advance xr" advance_xr!(prob, ci) # This takes u0 from prob.u and must be done before advancing u
    copyto!(x, 1, x, nx+1, N*nx) # advance the state trajectory
    copyto!(u, 1, u, nu+1, (N-1)*nu) # advance the control trajectory
end

function Base.reduce(::typeof(hcat), hist::NonlinearMPCHistory)
    X = reduce(hcat, hist.X)
    E = reduce(hcat, hist.E)
    UE = reduce(hcat, hist.UE)
    R = reduce(hcat, hist.R)
    U = reduce(hcat, hist.U)
    Y = reduce(hcat, hist.Y)
    X,E,R,U,Y,UE
end

function lqr_cost(hist::NonlinearMPCHistory)
    prob = hist.prob
    U = reduce(hcat, hist.U)
    E = reduce(hcat, hist.E)
    UE = reduce(hcat, hist.UE)
    size(E, 1) == size(prob.Q1, 1) || return 0
    lqr_cost(E, UE, prob.Q1, prob.Q2, prob.Q3)
end

get_u!(prob::AbstractMPCProblem, x, u) = u


"b may be either a scalar, a vector or a matrix that is as wide or wider than a"
add_as_much_as_possible(a, b) = a .+ b
add_as_much_as_possible(a, b::AbstractMatrix) = @views a .+ b[:, 1:size(a, 2)]

"""
    step!(prob::QMPCProblem, u, y, r, p, t; kwargs...)

Take a single step using the MPC controller. 

where `u` is a matrix ``n_u \\times n_T`` where the first column corresponds to the control signal that was last taken. The rest of `u` is used as an initial guess for the optimizer. `y` is the latest measurement and is used to update the observer in `prob`. Internally, `step!` performs the following actions:
1. Measurement update of the observer, forms ``\\hat x_{k | k}``.
2. Solve the optimization problem with the state of the observer as the initial condition.
3. Advance the state of the observer using its prediction model, forms ``\\hat x_{k+1 | k}``.
4. Advance the problem caches, including the reference trajectory if `xr` is a full trajectory.

The return values of `step!` are
- `uopt`: the optimal trajectory (usually, only the first value is used in an MPC setting). This value is given in the correct space for interfacing with the true plant.
- `x`: The optimal state trajectory as seen by the optimizer, note that this trajectory will only correspond to the actual state trajectory for linear problems around the origin.
- `u0` The control signal used to update the observer in the prediction step. Similar to `xopt`, this value may contain offsets and is usually of less external use than `uopt` which is transformed to the correct units of the actual plant input.
"""
@views function step!(prob::QMPCProblem, u, y, r, p, t; kwargs...)
    mpc_observer_correct!(prob.observer, u[:, 1], y, r[:, 1], p, t)

    w = []
    controllerinput = ControllerInput(state(prob.observer), r, w, u)
    controlleroutput = optimize!(prob, controllerinput, p, t; kwargs...)
    xopt, u = controlleroutput.x, controlleroutput.u
    # The values returned by the optimizer may, depending on the dynamics in use, be in Δ-coordinates (Δx = x-x₀)
    # get_u! below will in these cases transform u to real coordinates. The correct! and predict! functions should operate in the same coordinates as the optimizer, hence we do not adjust u before calling predict!.
    # The u that is passed in to this function is also in Δ-coordinates, but y and r are in real coordinates.
    # prob.u .= u # already done in optimize!
    u0 = u[:, 1]
    mpc_observer_predict!(prob.observer, u0, r[:, 1], p, t)
    uopt = get_u!(prob, xopt, u)
    advance!(prob) # 
    ControllerOutput(xopt, uopt, controlleroutput.sol), u
end

"""
    solve(prob::AbstractMPCProblem, alg = nothing; x0, T, p, verbose = false)

Solve an MPC problem for `T` time steps starting at initial state `x0`.

Set `verbose = true` to get diagnostic outputs, including tuning tips, in case you experience poor performance.
"""
function solve(prob::QMPCProblem, alg = nothing; x0, T, verbose = false, p = parameters(prob), callback=(args...)->(), sqp_callback=(args...)->(), lqr_init=false, noise=0, reset_observer=true, dyn_actual = deepcopy(prob.dynamics), x0_actual = copy(x0), p_actual=p, disturbance = (u,t)->0)

    reset_timer!(to)
    observer = prob.observer
    if reset_observer # TODO: reconsider this strategy
        reset_observer!(observer, x0)
    end
    hist = NonlinearMPCHistory(prob)
    if lqr_init
        @timeit to "LQR init" lqr_rollout!(prob, x0)
    else
        prob.x[:, 1] .= state(observer)
    end

    actual_xr = prob.xr[:,1]
    actual_ur = prob.ur[:,1]
    actual_x = zeros(length(x0_actual), prob.N+1)
    actual_x[:, 1] .= x0_actual
    # prob.x[:,1] .= x0 # Not always of right size
    actual_u = copy(prob.u)

    u = prob.u # initialize u 
    rollout!(prob, p) # Initialize x
    # @timeit to "Main MPC loop" 
    w = []
    for t = 1:T
        if dyn_actual isa AbstractStateSpace
            msys = dyn_actual
            y = msys.C*actual_x[:, 1] + msys.D*actual_u[:, 1]
        else
            g = dyn_actual.measurement
            y = g(actual_x[:, 1], actual_u[:, 1], p, t)
        end
        if noise != false
            y .+= noise .* randn.()
        end
        # observerinput = ObserverInput(u[:, 1], y, prob.xr, w)
        controlleroutput, u = step!(prob, u, y, prob.xr, p, t; verbose, callback=sqp_callback)
        actual_u, xopt = controlleroutput.u, controlleroutput.x

        d_u = disturbance(copy(actual_u), t)
        sim_u = actual_u .+ d_u
        rollout!(dyn_actual, actual_x, sim_u, p_actual, t) # This does not update prob.x[:, 1], it's updated by advance below
        push!(hist, actual_x, actual_u, copy(actual_xr), copy(actual_ur), y) # NOTE: hist.E will use actual_x(t) but xr(t+1) since xr is advanced in step
        actual_x[:, 1:end-1] .= actual_x[:, 2:end] # For next iteration of the loop
        @views actual_xr .= prob.xr[:,1] # prob.xr has already been advanced in step!
        @views actual_ur .= prob.ur[:,1] # prob.ur has already been advanced in step!
        callback(actual_x, actual_u, xopt, hist.X, hist.U)
        # advance!(hist) # This changes u
    end
    push!(hist.X, actual_x[:, 1]) # add one last state to make the state array one longer than control

    hist
end


#=
The ploty keyword determines whether to plot outputs (y) or states (x). 
=#
@recipe function plot_MPCHistory(hist::NonlinearMPCHistory; ploty=false, plotr=true)
    X,E,R,U,Y = reduce(hcat, hist)
    dyn = hist.prob.dynamics
    prob = hist.prob
    if prob isa QMPCProblem
        cost = lqr_cost(hist)
    else
        oi2 = MPC.remake(prob.objective_input, x=X, u=U)
        cost = MPC.evaluate(prob.optprob.f.f.objective, oi2, get_mpc_parameters(get_parameter_index(prob.p, 1)), 0)
    end
    layout --> 2
    xguide --> "Time [s]"

    if ploty # Plot outputs
        @series begin
            subplot --> 1
            title --> "Outputs"
            label --> permutedims(string.(output_names(dyn)))
            timevec = range(0, step=hist.Ts, length=size(Y, 2))
            timevec, Y'
        end
        if size(R, 1) == size(Y, 1) && plotr
            # only plot references if they are state references
            @series begin
                subplot --> 1
                title --> "Outputs"
                label --> permutedims(string.(output_names(dyn))).*"ᵣ"
                linestyle --> :dash
                seriescolor --> (1:prob.ny)'
                timevec = range(0, step=hist.Ts, length=size(R, 2))
                timevec, R'
            end
        end
    else # Plot states
        @series begin
            subplot --> 1
            title --> "States"
            label --> permutedims(string.(state_names(dyn)))
            timevec = range(0, step=hist.Ts, length=size(X, 2))
            timevec, X'
        end
        if size(R, 1) == size(X, 1) && plotr
            # only plot references if they are state references
            @series begin
                subplot --> 1
                title --> "States"
                label --> permutedims(string.(state_names(dyn))).*"ᵣ"
                linestyle --> :dash
                seriescolor --> (1:prob.nx)'
                timevec = range(0, step=hist.Ts, length=size(R, 2))
                timevec, R'
            end
        end
    end
    @series begin
        subplot --> 2
        label --> permutedims(string.(input_names(dyn)))
        title --> "Control signal"
        timevec = range(0, step=hist.Ts, length=size(U, 2))
        timevec, U'
    end
end



## Forward methods to observer
for f in [:state, :correct!, :predict!]
    @eval LowLevelParticleFilters.$f(prob::AbstractMPCProblem, args...; kwargs...) = $f(prob.observer, args...; kwargs...)
end