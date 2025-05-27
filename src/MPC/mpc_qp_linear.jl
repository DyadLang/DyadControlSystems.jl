using ForwardDiff
using OSQP
using SparseArrays
using LinearAlgebra, Statistics
using RecipesBase
import ControlSystems
using ControlSystems: AbstractStateSpace, ssdata, Discrete
using UnPack
import CommonSolve: solve
using TimerOutputs

using LowLevelParticleFilters: measurement, sample_measurement, parameters


## =============================================================================
include("problem_constructors_linear.jl")
## =============================================================================



"""
    lqr_rollout!(prob, x = prob.x, u = prob.u, xr = prob.xr; update_L = true)

Trajectory rollout using an LQR controller.

# Arguments:
- `xr`: Reference state. May be a vector or a matrix with time in second dimension.
- `update_L`: if false, the state feedback is calculated using an infinite horizon cost function and the dynamics is linearized around the final reference state.
"""
function lqr_rollout!(prob::LQMPCProblem, x=prob.x, u=prob.u, r=prob.r; update_L = true)
    ref(r::AbstractVector, i) = r
    ref(r::AbstractMatrix, i) = @views r[:, i]
    f = prob.dynamics
    if !update_L
        ri = r[:, end]
        QN, A, B = calc_QN_AB(prob)
        L = (B'*QN*B + R)\(B'QN*A)
    end
    @views for i = 1:size(x, 2)-1
        ri = ref(r, i)
        if update_L
            QN, A, B = calc_QN_AB(prob)
            L = (B'*QN*B + prob.Q2)\(B'QN*A)
        end
        u[:, i] = L*(ri - x[:, i])
        x[:, i+1] = f(x[:, i], u[:, i], nothing, i)
    end
    x, u
end



"""
    update_constraints!(prob::LQMPCProblem, x, u, x0, p=parameters(prob); update_dynamics = true)

Update the problem and internal solver with new constraint vectors. Constraints are used to set the initial state as well as lower and upper bounds on states and control inputs. Currently, only the initial state is updated by this function.

If `update_dynamics = true`, the constraint matrix `A` will also be updated using new linearized dynamics.
"""
function update_constraints!(prob::LQMPCProblem, x, u, x0, p=parameters(prob); update_dynamics=true)
    # Update initial state
    nx = length(x0)
    @. prob.lb[1:nx] = -x0
    @. prob.ub[1:nx] = -x0
    @timeit to "update l u" update_constraints!(prob.solver, prob.lb, prob.ub)
    # For a linear system, we never need update any dynamics
end


function calc_QN_AB(Q1, Q2, Q3, dyn::Union{LinearPredictionModel, ControlSystemsBase.AbstractStateSpace}, p=nothing) # TODO: do not accept p=nothing
    nx, nu = size(Q1, 1), size(Q2, 1)
    A0, B0 = dyn.A, dyn.B
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

"""
    update_xr!(prob::LQMPCProblem, r, u0; update_QN=false)

Update the reference value in `prob` to `r`. This function also allows you to set a new initial control value`u0`.

If `update_QN = true`, the cost function will be updated using the infinite-horizon LQR cost at the final point in `r`, this is computationally costly and can be avoided if not required.
"""
function update_xr!(prob::LQMPCProblem, r, u0; force=false, update_QN=false)
    Q1, Q2, Q3 = prob.Q1, prob.Q2, prob.Q3
    N = prob.N
    (isempty(r) || isnothing(r)) && return
    if !force && prob.r === r
        return
    end
    prob.r == r && return # Weaker but slower form of comparison
    size(r, 1) == size(prob.r, 1) || error("The size of the updated reference trajectory was incorrect, expected size $(size(prob.r)) and got $(size(r)).")
    size(r,2) ∈ (1, size(prob.r, 2)) || error("The size of the updated reference trajectory was incorrect, expected size $(size(prob.r)) and got $(size(r)).")
    prob.r .= r
    if update_QN
        QN, A0, B0 = calc_QN_AB(prob)
        prob.QN .= QN
    end
    update_q!(prob, r, u0)
    update_q!(prob.solver, prob.q)
    if update_QN
        update_hessian!(prob, prob.QN)
    end
end

function update_hessian!(prob::LQMPCProblem, QN)
    nx = prob.nx
    prob.P[end-nx+1:end, end-nx+1:end] .= QN
    @timeit to "update_hessian" update_hessian!(prob.solver, prob.P)
end

function update_q!(prob::LQMPCProblem, r,u0)
    update_q!(prob.q, prob.Q1, prob.Q3, prob.QN, prob.CzQ, prob.Cz, prob.opz, prob.N, r, u0)
end

@views function update_q!(q, Q1, Q3, QN, CzQ, Cz, opz, N, r, u0; init=false)

    r = r .- opz # TODO: reduce allocs

    nz = size(Cz, 1)
    nx = size(Q1, 1)
    nu = length(u0)
    ns = (length(q) - (N+1)*nx - N*nu) ÷ N # total number of slack vars (typically 2nx for state constraints) 
    n_tot = nx + nu + ns
    CzQN = QN*pinv(Cz) # TODO: This is maybe not correct, the problem is that QN is directly generated and thus not of the correct size for r
    if r isa AbstractVector # single goal point
        Q1r = -(CzQ * r)
        si = 1
        for i = 1:N
            copyto!(q, si, Q1r, 1, nx) # update q at all inds corresponding to x vars
            si += nx+nu+ns
        end
        mul!(q[end-nx+1:end], CzQN, -r) # xN is last variable
    else # a full trajectory
        size(r,1) == nz || error("The size of the reference trajectory was incorrect, expected size (nz, ≥N+1) = ($nz, $(N+1)) and got $(size(r)). Has the correct output matrix Cz been supplied to LinearMPCModel?")
        size(r,2) ≥ N+1 || error("The size of the reference trajectory was incorrect, expected size (nz, ≥N+1) = ($nz, $(N+1)) and got $(size(r)).")
        inds = 1:nx
        for i = 1:N
            mul!(q[inds], CzQ, r[:, i], -1, 0) # update q at all inds corresponding to x vars
            inds = inds .+ n_tot
        end
    
        mul!(q[end-nx+1:end], CzQN, r[:, N+1], -1, 0)
    end
    if Q3 !== nothing && !init # The very first run should have no Δu penalty
        mul!(q[(1:nu) .+ nx], Q3, -u0) # First u 
    end
    nothing
end

function advance_xr!(prob::LQMPCProblem, ci::Union{ControllerInput, Nothing} = nothing) # TODO: update Q3 term when required
    need_xr_advance(prob) || return
    @unpack q, nx, nz, nu, N, Q3, r, Cz, dynamics = prob
    u0 = prob.u[:, 1]
    if Q3 !== nothing
        @views mul!(q[(1:nu) .+ nx], Q3, -u0) # First u 
    end
    if ndims(r) < 2
        update_q!(prob.solver, q)
        return # we only advance reference trajectories, not goal points 
    end
    copyto!(r, 1, r, nz+1, length(r)-nz) # advance xr

    if ci !== nothing
        if ci.r isa AbstractVector
            length(ci.r) == size(r, 1) || throw(DimensionMismatch("The length of the updated reference vector `r` must match the size of the reference trajectory stored in the problem."))
            r[:, end] .= ci.r
        elseif ci.r isa AbstractMatrix
            size(ci.r) == size(r) || throw(DimensionMismatch("The size of the updated reference trajectory matrix `r` must match the size of the reference trajectory stored in the problem."))
            r .= ci.r
        end
    end
    
    update_q!(prob, r, u0) 
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
need_xr_advance(prob::LQMPCProblem) = true # isa(prob, LinearMPCProblem) || size(prob.xr, 2) > 1

"""
    optimize!(prob::LQMPCProblem, x0, p, t; verbose = true)
    optimize!(prob::LQMPCProblem, ci::ControllerInput, p, t; verbose = true)

Solve a single instance of the optimal-control problem in the MPC controller.

!!! NOTE
    This low-level function does not adjust for operating points, the user is thus responsible for handling operating points in the correct way when calling this function directly. This adjustment typically takes the form
    ```
    Δx = x - op.x
    co = MPC.optimize!(prob, Δx, p, t)
    Δu = co.u[1:nu]
    u = Δu + op.u
    ```

# Arguments:
- `x0`: Initial state
- `ci`: An instance of [`ControllerInput`](@ref) containing the initial state, reference, and previous control input. Providing this struct offers the possibility to update the reference as well.
- `t`: Initial time

# Returns
An instance of [`ControllerOutput`](@ref) containing the optimal control input trajectory and the optimal state trajectory as seen by the optimizer. Note that the state trajectory will only correspond to the actual state trajectory for linear problems around the origin.
"""
function optimize!(prob::LQMPCProblem, x0, p, t; verbose = true, kwargs...)
    @views prob.x[:, 1] .= x0[:, 1]
    # update_constraints!(prob, prob.x, prob.u, x0; update_dynamics=false) # This is done in optimize!

    co = optimize!(prob.solver, prob, p, t; verbose, kwargs...)
    prob.u .= co.u
    co
end

function optimize!(prob::LQMPCProblem, controllerinput::ControllerInput, args...;  kwargs...)
    @views prob.x[:, 1] .= controllerinput.x[:, 1]
    update_xr!(prob, controllerinput.r, controllerinput.u0, update_QN=false)
    # update_constraints!(prob, prob.x, prob.u, x0; update_dynamics=false) # This is done in optimize!
    optimize!(prob.solver, prob, args...; kwargs...)
end


struct LinearMPCHistory{T} <: AbstractMPCHistory
    prob::LQMPCProblem
    X::Vector{T}
    E::Vector{T}
    R::Vector{T}
    U::Vector{T}
    Y::Vector{T}
    UE::Vector{T}
end

function LinearMPCHistory(prob::LQMPCProblem; sizehint=100)
    # output arrays
    sizehint += 2 # for good measure in case we save N+1 and initial point etc.
    X = sizehint!(Vector{Float64}[], sizehint)
    E = sizehint!(Vector{Float64}[], sizehint)
    UE = sizehint!(Vector{Float64}[], sizehint)
    R = sizehint!(Vector{Float64}[], sizehint)
    U = sizehint!(Vector{Float64}[], sizehint)
    Y = sizehint!(Vector{Float64}[], sizehint)

    LinearMPCHistory(prob,X,E,R,U,Y,UE)
end



function Base.push!(hist::LinearMPCHistory, x, u, r, y=[], z=[])
    prob = hist.prob
    @unpack nx, nu, N = prob
    x1 = x[:, 1]
    u1 = u[:, 1]
    push!(hist.U, u1)
    push!(hist.X, x1)

    # @unpack C, Cz = prob.dynamics
    if prob.dynamics.W1_integrator
        e = y - r # to measure error from reference
    else
        e = z - r # to measure error from reference
    end

    push!(hist.E, e) # errors
    push!(hist.R, r) # errors
    push!(hist.Y, y)
end


function advance!(prob::LQMPCProblem)
    @unpack nx, nu, N, x, u = prob
    @timeit to "advance xr" advance_xr!(prob) # This takes u0 from prob.u and must be done before advancing u
    copyto!(x, 1, x, nx+1, N*nx) # advance the state trajectory
    copyto!(u, 1, u, nu+1, (N-1)*nu) # advance the control trajectory
end

function Base.reduce(::typeof(hcat), hist::LinearMPCHistory)
    X = reduce(hcat, hist.X)
    E = reduce(hcat, hist.E)
    R = reduce(hcat, hist.R)
    U = reduce(hcat, hist.U)
    Y = reduce(hcat, hist.Y)
    X,E,R,U,Y
end

function lqr_cost(hist::LinearMPCHistory)
    prob = hist.prob
    U = reduce(hcat, hist.U)
    E = reduce(hcat, hist.E)
    size(E, 1) == size(prob.Q1, 1) || return 0
    lqr_cost(E, U, prob.Q1, prob.Q2, prob.Q3)
end

get_u!(prob::LQMPCProblem{<:LinearMPCModel}, x, us) = add_as_much_as_possible(us, prob.dynamics.op.u)
# get_u!(prob::LQMPCProblem{<:ControlSystemsBase.LTISystem}, x, us) = us .+ prob.ur[:,1]
function get_u!(prob::LQMPCProblem{<:RobustMPCModel}, x, us)
    # Filter us through W1 to get u, see fig 2
    pm = prob.dynamics
    W1 = pm.W1
    if W1 isa AbstractStateSpace
        # # Filter us through W1
        # u1 and u2 produce the same u[:, 1] but differ later. Using u2 since that does not require an extra lsim
        # x = pm.xw
        # x .= W1.A*x .+ W1.B*us[:, 1]
        # u1 = ControlSystemsBase.lsim(W1, us, x0=x).y

        # extract the states that correspond to u from xw
        # TODO: might not need to return the whole array u here, keep only first index
        u2 = add_as_much_as_possible(W1.C*prob.x[pm.w1inds,1:end-1], W1.D*us)

        return add_as_much_as_possible(u2, pm.op.u)
    elseif W1 === nothing
        return add_as_much_as_possible(us, pm.op.u)
    else
        return add_as_much_as_possible(W1*us, pm.op.u)
    end
end


"""
    step!(prob::LQMPCProblem, observerinput, p, t; kwargs...)

Take a single step using the MPC controller. 


Internally, `step!` performs the following actions:
1. Measurement update of the observer, forms ``\\hat x_{k | k}``.
2. Solve the optimization problem with the state of the observer as the initial condition.
3. Advance the state of the observer using its prediction model, forms ``\\hat x_{k+1 | k}``.
4. Advance the problem caches, including the reference trajectory if `xr` is a full trajectory.

The return value of `step!` is an instance of [`ControllerOutput`](@ref), containing the fields
- `u`: the optimal trajectory (usually, only the first value is used in an MPC setting). This value is given in the correct space for interfacing with the true plant.
- `x`: The optimal state trajectory as seen by the optimizer, note that this trajectory will only correspond to the actual state trajectory for linear problems around the origin.

To bypass the observer handling, call the function `optimize!` with the initial state `x0` or an instance of [`ControllerInput`](@ref) directly:
```julia
controlleroutput = MPC.optimize!(prob, x0, p, t; kwargs...)
```
"""
@views function step!(prob::LQMPCProblem, observerinput, p, t; kwargs...)
    @unpack u,y,r,w = observerinput


    # TODO: update docstring
    mpc_observer_correct!(prob.observer, u[:, 1], y, r[:, 1], p, t)
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
    ControllerOutput(xopt, uopt, controlleroutput.sol)
end

"""
    solve(prob::AbstractMPCProblem, alg = nothing; x0, T, p, verbose = false)

Solve an MPC problem for `T` time steps starting at initial state `x0`.

Set `verbose = true` to get diagnostic outputs, including tuning tips, in case you experience poor performance.
"""
function solve(prob::LQMPCProblem, alg = nothing; x0 = prob.x0, T, verbose = false, p = parameters(prob), callback=(args...)->(), lqr_init=false, noise=0, reset_observer=true, dyn_actual = deepcopy(prob.dynamics), x0_actual = copy(x0), p_actual=p, disturbance = (u,t)->0, Cz_actual = prob.dynamics.Cz)

    reset_timer!(to)
    observer = prob.observer
    if reset_observer # TODO: reconsider this strategy
        reset_observer!(observer, x0)
    end
    hist = LinearMPCHistory(prob, sizehint=T)
    if lqr_init
        @timeit to "LQR init" lqr_rollout!(prob, x0)
    else
        prob.x[:, 1] .= state(observer)
    end

    actual_r = prob.r[:, 1]
    # actual_ur = prob.ur[:,1]
    actual_x = zeros(length(x0_actual), prob.N+1)
    actual_x[:, 1] .= x0_actual
    # prob.x[:,1] .= x0 # Not always of right size
    actual_u = copy(prob.u)
    sim_u = copy(prob.u)

    u = prob.u # initialize u 
    rollout!(prob, p) # Initialize x
    # @timeit to "Main MPC loop" 
    w = []
    for t = 1:T
        if dyn_actual isa AbstractStateSpace
            y = dyn_actual.C*actual_x[:, 1] + dyn_actual.D*actual_u[:, 1]
        else
            g = dyn_actual.measurement
            y = @views g(actual_x[:, 1], actual_u[:, 1], p, t)
        end
        if noise != false
            y .+= noise .* randn.()
        end
        observerinput = ObserverInput(u[:, 1], y, prob.r, w)
        controlleroutput = step!(prob, observerinput, p, t; verbose)

        actual_u, xopt = controlleroutput.u, controlleroutput.x

        d_u = disturbance(copy(actual_u), t)
        @views sim_u .= actual_u .+ d_u
        rollout!(dyn_actual, actual_x, sim_u, p_actual, t) # This does not update prob.x[:, 1], it's updated by advance below
        actual_z = Cz_actual*actual_x[:,1]
        push!(hist, actual_x, actual_u, copy(actual_r), y, actual_z) # NOTE: hist.E will use actual_x(t) but xr(t+1) since xr is advanced in step
        @views actual_x[:, 1:end-1] .= actual_x[:, 2:end] # For next iteration of the loop
        @views actual_r .= prob.r[:,1] # prob.xr has already been advanced in step!
        # @views actual_ur .= prob.ur[:,1] # prob.ur has already been advanced in step!
        callback(actual_x, actual_u, xopt, hist.X, hist.U)
        # advance!(hist) # This changes u
    end
    push!(hist.X, actual_x[:, 1]) # add one last state to make the state array one longer than control

    hist
end


#=
The ploty keyword determines whether to plot outputs (y) or states (x). 
=#
@recipe function plot_MPCHistory(hist::LinearMPCHistory; ploty=false)
    X,E,R,U,Y = reduce(hcat, hist)
    dyn = hist.prob.dynamics
    cost = lqr_cost(hist)
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
        if size(R, 1) == size(Y, 1)
            # only plot references if they are output references
            @series begin
                subplot --> 1
                title --> "Outputs"
                label --> permutedims(string.(output_names(dyn))).*"ᵣ"
                linestyle --> :dash
                seriescolor --> (1:hist.prob.ny)'
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
        if size(R, 1) == size(X, 1)
            # only plot references if they are state references
            @series begin
                subplot --> 1
                title --> "States"
                label --> permutedims(string.(state_names(dyn))).*"ᵣ"
                linestyle --> :dash
                seriescolor --> (1:hist.prob.nx)'
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


