import DyadControlSystems: FixedGainObserver
abstract type AbstractMPCProblem end

struct QMPCProblem{F,O,XR,TQ1,TQ2,TQN,C<:NonlinearMPCConstraints,B,PQ,SP,S,CA,CB,X,PT} <: AbstractMPCProblem
    dynamics::F # TODO: add constrained outputs and performance outputs to FunctionSystem
    observer::O
    xr::XR
    ur::XR
    Q1::TQ1
    Q2::TQ2
    Q3::Union{TQ2,Nothing}
    QN::TQN
    qs::Float64
    qs2::Float64
    constraints::C
    N::Int
    P::PQ
    q::B
    A::PQ
    lb::B
    ub::B
    Ad_pattern::SP
    Bd_pattern::SP
    solver::S
    chunkA::CA
    chunkB::CB
    x::X
    u::X
    p::PT
    ns::Int # Number of soft constraints (num slack is 2ns)
    s::X
end

LowLevelParticleFilters.parameters(prob::AbstractMPCProblem) = prob.p

function Base.getproperty(prob::AbstractMPCProblem, s::Symbol)
    s ∈ fieldnames(typeof(prob)) && return getfield(prob, s)
    if s === :nx
        return prob.dynamics.nx
    elseif s === :nu
        return prob.dynamics.nu
    elseif s === :ny
        return prob.observer.ny
    elseif s === :nz
        return prob.dynamics.nz
    elseif s === :nv
        return length(prob.constraints.min)
    elseif s === :Ts
        return prob.dynamics.Ts
    else
        throw(ArgumentError("QMPCProblem has no property named $s"))
    end
end

"""
    QMPCProblem(dynamics ;
        N::Int,
        Q1::AbstractMatrix,
        Q2::AbstractMatrix,
        Q3::Union{AbstractMatrix,Nothing} = nothing,
        qs2 = 1000*maximum(Q1),
        qs  = sqrt(maximum(Q1)),
        constraints::NonlinearMPCConstraints,
        observer = nothing,
        xr,
        ur = zeros(size(Q2,1)),
        solver::AbstractMPCSolver = OSQPSolver(),
        chunkA = ForwardDiff.Chunk{min(8, size(Q1, 1))}(),
        chunkB = ForwardDiff.Chunk{min(8, size(Q2, 1))}(),
    )

Defines a Nonlinear Quadratic MPC problem. The cost is on the form `(z - zᵣ)'Q1*(z - zᵣ) + (u-uᵣ)'Q2*(u-uᵣ) + Δu'Q3*Δu`.

# Arguments:
- `dynamics`: An instance of [`FunctionSystem`](@ref) representing `x(t+1) = f(x(t), u(t), p, t)`, i.e., already discretized.
- `observer`: An instance of `AbstractObserver`, defaults to `StateFeedback(dynamics, zeros(nx))`.
- `N`: Prediction horizon in the MPC problem (number of samples, equal to the control horizon)
- `Q1`: Controlled variable penalty matrix
- `Q2`: Control signal penalty matrix
- `Q3`: Control derivative penalty matrix
- `qs`: Soft state constraint linear penalty (scalar). Set to zero for hard constraints (hard constraints may render problem infeasible).
- `qs2`: Soft state constraint quadratic penalty (scalar). Set to zero for hard constraints (hard constraints may render problem infeasible).
- `constraints`: An instance of [`NonlinearMPCConstraints`](@ref)
- `xr`: Reference state (or reference output if `state_reference=false` for `LinearMPCModel`). If `dynamics` contains an operating point, `dynamics.op.x` will be the default `xr` if none is provided.
- `ur`: Reference control. If `dynamics` contains an operating point, `dynamics.op.u` will be the default `ur` if none is provided.
- `solver`: The solver to use. See [`OSQPSolver`](@ref)

"""
function QMPCProblem(
    dynamics;
    Q1::AbstractMatrix,
    Q2::AbstractMatrix,
    Q3::Union{AbstractMatrix,Nothing} = nothing,
    qs2 = 1000*maximum(Q1),
    qs = sqrt(maximum(Q1)),
    constraints::NonlinearMPCConstraints,
    observer = nothing,
    N::Int,
    xr,
    ur = zeros(size(Q2,1)),
    solver::AbstractMPCSolver = OSQPSolver(),
    chunkA = ForwardDiff.Chunk{min(8, size(Q1, 1))}(),
    chunkB = ForwardDiff.Chunk{min(8, size(Q2, 1))}(),
    p = DiffEqBase.NullParameters(),
    QN = nothing,
)
    # Method exists only to forward keyword args so that we cam dispatch on types
    QMPCProblem(
        dynamics,
        Q1,
        Q2,
        Q3,
        qs2,
        qs,
        constraints,
        observer,
        N,
        xr,
        ur,
        solver,
        chunkA,
        chunkB,
        p,
        QN
    )
end

# General case =================================================================
function QMPCProblem(
    dynamics::FunctionSystem,
    Q1::AbstractMatrix,
    Q2::AbstractMatrix,
    Q3::Union{AbstractMatrix,Nothing},
    qs2,
    qs,
    constraints::NonlinearMPCConstraints,
    observer,
    N::Int,
    xr,
    ur,
    solver::AbstractMPCSolver,
    chunkA,
    chunkB,
    p,
    QN = nothing,
)

    observer === nothing && (observer = StateFeedback(dynamics, zeros(size(Q1, 1)), size(Q2, 1), size(Q1, 1)))
    
    c = constraints
    soft_indices = c.soft_indices

    BT = float(promote_type(eltype(c.min), eltype(Q1), eltype(Q2)))
    slack = !isempty(soft_indices) && (qs > 0 || qs2 > 0)
    if !slack
        qs = zero(qs)
        qs2 = zero(qs2)
        soft_indices = Int[]
    end
    nx = dynamics.nx
    nu = size(Q2, 1)
    Cv, Dv = linearize(c.fun, xr[:,1], ur[:,1], p, 0)
    nv = size(Cv, 1)
    nv == length(c.min) == length(c.max) || throw(ArgumentError("The constraint-vector lengths do not match the output of the constraint function."))
    is_feasible(c, xr[:,1], ur[:,1], p, 0) || error("Initial reference point (xr,ur) is not feasible according to the provided constraints.")

    Cz = if hasproperty(dynamics, :Cz)
        dynamics.Cz
    elseif hasproperty(dynamics, :z) && dynamics.z !== nothing
        _make_matrix(dynamics.z, nx)
    else
        I(nx)
    end
    nz = size(Cz, 1)
    Q1 = Cz'Q1*Cz # Expand matrix to size of state
    xr = copy(xr)
    ur = copy(ur)
    u0 = zeros(nu)
    if QN === nothing
        QN, A0, B0 = calc_QN_AB(Q1, Q2, Q3, dynamics, xr, ur, p)
    else
        A0, B0 = linearize(dynamics, 1e-16*ones(nx)+xr[:,end], ur[:,end], p, 0)
    end

    cQN = cond(Matrix(QN))
    cQN > 1e10 && @warn "Condition number of QN is high, $cQN. This may lead to a poorly conditioned optimization problem. Use verbose=true, when constructing the OSQPSolver or the call to MPC.solve to debug solver problems. QN may become ill-conditioned if there are uncontrollable modes on the stability boundary. If possible, consider moving such modes slightly into the stable region."

    nz == size(Q1, 1) || throw(
        ArgumentError(
            "The size of Q1 $(size(Q1, 1)) does not match the number of controlled variables $(nz)",
        ),
    )
    size(xr, 1) == nz || throw(
        ArgumentError(
            "The first dimension of xr $(size(xr, 1)) does not match the number of controlled variables $(nz)",
        ),
    )

    Ad = sparse(A0) # TODO: Will not get the correct sparsity pattern if some element is 0 by chance
    Bd = sparse(B0)

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # If slack variables for soft state constraints are added, they come after the u variables. There will be twice as many slacks as states x
    # - quadratic objective
    # P = blockdiag(kron(speye(N), Q1), QN, kron(speye(N), Q2))
    nU = N*nu
    nX = (N + 1)*nx
    nZ = (N + 1)*nz
    nV = (N)*nv
    ns = length(soft_indices)
    nS = slack ? 2*(N)*ns : 0
    n_tot = nX + nU + nS
    P = MPC_hessian(N, Q1, Q2, Q3, QN, qs2, ns, slack)

    # - linear objective is moved to update_q!
    sinds = [repeat([falses(nx+nu); trues(2ns)], N); falses(nx)]
    q = zeros(BT, nX + nU + nS)
    q[sinds] .= qs
    if solver.sqp_iters > 1
        if xr isa AbstractVector
            xlin = zeros(nx, N+1) .= xr
        else
            xlin = zeros(nx, N+1) .= xr[:,1:N+1]
        end
        if ur isa AbstractVector
            ulin = zeros(nu, N) .= ur
        else
            ulin = zeros(nu, N) .= ur[:,1:N]
        end
        update_q_sqp!(q, Q1, Q2, Q3, QN, N, xr, ur, xlin, ulin; init = true)
    else
        update_q!(q, Q1, Q3, QN, N, xr, ur, u0; init = true)
    end
    # - linear dynamics
    Aeq, leq, ueq = MPC_eq(N, Ad, Bd, nx, ns)
    # - input and state constraints
    uineqinds = 1:N*nu
    vmin, vmax = c.min, c.max

    S = _make_matrix_t(soft_indices, nv, nx) |> sparse
    Spositive = kron(I(N), [spzeros(2ns, nx+nu) I(2ns)])
    Spositive = [Spositive spzeros(nS, nx)]
    @assert size(Spositive, 1) == nS
    @assert size(Spositive, 2) == n_tot

    Aineq = kron(speye(N), [Cv Dv -S 0S; Cv Dv 0S S])
    Aineq = [Aineq spzeros(2nV, nx)]
    @assert size(Aineq, 1) == 2nV
    @assert size(Aineq, 2) == n_tot

    Aineq = [Aineq; Spositive]

    lineq = [repeat([fill(-Inf, nv); vmin], N); fill(0, 2*ns*N)]
    uineq = [repeat([vmax; fill(Inf, nv)], N); fill(Inf, 2*ns*N)]

    @assert length(lineq) == length(uineq) == size(Aineq, 1)

    @assert length(vmin) == size(Cv, 1)
    # - OSQP constraints
    A, lb, ub = [Aeq; Aineq], [leq; lineq], [ueq; uineq]

    solver = init_solver_workspace(solver; P, q, A, lb, ub)

    # Initial traj
    u = zeros(nu, N)
    x = zeros(nx, N + 1)
    s = zeros(2ns, N)

    Ad_pattern = findall(Ad .!= 0)
    Bd_pattern = findall(Bd .!= 0)

    prob = QMPCProblem(
        dynamics,
        observer,
        xr,
        ur,
        Q1,
        Q2,
        Q3,
        QN,
        Float64(qs),
        Float64(qs2),
        constraints,
        N,
        P,
        q,
        A,
        lb,
        ub,
        Ad_pattern,
        Bd_pattern,
        solver,
        chunkA,
        chunkB,
        x,
        u,
        p,
        ns,
        s,
    )
    # update_xr!(prob, xr)
    prob
end



# Internal function populating the hessian of the QP
function MPC_hessian(N, Q1, Q2, Q3, QN, qs2, ns, slack)
    if Q3 === nothing
        Q3 = 0Q2
    end
    nx = size(Q1,1)
    nu = size(Q3,1)
    Q = cat(Q1, Q2+2Q3, dims=(1,2))
    Q3neg = cat(0Q1, -Q3, dims=(1,2))
    if slack
        Q = cat(Q, qs2*speye(2ns), dims = (1, 2)) # No quadratic term for slack variables
        Q3neg = cat(Q3neg, spzeros(2ns,2ns), dims = (1, 2))
    end
    # add control derivative penalty
    P3neg = kron(spdiagm(-1 => ones(N), 1 => ones(N)), Q3neg)[1:end-2ns-nu, 1:end-2ns-nu]


    # Currently no slack for xN
    P = cat(kron(speye(N), Q), QN, dims = (1, 2))
    P = P + P3neg 
    
    ntot = size(P, 1)
    last_u_inds = (1:nu) .+ ((N-1)*(nx + nu + 2ns) + nx)

    P[last_u_inds, last_u_inds] .-= Q3 # last term was added one time too much above
    # u1inds = (1:nu) .+ nx
    # P[u1inds, u1inds] .-= Q3 # last term was added one time too much above

    P
end

# Internal function constructing the equality-constraint arrays of the QP
function MPC_eq(N, Ad, Bd, nx, ns)

    nu = size(Bd, 2)
    S = zeros(nx, 2ns)

    # We currently have no constraint on the final state. The slacks are laied out after the control which is a bit inconvenient when it comes to adding a slack for the final state
    Ap1 = kron(speye(N+1), [-speye(nx) zeros(nx, nu+2ns)])[:, 1:end-nu-2ns] # Indexing in end to remove control and slack for N+1
    ABS = [Ad Bd S]
    Aeq = kron(spdiagm(-1 => ones(N)), ABS)[:, 1:end-nu-2ns] + Ap1

    @assert size(Aeq, 1) == (N+1)*nx
    @assert size(Aeq, 2) == (N+1)*(nx) + N*(nu+2ns)

    x0 = zeros(nx)
    leq = [-x0; zeros(N*nx)]
    ueq = leq
    Aeq, leq, ueq
end
