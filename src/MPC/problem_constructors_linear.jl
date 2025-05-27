import DyadControlSystems: FixedGainObserver
import SciMLBase: remake

struct LQMPCProblem{F,XR,TQ1,TQ2,TQN,TCZ,OPZ,B,PQ,S,X,PT,X2} <: AbstractMPCProblem
    dynamics::F
    r::XR
    Q1::TQ1
    Q2::TQ2
    Q3::Union{TQ2,Nothing}
    QN::TQN
    CzQ::TQ1
    qs::Float64
    qs2::Float64
    Cz::TCZ
    opz::OPZ;
    vmin::B
    vmax::B
    N::Int
    P::PQ
    q::B
    A::PQ
    lb::B
    ub::B
    solver::S
    x::X
    u::X
    p::PT
    x0::X2
end

function Base.getproperty(prob::LQMPCProblem, s::Symbol)
    s ∈ fieldnames(typeof(prob)) && return getfield(prob, s)
    if s === :observer
        return getfield(prob, :dynamics)
    elseif s === :nx
        return size(prob.Q1, 1)::Int
    elseif s === :nu
        return size(prob.Q2, 1)::Int
    elseif s === :ny
        return prob.dynamics.ny::Int
    elseif s === :nz
        return prob.dynamics.nz::Int
    elseif s === :nv
        return prob.dynamics.nv::Int
    elseif s === :Ts
        return getfield(prob, :dynamics).Ts
    else
        throw(ArgumentError("LQMPCProblem has no property named $s"))
    end
end


"""
    LQMPCProblem(dynamics ;
        N::Int,
        Q1::AbstractMatrix,
        Q2::AbstractMatrix,
        Q3::Union{AbstractMatrix,Nothing} = nothing,
        qs2 = 1000*maximum(Q1),
        qs  = sqrt(maximum(Q1)),
        constraints,
        r,
        solver::AbstractMPCSolver = OSQPSolver(),
    )

Defines a Linear Quadratic MPC problem. The cost is on the form `(z - zᵣ)'Q1*(z - zᵣ) + (u-uᵣ)'Q2*(u-uᵣ) + Δu'Q3*Δu`.

# Arguments:
- `dynamics`: An instance of [`LinearMPCModel`](@ref) or [`RobustMPCModel`](@ref)
- `N`: Prediction horizon in the MPC problem (number of samples, equal to the control horizon)
- `Q1`: State penalty matrix
- `Q2`: Control penalty matrix
- `Q3`: Control derivative penalty matrix
- `qs`: Soft state constraint linear penalty (scalar). Set to zero for hard constraints (hard constraints may render problem infeasible).
- `qs2`: Soft state constraint quadratic penalty (scalar). Set to zero for hard constraints (hard constraints may render problem infeasible).
- `constraints`: An instance of [`MPCConstraints`](@ref)
- `r`: References. If `dynamics` contains an operating point, `dynamics.op.x` will be the default `r` if none is provided.
- `ur`: Reference control. If `dynamics` contains an operating point, `dynamics.op.u` will be the default `ur` if none is provided.
- `solver`: The solver to use. See [`OSQPSolver`](@ref)
"""
function LQMPCProblem(
    dynamics::LinearMPCModel;
    Q1::AbstractMatrix,
    Q2::AbstractMatrix,
    Q3::Union{AbstractMatrix,Nothing} = nothing,
    qs2 = 1000*maximum(Q1),
    qs = sqrt(maximum(Q1)),
    N::Int,
    r,
    solver::AbstractMPCSolver = OSQPSolver(),
    p = DiffEqBase.NullParameters(),
)
    # Method exists only to forward keyword args so that we cam dispatch on types
    LQMPCProblem(
        dynamics,
        Q1,
        Q2,
        Q3,
        qs2,
        qs,
        N,
        r,
        solver,
        p,
    )
end

function LQMPCProblem(
    dynamics::RobustMPCModel;
    Q3::Union{AbstractMatrix,Nothing} = nothing,
    qs2 = 1000*maximum(dynamics.Q1),
    qs = sqrt(maximum(dynamics.Q1)),
    N::Int,
    r,
    solver::AbstractMPCSolver = OSQPSolver(),
    p = DiffEqBase.NullParameters(),
    kwargs...
)
    # Method exists only to forward keyword args so that we cam dispatch on types
    LQMPCProblem(
        dynamics,
        dynamics.Q1,
        dynamics.Q2,
        Q3,
        qs2,
        qs,
        N,
        r,
        solver,
        p;
        kwargs...
    )
end

# Linear case ==================================================================
function LQMPCProblem(
    dynamics::LinearPredictionModel,
    Q1::AbstractMatrix,
    Q2::AbstractMatrix,
    Q3::Union{AbstractMatrix,Nothing},
    qs2,
    qs,
    N::Int,
    r,
    solver::AbstractMPCSolver,
    p,
    QN = nothing,
)
    r = copy(r) # We modify this
    @unpack Cz, Cv, Dv, vmin, vmax, soft_indices = dynamics
    BT = float(promote_type(eltype(vmin), eltype(Q1), eltype(Q2)))
    vmin, vmax = (v -> BT.(v)).((vmin, vmax))
    slack = !isempty(soft_indices) && (qs > 0 || qs2 > 0)
    if !slack
        qs = zero(qs)
        qs2 = zero(qs2)
        soft_indices = Int[]
    end

    @unpack nx, nu = dynamics
    nz = size(Cz, 1)
    nv = size(Cv, 1)
    nv == size(Dv, 1) || throw(ArgumentError("Cv and Dv have inconsistent row sizes."))
    nz == size(Q1, 1) || throw(
        ArgumentError(
            "The size of Q1 $(size(Q1, 1)) does not match the number of controlled variables $(nz)",
        ),
    )
    nu == size(Q2, 1) || throw(
        ArgumentError(
            "The size of Q2 $(size(Q2, 1)) does not match the number of control inputs $(nu)",
        ),
    )
    # nz == size(xr, 1) || throw(
    #     ArgumentError(
    #         "The first dimension of xr $(size(xr, 1)) does not match the number of controlled variables $(nz)",
    #     ),
    # )
    xconstraints = length(vmin) == nx + nu # TODO: this could be true also for Cv != I
    CzQ = Cz'Q1 # This is used to update the linear term with output references
    Q1 = CzQ*Cz # Expand matrix to size of state
    u0 = zeros(nu)
    Ad, Bd = sparse(dynamics.A), sparse(dynamics.B)
    @assert size(Ad,1) == nx
    @assert size(Bd,2) == nu
    
    if QN === nothing
        if Q3 === nothing || iszero(Q3)
            QNdense = ControlSystemsBase.are(Discrete, Matrix(dynamics.A), Matrix(dynamics.B), Matrix(Q1 + 1e-13I), Matrix(Q2))
        else
            QNdense = dare3(dynamics.A, dynamics.B, Matrix(Q1 + 1e-13I), Matrix(Q2), Matrix(Q3))
        end
        cQN = cond(QNdense)
        cQN > 1e10 && @warn "Condition number of QN is high, $cQN. This may lead to a poorly conditioned optimization problem. Use verbose=true, when constructing the OSQPSolver or the call to MPC.solve to debug solver problems. QN may become ill-conditioned if there are uncontrollable modes on the stability boundary. If possible, consider moving such modes slightly into the stable region."
        QN = sparse(QNdense)
    end


    @assert size(Q1, 1) == nx
    @assert size(Q2, 1) == nu
    @assert size(QN, 1) == nx
    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # If slack variables for soft state constraints are added, they come after the u variables. There will be twice as many slacks as states x
    # - quadratic objective
    nU = N*nu
    nX = (N + 1)*nx
    nZ = (N + 1)*nz
    nV = (N)*nv
    ns = length(soft_indices)
    nS = slack ? 2*(N)*ns : 0
    n_tot = nX + nU + nS
    P = MPC_hessian(N, Q1, Q2, Q3, QN, qs2, ns, slack)
    @assert size(P, 1) == nX + nU + nS "Hessian size error $(size(P)) expected $(nX + nU + nS)"

    r isa Number && (r = fill(r, nx))
    # - linear objective is moved to update_q!
    q = zeros(BT, nX + nU + nS)
    op = dynamics.op
    if ns > 0
        sinds = [repeat([falses(nx+nu); trues(2ns)], N); falses(nx)]
        q[sinds] .= qs
    end
    if dynamics.W1_integrator # Different realizations have r appearing in different locations
        # TODO: figure out how to handle r now when we never have state references
        opz = fill(0, size(Cz, 1))
        update_q!(q, Q1, Q3, QN, CzQ, Cz, opz, N, zeros(nz), u0; init = true) # Eq. 10.a, r enters at measurement
    else # W2 integrator 
        
        # Expand r to size of x and adjust for operating point
        opz = Cz * (op.x isa Number ? fill(op.x, nx) : op.x) # TODO: this is not correct if we add support for W2 integrator
        update_q!(q, Q1, Q3, QN, CzQ, Cz, opz, N, r, u0; init = true) # TODO: Eq 12.a, r enters state equation by Br
    end
    # - linear dynamics
    Aeq, leq, ueq = MPC_eq(N, Ad, Bd, nx, ns)
    # - input and state constraints

    if slack
        S = _make_matrix_t(soft_indices, nv, nx) |> sparse
        Spositive = kron(I(N), [spzeros(2ns, nx+nu) I(2ns)])
        Spositive = [Spositive spzeros(nS, nx)]
        @assert size(Spositive, 1) == nS
        @assert size(Spositive, 2) == n_tot

        Aineq = kron(speye(N), [Cv Dv -S 0S; Cv Dv 0S S])
        Aineq = [Aineq spzeros(2nV, nx)] # add xN
        # Terminal-set constraints could be added here by appending below Aineq
        @assert size(Aineq, 1) == 2nV
        @assert size(Aineq, 2) == n_tot

        Aineq = [Aineq; Spositive]

        lineq = [repeat([fill(-Inf, nv); vmin], N); fill(0, 2*ns*N)]
        uineq = [repeat([vmax; fill(Inf, nv)], N); fill(Inf, 2*ns*N)]

        @assert length(lineq) == length(uineq) == size(Aineq, 1)

        @assert length(vmin) == size(Cv, 1)
    else # No soft constraints
        # We keep this case separate since including soft constraints causes a doubling of the number of constraints. This is thus more effective.
        Aineq = kron(speye(N), [Cv Dv])
        Aineq = [Aineq spzeros(nV, nx)] # add xN

        lineq = repeat(vmin, N)
        uineq = repeat(vmax, N)
    end

    # - OSQP constraints
    A, lb, ub = [Aeq; Aineq], [leq; lineq], [ueq; uineq]

    solver = init_solver_workspace(solver; P, q, A, lb, ub)

    # Initial traj
    u = zeros(nu, N)
    x = zeros(nx, N + 1)
    x0 = nothing # For the ensembler for MPCSurrogates
    prob = LQMPCProblem(
        dynamics,
        r,
        Q1,
        Q2,
        Q3,
        QN,
        CzQ,
        Float64(qs),
        Float64(qs2),
        Cz,
        opz,
        vmin,
        vmax,
        N,
        P,
        q,
        A,
        lb,
        ub,
        solver,
        x,
        u,
        p,
        x0
    )
    # update_xr!(prob, xr)
    prob


end


"""
    remake(prob::LQMPCProblem; 
    dynamics = missing,
    r = missing, 
    Q1 = missing, 
    Q2 = missing, 
    Q3 = missing,
    QN = missing,
    CzQ = missing,
    qs = missing,
    qs2 = missing,
    Cz = missing,
    opz = missing,
    vmin = missing,
    vmax = missing,
    N = missing,
    P = missing,
    q = missing,
    A = missing,
    lb = missing,
    ub = missing,
    solver = missing,
    x = missing,
    u = missing,
    p = missing,
    x0 = missing)
Remake the given `LQMPCProblem`.
"""
function SciMLBase.remake(prob::LQMPCProblem;
    dynamics = missing,
    r = missing,
    Q1 = missing,
    Q2 = missing,
    Q3 = missing,
    QN = missing,
    CzQ = missing,
    qs = missing,
    qs2 = missing,
    Cz = missing,
    opz = missing,
    vmin = missing,
    vmax = missing,
    N = missing,
    P = missing,
    q = missing,
    A = missing,
    lb = missing,
    ub = missing,
    solver = missing,
    x = missing,
    u = missing,
    p = missing,
    x0 = missing
    )
    if dynamics === missing
        dynamics = prob.dynamics
    end
    if r === missing
        r = prob.r
    end
    if Q1 === missing
        Q1 = prob.Q1
    end
    if Q2 === missing
        Q2 = prob.Q2
    end
    if Q3 === missing
        Q3 = prob.Q3
    end
    if QN === missing
        QN = prob.QN
    end
    if CzQ === missing
        CzQ = prob.CzQ
    end
    if qs === missing
        qs = prob.qs
    end
    if qs2 === missing
        qs2 = prob.qs2
    end
    if Cz === missing
        Cz = prob.Cz
    end
    if opz === missing
        opz = prob.opz
    end
    if vmin === missing
        vmin = prob.vmin
    end
    if vmax === missing
        vmax = prob.vmax
    end
    if N === missing
        N = prob.N
    end
    if P === missing
        P = prob.P
    end
    if q === missing
        q = prob.q
    end
    if A === missing
        A = prob.A
    end
    if lb === missing
        lb = prob.lb
    end
    if ub === missing
        ub = prob.ub
    end
    if solver === missing
        solver = prob.solver
    end
    if x === missing
        x = prob.x
    end
    if u === missing
        u = prob.u
    end
    if p === missing
        p = prob.p
    end
    if x0 === missing
        x0 = prob.x0
    end
    LQMPCProblem(
        dynamics,
        r,
        Q1,
        Q2,
        Q3,
        QN,
        CzQ,
        Float64(qs),
        Float64(qs2),
        Cz,
        opz,
        vmin,
        vmax,
        N,
        P,
        q,
        A,
        lb,
        ub,
        solver,
        x,
        u,
        p,
        x0
    )

end



