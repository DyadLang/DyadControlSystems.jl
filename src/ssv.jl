using LinearAlgebra
# using SCS
using Convex
using Convex: MOI
using RobustAndOptimalControl: getinds, ρ
#=
References for these algorithms include
- Boyd et al., "Linear Matrix Inequalities in System and Control Theory"
- Skogestad, "Multivariable Feedback Control: Analysis and Design"

The general approach is to find the smallest `b` such that the an LMI is satisfied.
This is done by bisection over b

Most of the time is spent on setting up the problem by Convex. Unfortunately, a Convex.jl problem can not be reused, and has to be created from scratch for each frequency. Interfacing the solver directly could mitigate this problem, at the expense of having to figure out the translation of the LMI into a SOCP problem manually.
=#

"""
    mussv(M::AbstractMatrix; optimizer = Hypatia.Optimizer{eltype(M)}, bu = opnorm(M), tol = 0.001)

Compute (an upper bound of) the structured singular value μ for diagonal Δ of complex perturbations (other structures of Δ are handled by supplying the block structure mussv(M, blocks)).
`M` is assumed to be an (n × n × N_freq) array or a matrix.

Solves a convex LMI.

# Arguments:
- `bu`: Upper bound for bisection. 
- `tol`: tolerance.

See also [`mussv`](@ref), [`mussv_tv`](@ref), [`mussv_DG`](@ref).

# Extended help
By default, the Hypatia solver (native Julia) is used to solve the problem. Other solvers that handle the this problem formulation is
- Mosek (requires license)
- SCS
The solver can be selected using the keyword argument `optimizer`.
"""
function mussv(M::AbstractMatrix; optimizer=Hypatia.Optimizer{real(eltype(M))}, bu=opnorm(M), bl = ρ(M), tol=1e-3)
    bu-bl < tol && return bu
    n = size(M, 1)
    iters = ceil(Int, log2((bu-bl+1e-16)/tol))  
    C = similar(M, real(eltype(M)))
    x = Convex.Variable(n)
    X = Diagonal(x)
    for j = 1:iters
        b = (bu+bl)/2
        con = M'X*M - b^2*X ⪯ -I(n)
        prob = Convex.satisfy([con, x > 0], numeric_type=real(eltype(M)))
        Convex.solve!(prob, optimizer, warmstart=false, silent=true)
        Xe = Convex.evaluate(X)
        C .= real.(M'Xe*M - b^2*Xe)
        satisfied = Int(prob.status) == 1 && all(<(0), real(eigvals(Symmetric(C))))
        if satisfied
            bu = b
        else
            bl = b
        end
    end
    bu
end

"""
    mussv_DG(M::AbstractMatrix; optimizer = Hypatia.Optimizer{eltype(M)}, bu = opnorm(M), tol = 0.001)

Compute (an upper bound of) the structured singular value μ for diagonal Δ of real perturbations (other structures of Δ are not yet supported).
`M` is assumed to be an (n × n × N_freq) array or a matrix.

See [`mussv`](@ref) for more details. See also [`mussv`](@ref), [`mussv_tv`](@ref), [`mussv_DG`](@ref).
"""
function mussv_DG(M::AbstractMatrix; optimizer=Hypatia.Optimizer{real(eltype(M))}, bu=opnorm(M), bl = 0.0, tol=1e-3)
    bu-bl < tol && return bu
    n = size(M, 1)
    iters = ceil(Int, log2((bu - bl)/tol)) 
    C = similar(M, real(eltype(M)))
    # P = Convex.Semidefinite(n)
    # G = Convex.Variable(n,n) # This amount of freedoom is too muc§h
    p = Convex.Variable(n)
    g = Convex.Variable(n)
    P = Diagonal(p)
    G = Diagonal(g)
    for j = 1:iters
        b = (bu+bl)/2
        con = M'P*M + im*(M'G - G*M) - b^2*P ⪯ -I(n) # eq 3.5 in LMI book
        prob = Convex.satisfy([con, p > 0], numeric_type=real(eltype(M)))
        Convex.solve!(prob, optimizer, warmstart=false, silent=true)
        Pe = Convex.evaluate(P)
        Ge = Convex.evaluate(G)
        C = (M'Pe*M + im*(M'Ge - Ge*M) - b^2*Pe)
        satisfied = Int(prob.status) == 1 && all(<(0), real(eigvals(C)))
        if satisfied
            bu = b
        else
            bl = b
        end
    end
    bu
end

@doc raw"""
    mussv_tv(M::AbstractArray{<:Any, 3}; optimizer = Hypatia.Optimizer, bu = 20, tol = 0.001)
    mussv_tv(G::UncertainSS; optimizer = Hypatia.Optimizer, bu0 = 20, tol = 0.001)

Compute (an upper bound of) the structured singular value μ for diagonal, complex and time-varying Δ(t) using constant (over frequency) matrix scalings.
This value will in general be larger than the one returned by [`mussv`](@ref), but can be used to guarantee stability for *infinitely fast time-varying perturbations*, i.e., if the return value is < 1, the system is stable no matter how fast the dynamics of the perturbations change.

`M` is assumed to be an (n × n × N_freq) array or a matrix.
`G` is an [`UncertainSS`](@ref). The result will in general be more accurate if `G` is passed rather than `M`, unless a very dense grid around the critical frequency is used for to calculate `M`.

Solves a convex LMI.

# Arguments:
- `bu`: Upper bound for bisection. 
- `tol`: tolerance.

See also [`mussv`](@ref), [`mussv_tv`](@ref), [`mussv_DG`](@ref).

# Extended help
The normal μ is calculated by minimizing
```math
\operatorname{min}_D ||D(\omega) M(\omega) D(\omega)^{-1}||
```
where a unique $D(\omega)$ is allowed for each frequency. However, in this problem, the $D$ scalings are constant over frequency, yielding a more conservative answer, with the additional bonus of being applicable for time-varying perturbations.

This strong guarantee can be used to prove stability of nonlinear systems by formulating them as linear systems with time-varying, norm-bounded perturbations. For such systems, `mussv_tv < 1` is a sufficient condition for stability. See 
Boyd et al., "Linear Matrix Inequalities in System and Control Theory" for more details.
"""
function mussv_tv(M::AbstractArray{T, 3}; optimizer=Hypatia.Optimizer{real(T)}, bu=20, tol=1e-3) where T
    n = size(M, 1)
    iters = ceil(Int, log2(bu/tol))
    bl = 0.0    
    x = Convex.Variable(n)
    X = Diagonal(x)
    for j = 1:iters
        b = (bu+bl)/2
        cons = map(axes(M, 3)) do i
            Mi = M[:, :, i]
            Mi'X*Mi - b^2*X ⪯ -I(n)
        end
        prob = Convex.satisfy([cons; x > 0], numeric_type=real(T))
        Convex.solve!(prob, optimizer, warmstart=false, silent=true)
        Xe = Convex.evaluate(X)
        satisfied = Int(prob.status) == 1 && all(axes(M,3)) do i
            Mi = @views M[:,:,i]
            all(<(0), real(eigvals(Symmetric(real(Mi'Xe*Mi - b^2*Xe)))))
        end
        if satisfied
            bu = b
        else
            bl = b
        end
    end
    bu
end

@doc raw"""
    mussv_tv(G::AbstractStateSpace, blocks; optimizer = Hypatia.Optimizer, bu = 20, tol = 0.001)

Compute (an upper bound of) the structured singular value margin when `G` is interconnected with *time-varying* uncertainty structure described by `blocks`.
This value will in general be larger than the one returned by [`mussv`](@ref), but can be used to guarantee stability for *infinitely fast time-varying perturbations*, i.e., if the return value is < 1, the system is stable no matter how fast the dynamics of the perturbations change.

The result will in general be more accurate if `G` is passed rather than a matrix `M`, unless a very dense grid around the critical frequency is used for to calculate `M`.

Solves a convex LMI.

# Arguments:
- `bu`: Upper bound for bisection. 
- `tol`: tolerance.

See also [`mussv`](@ref), [`mussv_tv`](@ref), [`mussv_DG`](@ref).
"""
function mussv_tv(G::AbstractStateSpace, blocks; optimizer=Hypatia.Optimizer, bu=20, tol=1e-3)
    # reference: MAE509 (LMIs in Control): Lecture 14, part C - LMIs for Robust Control with Structured Uncertainty
    iters = ceil(Int, log2(bu/tol))
    bl = 0.0 
    Θ = Convex.HermitianSemidefinite(size(G, 1))
    cons = build_Q_con(Θ, blocks)
    # For this algorithm, we bisect over `b` scaling the gain of the uncertainty mapping
    for j = 1:iters
        b = (bu+bl)/2
        A,M,N,Q = ssdata(G)
        N = N ./ b
        Q = Q ./ b
        n,m = size(M)   
        P = Semidefinite(n)
        # con = [ # This version is not linear in μ
        #     A*P+P*A' P*N'  M'
        #     N*P    -μ*I(n) Q'
        #     M         Q    -1/μ*I(m) 
        # ] ⪯ -I(3n)
        MQ = [M; Q]
        con = [
            A*P+P*A' P*N'
            N*P     0*I(m)
        ]  + [
            M*Θ*M'  M*Θ*Q'
            Q*Θ*M'  Q*Θ*Q'-Θ
        ] ⪯ -I(n+m)
        prob = satisfy(con)
        prob.constraints += cons
        Convex.solve!(prob, optimizer, silent=true)
        satisfied = Int(prob.status) == 1
        if satisfied
            bu = b
        else
            bl = b
        end
    end
    bu
end

function mussv(M::AbstractMatrix, blocks; optimizer=Hypatia.Optimizer{real(eltype(M))}, bu=opnorm(M), tol=1e-3, shortcut=true, retQ=false)
    if shortcut
        if all(b[1] < 0 for b in blocks) && all(b[2] <=1 for b in blocks) # only real scalar blocks
            return mussv_DG(M; optimizer, bu, tol, retQ)
        elseif all(b[1] > 0 for b in blocks) && all(b[2] <=1 for b in blocks) # only complex scalar blocks
            return mussv(M; optimizer, bu, tol, retQ)
        elseif length(blocks) == 1 # only one block but not scalar
            return bu # full complex block
        end
    end
    n = size(M, 1)
    bl = ρ(M)
    bu-bl < tol && return retQ ? (bu, I(n)) : bu
    iters = ceil(Int, log2((bu-bl)/tol))
    hasreal = any(b[1] < 0 for b in blocks)
    if hasreal
        error("This function does not yet support real perturbations")
        if any(b[1] > 0 for b in blocks)
            error("Mixed real and complex perturbations not yet supported. Consider approximating all uncertainties as a single complex uncertainty.")
        end
    end
    Q = Convex.HermitianSemidefinite(n)
    cons = build_Q_con(Q, blocks)

    # for i = 1:n
    #     push!(cons, imag(Q[i,i]) == 0)
    # end
    retQ && (Qout = Matrix(1.0I(n)))
    for j = 1:iters
        b = (bu+bl)/2
        con = M'Q*M - b^2*Q ⪯ -I(n)
        prob = Convex.satisfy([con; cons]; numeric_type=real(eltype(M)))
        opt = optimizer()
        # MOI.set(opt, MOI.RawOptimizerAttribute("eps_infeas"), 1e-12) # needed for SCS
        Convex.solve!(prob, optimizer, warmstart=false, silent=true)
        Qe = Convex.evaluate(Q)
        C = (M'Qe*M - b^2*Qe)
        satisfied = Int(prob.status) == 1 && all(<(0), real(eigvals(Hermitian(C))))
        if satisfied
            bu = b
            retQ && (Qout = Qe)
        else
            bl = b
        end
    end
    retQ ? (bu, Qout) : bu
end

mussv(G::UncertainSS, blocks; kwargs...) = mussv(G.M, blocks; kwargs...)
mussv_tv(G::UncertainSS, blocks; kwargs...) = mussv_tv(G.M, blocks; kwargs...)


function mussv(G::UncertainSS, w::AbstractVector{<:Number}; kwargs...)
    blocks, M0 = RobustAndOptimalControl.blocksort(G)
    M = freqresp(M0, w)
    mussv(M, blocks; kwargs...)
end

for f in [:mussv, :mussv_DG]
    @eval function $f(M::AbstractArray{<:Any, 3}, args...; kwargs...)
        map(axes(M,3)) do i
            @views $f(M[:,:,i], args...; kwargs...)
        end
    end
    @eval function $f(M0::AbstractStateSpace, w::AbstractVector{<:Number}, args...; kwargs...)
        M = freqresp(M0, w)
        $f(M, args...; kwargs...)
    end
end


"""
    mussv(G::UncertainSS; kwargs...)

Compute (an upper bound of) the structured singular value μ of uncertain system model `G`.
"""
function mussv(G::UncertainSS; kwargs...)
    blocks, M0 = RobustAndOptimalControl.blocksort(G)
    mussv(M0, blocks; kwargs...)
end

function mussv_tv(G::UncertainSS; kwargs...)
    blocks, M0 = RobustAndOptimalControl.blocksort(G)
    mussv_tv(M0, blocks; kwargs...)
end

"""
    mussv(M::AbstractStateSpace, blocks; optimizer = Hypatia.Optimizer, bu0 = 20, tol = 0.001)

Compute (an upper bound of) the structured singular value μ of statespace model `M` interconnected with uncertainty structure described by `blocks`.
Reference: MAE509 (LMIs in Control): Lecture 14, part C - LMIs for Robust Control with Structured Uncertainty

# Example:
The following example illustrates how to use the structured singular value to determine how large diagonal complex uncertainty can be added at the input of a plant `P` before the closed-loop system becomes unstable
```
julia> Δ = uss([δc(), δc()]); # Diagonal uncertainty element

julia> a = 1;

julia> P = ss([0 a; -a -1], I(2), [1 a; 0 1], 0) * (I(2) + Δ);

julia> K = ss(I(2));

julia> G = lft(P, -K);

julia> stabmargin = 1/mussv(G) # We can handle 134% of the modeled uncertainty
1.3429508196721311

julia> # Compare with the input diskmargin
       diskmargin(K*system_mapping(P), -1)
Disk margin with:
Margin: 1.3469378397689664
Frequency: -0.40280561122244496
Gain margins: [-0.3469378397689664, 2.3469378397689664]
Phase margin: 84.67073122411068
Skew: -1
Worst-case perturbation: missing
```

# Extended help
By default, the Hypatia solver is used to solve the problem. Other solvers that handle the this problem formulation is
- Mosek (requires license)
- Hypatia.jl (native Julia)
- Clarabel.jl (native Julia)
- SCS (typically performs poorly for this problem)
The solver can be selected using the keyword argument `optimizer`.
"""
function mussv(M::AbstractStateSpace, blocks; optimizer=Hypatia.Optimizer{ControlSystemsBase.numeric_type(M)}, bu0=20, tol=1e-3, retQ=false, γrel=1, perf = 0, ϵ=1)
    isstable(M) || error("M is not stable")
    M = minreal(M)
    iters = ceil(Int, log2(bu0/tol))
    bu = bu0
    bl = 0.0
    hasreal = any(b[1] < 0 for b in blocks)
    if hasreal
        error("This function does not yet support real perturbations")
        if any(b[1] > 0 for b in blocks)
            error("Mixed real and complex perturbations not yet supported. Consider approximating all uncertainties as a single complex uncertainty.")
        end
    end
    A,B,C,D = ssdata(M)
    n, m = size(B)
    Q = Convex.HermitianSemidefinite(m)
    X = Convex.Semidefinite(n)
    cons = build_Q_con(Q, blocks)
    if perf > 0
        push!(cons, Q[end-perf+1:end, end-perf+1:end] == I(perf))
    end
    # for i = 1:n
    #     push!(cons, imag(Q[i,i]) == 0)
    # end
    CD = [C D]
    Qout = Matrix(1.0I(m))
    for j = 1:iters
        b = (bu+bl)/2
        con = [
            A'X+X*A   X*B
            B'X       -Q
        ] + 1/b^2*CD'Q*CD ⪯ -ϵ*I(n+m)
        prob = Convex.satisfy([con; cons]; numeric_type=ControlSystemsBase.numeric_type(M))
        opt = optimizer()
        Convex.solve!(prob, optimizer, silent=true)
        satisfied = Int(prob.status) == 1
        if satisfied
            bu = b
            Qout = Convex.evaluate(Q)
        else
            bl = b
        end
    end
    if γrel > 1 && retQ
        con = [
            A'X+X*A   X*B
            B'X       -Q
        ] + 1/(γrel*bu)^2*CD'Q*CD ⪯ -ϵ*I(n+m)
        prob = Convex.satisfy([con; cons]; numeric_type=ControlSystemsBase.numeric_type(M))
        opt = optimizer()
        Convex.solve!(prob, optimizer, silent=true)
        if Int(prob.status) == 1
            Qout = Convex.evaluate(Q)
        end
    end
    retQ ? (bu, Qout) : bu
end

function build_Q_con(Q, blocks)
    rinds = getinds(blocks, 1)
    cinds = getinds(blocks, 2)
    cons = map(zip(blocks, rinds, cinds)) do (b, r, c)
        if b[2] <= 1 # scalar block
            b[1] < 0 && @error "Mixed real and complex perturbations is currently not supported, the real perturbation will be approximated by a comlex perturbation (conservative)."
            enforce_full(Q, r, c)
        else
            b[1] == b[2] || error("Only square full blocks currently supported")
            enforce_identity(Q, r)
        end
    end
    cons = reduce(vcat, cons)
end


function enforce_identity(Q, inds)
    c = map(inds[2:end]) do i
        Q[i,i] == Q[inds[1], inds[1]]
    end
    # for j in inds
    #     push!(c, imag(Q[j,j]) == 0)
    # end
    for i in axes(Q,1), j in inds
        i == j && continue
        push!(c, Q[i,j] == 0)
    end
    for i in inds, j in axes(Q,2)
        i == j && continue
        push!(c, Q[i,j] == 0)
    end
    c
end

function enforce_full(Q, r, c)
    c = Convex.Constraint[]
    for i in axes(Q,1), j in c
        i ∈ c && continue
        push!(c, Q[i,j] == 0)
    end
    for i in r, j in axes(Q,2)
        j ∈ r && continue
        push!(c, Q[i,j] == 0)
    end
    c
end