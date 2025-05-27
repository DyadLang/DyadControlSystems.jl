import Convex
using Convex: ⪯, ⪰
using ControlSystems: balance_statespace
using JuMP: PSDCone, @constraint

function safe_lqr(timeevol, A, B, Q, R, args...; kwargs...)
    local V2, k1
    are = timeevol isa ControlSystemsBase.ContinuousType ? arec : ared
    try
        V2, _, k1 = are(A, B, R, Q, args...; kwargs...)
    catch e
        @warn "Failed to find solution to the Ricatti equation, error repreoduced below. Performing simplification of the linearized system and finding a linear controller for the reduced-order system" e
        # C = [Q; zeros(nu,nx)]
        # D = [zeros(nx,nu); R]
        sysr = sminreal(ss(A, B, I, 0))
        Qr = sysr.C'Q*sysr.C
        V2r, _, k1r = are(sysr.A, sysr.B, R, Qr, args...; kwargs...)
        k1 =  k1r*sysr.C'
        V2 = sysr.C*V2r*sysr.C'
    end
    k1, V2
end

"""
    K, γ = hinfsyn_lmi(P::ExtendedStateSpace;
        opt = Hypatia.Optimizer(), γrel = 1.01, ϵ = 1e-3, balance = true, perm = false,)

Computes an H-infinity optimal controller `K` for an extended plant `P` such that
||F_l(P, K)||∞ < γ (`lft(P, K)`) for the smallest possible γ given P. This implementation solves a
convex LMI problem. 
# Arguments:
- `opt`: A MathOptInterface solver constructor.
- `γrel`: If `γrel > 1`, the optimal γ will be found by γ iteration after which a controller will be designed for `γ = γopt * γrel`. It is often a good idea to design a slightly suboptimal controller, both for numerical reasons, but also since the optimal controller may contain very fast dynamics. If `γrel → ∞`, the computed controller will approach the 𝑯₂ optimal controller. Getting a mix between 𝑯∞ and 𝑯₂ properties is another reason to choose `γrel > 1`.
- `ϵ`: A small offset to enforce strict LMI inequalities. This can be tuned if numerical issues arise.
- `balance`: Perform a balancing transformation on `P` using `ControlSystemsBase.balance_statespace`.
- `perm`: If `balance=true, perm=true`, the balancing transform is allowed to permute the state vector. This is not allowed by default, but can improve the numerics slightly if allowed.

The Hypatia solver takes the following arguments
https://github.com/chriscoey/Hypatia.jl/blob/42e4b10318570ea22adb39fec1c27d8684161cec/src/Solvers/Solvers.jl#L73
"""
function hinfsyn_lmi(P0::ExtendedStateSpace{ControlSystemsBase.Continuous, T};
        ftype = Float64,
        opt = Hypatia.Optimizer{ftype},
        γrel = 1.01,
        ϵ = 1e-3,
        balance = true,
        perm = false,
        verbose = false,
    ) where T

    if balance
        P0, Tr = ControlSystemsBase.balance_statespace(P0, perm)
    end
    Thigh = promote_type(T, ftype)
    hp = Thigh != T
    if hp
        bb(x) = Thigh.(x)
        mats = bb.(ssdata_e(P0))
        P = ss(mats..., P0.timeevol)
    else 
        P = P0
    end
    A,B1,B2,C1,C2,D11,D12,D21,D22 = ssdata_e(P)
    n = size(A,1)
    ny1 = size(C1, 1)
    nu1 = size(B1, 2)
    ny2 = size(C2, 1)
    nu2 = size(B2, 2)
    γ  = Convex.Variable()
    X1 = Convex.Variable(n,n)
    Y1 = Convex.Variable(n,n)
    An = Convex.Variable(n,n)
    Bn = Convex.Variable(n, ny2) # Yes, sizes flipped
    Cn = Convex.Variable(nu2, n)
    Dn = Convex.Variable(nu2, ny2)

    # γ = 1.5
    M11 = A*Y1 + Y1*A' + B2*Cn + Cn'*B2'
    M21 = A' + An + (B2*Dn*C2)'
    M31 = (B1 + B2*Dn*D21)'
    M41 = C1*Y1 + D12*Cn
    M22 = X1*A + A'*X1 + Bn*C2 + C2'*Bn'
    M32 = (X1*B1 + Bn*D21)'
    M42 = C1 + D12*Dn*C2
    M43 = D11 + D12*Dn*D21
    M33 = -γ*Matrix(I(nu1))
    M44 = -γ*Matrix(I(ny1))
    M = [
        M11 M21' M31' M41'
        M21 M22  M32' M42'
        M31 M32  M33  M43'
        M41 M42  M43  M44
    ]
    prob = Convex.minimize(γ; numeric_type=Thigh)
    prob.constraints += γ > 0
    prob.constraints += M ⪯ -ϵ*I(size(M,1))
    prob.constraints += [X1 I(n); I(n) Y1] ⪰ ϵ*I(2n)
    Convex.solve!(prob, opt; silent=!verbose)

    γ = Convex.evaluate(γ)
    if γrel > 1
        γ *= γrel
        M33 = -γ*I(nu1)
        M44 = -γ*I(ny1)
        M = [
            M11 M21' M31' M41'
            M21 M22  M32' M42'
            M31 M32  M33  M43'
            M41 M42  M43  M44
        ]
        prob = Convex.satisfy(M ⪯ -ϵ*I(size(M,1)); numeric_type=Thigh)
        prob.constraints += [X1 I(n); I(n) Y1] ⪰ ϵ*I(2n)
        Convex.solve!(prob, opt; silent=!verbose)
    end

    X1, Y1, An, Bn, Cn, Dn = Convex.evaluate.((X1, Y1, An, Bn, Cn, Dn))

    Id(n) = Matrix(I(n)) # Dense version to handle BigFloat etc.
    X2 = I - X1*Y1
    Y2 = I(n)
    K2 = [X2 X1*B2; zeros(nu2, n) Id(nu2)] \ ([An Bn; Cn Dn] - [X1*A*Y1 zeros(n, ny2); zeros(nu2, n+ny2)]) / [Y2' zeros(n, ny2); C2*Y1 I]
    Ak2 = K2[1:n, 1:n]
    Bk2 = K2[1:n, n+1:end]
    Ck2 = K2[n+1:end, 1:n]
    Dk2 = K2[n+1:end, n+1:end]

    
    Dk = (I + Dk2*D22)\Dk2
    R = (I - D22*Dk)
    cond(R) > 1e10 && @warn "Condition number of (I - D22*Dk) is very high $(cond(R))"
    Bk = Bk2 * R
    Ck = (I - Dk*D22)*Ck2
    Ak = Ak2 - Bk*(R \ D22*Ck)
    K = ss(Ak, Bk, Ck, Dk)
    if hp
        bf(x) = T.(x)
        mats = bf.(ssdata(K))
        K = ss(mats..., K.timeevol)
    end
    γactual = hinfnorm2(lft(P0, K))[1]
    if balance && perm
        # If we applied state permutation, we permute back to the original coordinates.
        K = similarity_transform(K, Tr)
    end
    abs(γ - γactual) > 0.1 && @warn "Returned γ is adjusted to the γ achieved by the computed controller."
    K, γactual
end


"""
    ispassive_lmi(P::AbstractStateSpace{ControlSystemsBase.Continuous}; ftype = Float64, opt = Hypatia.Optimizer{ftype}(), ϵ = 0.001, balance = true, verbose = true, silent_solver = true)

Determine if square system `P` is passive, i.e., ``P(s) + Pᴴ(s) > 0``.

A passive system has a Nyquist curve that lies completely in the right half plane, and satisfies the following inequality (dissipation of energy)
```math
\\int_0^T y^T u dt > 0 ∀ T
```
The negative feedback-interconnection of two passive systems is stable and  parallel connections of two passive systems as well as the inverse of a passive system are also passive. A passive controller will thus always yeild a stable feedback loop for a passive system. A series connection of two passive systems *is not* always passive.

This functions solves a convex LMI related to the KYP (positive real) lemma.

# Arguments:
- `balance`: Balance the system before calculations?
- `verbose`: Print status messages
"""
function ispassive_lmi(P0::AbstractStateSpace{ControlSystemsBase.Continuous};
    ftype = Float64,
    opt = Hypatia.Optimizer{ftype},
    balance = true,
    verbose = true,
)

    P0.ny == P0.nu || throw(ArgumentError("ispassive only defined for square systems"))
    T = ControlSystemsBase.numeric_type(P0)
    if balance
        P0, Tr = ControlSystemsBase.balance_statespace(P0, true)
    end
    Thigh = promote_type(T, ftype)
    hp = Thigh != T
    if hp
        bb(x) = Thigh.(x)
        mats = bb.(ssdata(P0))
        P = ss(mats..., P0.timeevol)
    else 
        P = P0
    end
    A,B,C,D = ssdata(P)
    n = size(A,1)
    X = Convex.Semidefinite(n,n)

    M = [
        A'X+X*A   X*B-C'
        B'X-C     -(D' + D)
    ]
    prob = Convex.satisfy(M ⪯ -0*I(size(M,1)); numeric_type=Thigh)
    Convex.solve!(prob, opt; silent=!verbose)
    verbose && @info prob.status
    Int(prob.status) == 1
end



"""
    K, Gcl = spr_synthesize(P0::ExtendedStateSpace{Continuous};, opt = Hypatia.Optimizer, balance = true, verbose = true, silent_solver = true, ϵ = 1e-6)

Design a strictly positive real controller (passive) that optimizes the closed-loop H₂-norm subject to being passive. 

For plants that are known to be passive, control using a passive controller is guaranteed to be stable.

The returned controller is supposed to be used with positive feedback, so `ispassive(-K)` should hold.
The resulting closed-loop system from disturbances to performance outputs is also returned, `Gcl = lft(P0, K)`.

Implements the algorithm labeled as "Pseudocode 1" in 
"Synthesis of strictly positive real H2 controllers using dilated LMI", Forbes 2018

# Arguments:
- `P0`: An `ExtendedStateSpace` object. This object can be designed using H₂ or H∞ methods. See, e.g., [`hinfpartition`](@ref).
- `opt`: A JuMP compatible solver.
- `balance`: Perform balancing of the statespace system before solving.
- `verbose`: Print info?
- `silent_solver`: 
- `ϵ`: A small numerical constant to enforce strict positive definiteness.

See also [`h2synthesize`](@ref), [`hinfsynthesize`](@ref).
"""
function spr_synthesize(P0::ExtendedStateSpace{ControlSystemsBase.Continuous};
    opt = Hypatia.Optimizer,
    balance = true,
    verbose = true,
    silent_solver = true,
    ϵ = 1e-6,
    check = true,
    )
    
    model = JuMP.Model(opt)
    JuMP.set_optimizer_attribute(model, JuMP.MOI.Silent(), silent_solver)
    if balance
        P0, Tr = ControlSystemsBase.balance_statespace(P0, true)
    end

    
    A,B1,B2,C1,C2,D11,D12,D21,D22 = ssdata_e(P0)
    if check
        rank(ctrb(A, B1)) == size(A,1) || error("System is not controllable through B1")
        rank(ctrb(A, B2)) == size(A,1) || error("System is not controllable through B2")
        rank(obsv(A, C1)) == size(A,1) || error("System is not observable through C1")
        rank(obsv(A, C2)) == size(A,1) || error("System is not observable through C2")
        all(iszero, D12'C1) || error("Assumption D12'C1 = 0 not satisfied")
        all(iszero, D21*B1') || error("Assumption D21*B1' = 0 not satisfied")
    end
    all(iszero, D11) || error("Assumption D11 = 0 not satisfied")
    all(iszero, D22) || error("Assumption D22 = 0 not satisfied")

    R = lu(D21*D21', check = false)
    issuccess(R) || throw(ArgumentError("D21*D21' is not invertible, spr_synthesize failed."))

    PI, _, = ControlSystemsBase.MatrixEquations.arec(A', Symmetric(C2'*(R\C2)), Symmetric(B1*B1')) # from (2) and (3) Define observer gain.
    L = PI*C2'/R
    all(iszero, L) && @warn "Observer gain is all zeros"


    Bzw = [B1; L*D21]
    nx,nz = size(Bzw)
    

    AL  = A - L*C2
    Q = B2*L' + L*B2'
    
    n = size(A,1)
    nc = size(C1, 1)
    JuMP.@variable(model, X[1:nx,1:nx], PSD)
    JuMP.@variable(model, Z[1:nz,1:nz], PSD)
    JuMP.@variable(model, Gb[1:n,1:n], PSD)
    JuMP.@variable(model, Gh[1:n,1:n], PSD)
    G = [Gh Gb; Gb Gb]
    ng = size(G,1)

    AzwG = [
        A*Gh-B2*L'             A*Gb-B2*L'
        L*C2*Gh+AL*Gb-B2*L'    L*C2*Gb+AL*Gb-B2*L'
    ]
    CzwG = [
        C1*Gh-D12*L'   C1*Gb-D12*L'
    ]

    M1 = [
        Z Bzw'
        Bzw X
    ]

    M2 = [
        AzwG+AzwG' CzwG'
        CzwG       -I(nc)
    ]

    JuMP.@variable(model, ν)
    JuMP.@objective(model, Min, ν)
    @constraint(model, M1 >= ϵ*I, PSDCone())
    @constraint(model, -M2 >= ϵ*I, PSDCone())
    @constraint(model, tr(Z) <= ν)
    @constraint(model, -(AL*Gb + Gb*AL' - Q) >= ϵ*I, PSDCone())
    @constraint(model, G >= ϵ*I, PSDCone())

    JuMP.optimize!(model)
    ν = JuMP.value(ν)
    if verbose
        @info JuMP.termination_status(model), √ν
    end

    Gb = JuMP.value(Gb)
    K = L'/Gb
    all(iszero, K) && @warn "Controller gain is all zeros"
    Ac = AL - B2*K
    Bc = L
    Cc = -K


    Azw = [
        A   -B2*K
        L*C2 Ac
    ]
    Czw = [C1 -D12*K]
    Tzw = ss(Azw, Bzw, Czw, 0)

    ss(Ac,Bc,Cc,0), Tzw, √ν
end

# Below follws two implementations of alg 1 and 2 using Convex instead of JuMP. Convex appears buggy for these problems and completely fail to solve the problem no matter how it's tweaked.

# function spr_synthesize(P0::AbstractStateSpace{ControlSystemsBase.Continuous};
#     ftype = Float64,
#     opt = Hypatia.Optimizer{ftype}(),
#     balance = true,
#     verbose = true,
#     silent_solver = true,
#     ϵ = 1e-3,
# )

#     T = ControlSystemsBase.numeric_type(P0)
#     if balance
#         P0, Tr = ControlSystemsBase.balance_statespace(P0, true)
#     end
#     Thigh = promote_type(T, ftype)
#     hp = Thigh != T

    
#     A,B1,B2,C1,C2,_,D12,D21,_ = ssdata_e(P0)
#     PI, _, = ControlSystemsBase.MatrixEquations.arec(A', Symmetric(C2'*((D21*D21')\C2)), Symmetric(B1*B1')) # from (2) and (3) Define observer gain.
#     L = PI*C2'/(D21*D21')

#     if hp
#         bb(x) = Thigh.(x)
#         mats = bb.(ssdata_e(P0))
#         P = ss(mats..., P0.timeevol)
#     else 
#         P = P0
#     end
#     A,B1,B2,C1,C2,_,D12,D21,_ = ssdata_e(P)

#     Bzw = [B1; L*D21]
#     nx,nz = size(Bzw)
    

#     AL  = A - L*C2
#     Q = B2*L' + L*B2'
    
#     n = size(A,1)
#     nc = size(C1, 1)
#     X = Convex.Semidefinite(nx,nx)
#     Z = Convex.Semidefinite(nz,nz)
#     Gb = Convex.Semidefinite(n,n)
#     Gh = Convex.Semidefinite(n,n)
#     G = [Gh Gb; Gb Gb]
#     ng = size(G,1)

#     AzwG = [
#         A*Gh-B2*L'             A*Gb-B2*L'
#         L*C2*Gh+AL*Gb-B2*L'    L*C2*Gb+AL*Gb-B2*L'
#     ]
#     CzwG = [
#         C1*Gh-D12*L'   C1*Gb-D12*L'
#     ]

#     M1 = [
#         Z Bzw'
#         Bzw X
#     ]

#     M2 = [
#         AzwG+AzwG' CzwG'
#         CzwG       -I(nc)
#     ]

#     ν = Convex.Variable()
#     prob = Convex.minimize(ν; numeric_type=Thigh)
#     # prob.constraints += Gb == Gb'
#     prob.constraints += M1 > ϵ*I(nx+nz)
#     prob.constraints += M2 < -ϵ*I(size(M2,1))
#     prob.constraints += tr(Z) < ν
#     prob.constraints += AL*Gb + Gb*AL' - Q < -ϵ*I(n)
#     prob.constraints += G > ϵ

#     Convex.solve!(prob, opt; silent_solver, verbose)
#     ν = Convex.evaluate(ν)
#     if verbose
#         @info prob.status, √ν
#     end
#     Int(prob.status) == 1

#     Gb = Convex.evaluate(Gb)
#     # Gb = (Gb .+ Gb') ./ 2
#     K = L'/Gb
#     Ac = AL - B2*K
#     Bc = L
#     Cc = -K


#     Azw = [
#         A   -B2*K
#         L*C2 Ac
#     ]
#     Czw = [C1 -D12*K]
#     Tzw = ss(Azw, Bzw, Czw, 0)

#     ss(Ac,Bc,Cc,0), Tzw, √ν
# end


# function spr_synthesize(P0::AbstractStateSpace{ControlSystemsBase.Continuous};
#     ftype = Float64,
#     opt = Hypatia.Optimizer{ftype}(),
#     balance = true,
#     verbose = true,
#     silent_solver = true,
#     ϵ = 1e-3,
# )

#     T = ControlSystemsBase.numeric_type(P0)
#     if balance
#         P0, Tr = ControlSystemsBase.balance_statespace(P0, true)
#     end
#     Thigh = promote_type(T, ftype)
#     hp = Thigh != T

    
#     A,B1,B2,C1,C2,_,D12,D21,_ = ssdata_e(P0)
#     PI, _, = ControlSystemsBase.MatrixEquations.arec(A', Symmetric(C2'*((D21*D21')\C2)), Symmetric(B1*B1')) # from (2) and (3) Define observer gain.
#     L = PI*C2'/(D21*D21')

#     if hp
#         bb(x) = Thigh.(x)
#         mats = bb.(ssdata_e(P0))
#         P = ss(mats..., P0.timeevol)
#     else 
#         P = P0
#     end
#     A,B1,B2,C1,C2,_,D12,D21,_ = ssdata_e(P)

#     Bzw = [B1; L*D21]
#     nx,nz = size(Bzw)
    

#     AL  = A - L*C2
#     Q = B2*L' + L*B2'
    
#     n = size(A,1)
#     nc = size(C1, 1)
#     X = Convex.Semidefinite(nx,nx)
#     Z = Convex.Semidefinite(nz,nz)
#     Gb = Convex.Semidefinite(n,n)
#     Gh = Convex.Variable(n,n)
#     G = [Gh Gb; Gb Gb]
#     ng = size(G,1)

#     AzwG = [
#         A*Gh-B2*L'             A*Gb-B2*L'
#         L*C2*Gh+AL*Gb-B2*L'    L*C2*Gb+AL*Gb-B2*L'
#     ]
#     CzwG = [
#         C1*Gh-D12*L'   C1*Gb-D12*L'
#     ]

#     M1 = [
#         Z Bzw'
#         Bzw X
#     ]
#     M2 = [
#         0I(nx) -X    zeros(nx, nc)
#         -X    0I(nx) zeros(nx, nc)
#         zeros(nc, 2nx) -I(nc)
#     ]

#     M3 = [
#         AzwG -ϵ*AzwG zeros(ng, nc)
#         G    -ϵ*G    zeros(ng, nc)
#         CzwG -ϵ*CzwG zeros(nc, nc)
#     ]
#     M2 += M3 + M3'

#     ν = Convex.Variable()
#     prob = Convex.minimize(ν; numeric_type=Thigh)
#     # prob.constraints += Gb == Gb'
#     prob.constraints += M1 > 0#ϵ*I(nx+nz)
#     prob.constraints += M2 < 0#-ϵ*I(2nx+nc)
#     prob.constraints += tr(Z) < ν
#     prob.constraints += AL*Gb + Gb*AL' - Q < 0#-ϵ*I(n)

#     Convex.solve!(prob, opt; silent_solver, verbose)
#     ν = Convex.evaluate(ν)
#     if verbose
#         @info prob.status, √ν
#     end
#     Int(prob.status) == 1

#     Gb = Convex.evaluate(Gb)
#     # Gb = (Gb .+ Gb') ./ 2
#     K = L'/Gb
#     Ac = AL - B2*K
#     Bc = L
#     Cc = -K


#     Azw = [
#         A   -B2*K
#         L*C2 Ac
#     ]
#     Czw = [C1 -D12*K]
#     Tzw = ss(Azw, Bzw, Czw, 0)

#     ss(Ac,Bc,Cc,0), Tzw, √ν
# end
