using Convex
using ControlSystemsBase.MatrixEquations

abstract type InverseLQRMethod end

"""
    LMI <: InverseLQRMethod

`R` is a weighting matrix, determining the relative importance of matching each input to that provided by the original controller.
"""
Base.@kwdef struct LMI <: InverseLQRMethod
    R = nothing
end

"""
    Eq29 <: InverseLQRMethod

`R` is a weighting matrix, determining the relative importance of matching each input to that provided by the original controller.
"""
Base.@kwdef struct Eq29 <: InverseLQRMethod
    R = I
end

"""
    Eq32 <: InverseLQRMethod
    Eq32(q1, r1)

`r1` penalizes deviation from the original control signal while `q1` penalizes deviation from the original effect the cotrol signal had on the state.
In other words, the `r1` term tries to use the same actuator configuration as the original controller, while the `q1` term ensures that the effec of the controller is the same.
"""
struct Eq32 <: InverseLQRMethod
    q1
    r1
end

Eq32(q1) = Eq32(q1, I)


"""
    Q,R,S,P,L2 = inverse_lqr(method::LMI, G::AbstractStateSpace, L; optimizer = Hypatia.Optimizer)

Solve the inverse optimal control problem that finds the LQR cost function that leads to a controller approximating state feedback controller with gain matrix `L`. `P` is the solution to the Riccati equation and `L2` is the recreated feedback gain. Note, `L` is supposed to be used with negative feedback, i.e., it's designed such that `u = -Lx`.

Solves a convex LMI problem minimizing the cross term `S`. Note: there is no guarantee that the corss term will be driven to 0 exactly. If `S` remains large in relation to `Q` and `R`, `S` must be included in the cost function for high fidelity reproduction of the original controller.

Ref: "Designing MPC controllers by reverse-engineering existing LTI controllers", E. N. Hartley, J. M. Maciejowski
"""
function inverse_lqr(method::LMI, G::AbstractStateSpace, L; optimizer = Hypatia.Optimizer)
    nx,nu = G.nx, G.nu
    A,B = ssdata(G)
    Q = Convex.Semidefinite(nx)
    P = Convex.Semidefinite(nx)
    S = Convex.Variable(nx, nu)
    prob = Convex.minimize(Convex.sumsquares(S))
    if method.R === nothing
        R = Convex.Semidefinite(nu)
        prob.constraints += R ⪰ I(nu)
    else
        R = method.R
    end
    L = -L # CS requires negative feedback but paper assumes positive.
    if ControlSystemsBase.isdiscrete(G)
        # A'PA - P - (A'PB+S)(R+B'PB)^(-1)(B'PA+S') + Q = 0
        U = (B'P*B + R)*L
        prob.constraints += A'P*A - P - L'*U + Q == 0
        prob.constraints += U + (B'P*A + S') == 0
    else
        # A'P + PA - (PB+S)R^(-1)(B'P+S') + Q = 0
        U = R*L
        prob.constraints += A'P + P*A - L'*U + Q == 0
        prob.constraints += U + (B'P + S') == 0
    end
    # opt = SCS.Optimizer(max_iters=1_000_000, eps=1e-8)
    Convex.solve!(prob, optimizer, warmstart=false, silent=false)
    satisfied = Int(prob.status) == 1
    satisfied || @error("Failed to find a feasible solution")

    mat(x::Number) = [x;;]
    mat(x::AbstractArray) = x

    Q,R,S,P = Convex.evaluate.((Q,R,S,P))
    Q,R,S,P = mat.((Q,R,S,P)) # Convex do not differ between scalars and 1x1 matrices :/
    Q,R,S,P = Symmetric(Q),Symmetric(R),S,Symmetric(P)
    fun = ControlSystemsBase.isdiscrete(G) ? MatrixEquations.ared : MatrixEquations.arec
    _,_,L2 = fun(G.A, G.B, R, Q, S)
    # L2 .*= -1
    Q,R,S,P,L2
end

"""
    Q,R,S,P,L2 = inverse_lqr(method::Eq29, G::AbstractStateSpace, L)

Solve the inverse optimal control problem that finds the LQR cost function that leads to a controller approximating state feedback controller with gain matrix `L`. `P` is the solution to the Riccati equation and `L2` is the recreated feedback gain. Note: `S` will in general not be zero, and including this cross term in the cost function may be important. If including `S` is not possible, use the [`LMI`](@ref) method to find a cost function with as small `S` as possible.

Creates the stage-cost matrix
```math
\\begin{bmatrix}
L^T R L  &  -L^T R\\\\
-R L  &  R
\\end{bmatrix} = \\begin{bmatrix}
Q  &  -S\\\\
-S^T  &  R
\\end{bmatrix} 
```

Ref: "Designing MPC controllers by reverse-engineering existing LTI controllers", E. N. Hartley, J. M. Maciejowski
"""
function inverse_lqr(method::Eq29, G::AbstractStateSpace, L)
    Q2 = method.R
    Q1 = Symmetric(L'*Q2*L)
    Q12 = L'Q2
    fun = ControlSystemsBase.isdiscrete(G) ? MatrixEquations.ared : MatrixEquations.arec
    P,_,L2 = fun(G.A, G.B, Q2, Q1, Q12)
    Q1,Q2,Q12,P,L2
end

"""
    Q,R,S,P,L2 = inverse_lqr(method::Eq32, G::AbstractStateSpace, L)

Solve the inverse optimal control problem that finds the LQR cost function that leads to a controller approximating state feedback controller with gain matrix `L`. `P` is the solution to the Riccati equation and `L2` is the recreated feedback gain. Note: `S` will in general not be zero, and including this cross term in the cost function may be important. If including `S` is not possible, use the [`LMI`](@ref) method to find a cost function with as small `S` as possible.

Creates the cost function
```math
||Bu - BLu||^2_{q_1} + ||u - Lu||^2_{r_1}
```

Ref: "Designing MPC controllers by reverse-engineering existing LTI controllers", E. N. Hartley, J. M. Maciejowski
"""
function inverse_lqr(method::Eq32, G::AbstractStateSpace, L)
    r1,q1 = method.r1, method.q1
    r1 isa Real && (r1 = r1*I)
    q1 isa Real && (q1 = q1*I)
    B = G.B
    S = r1 + B'q1*B
    Q1 = Symmetric(L'*S*L)
    Q2 = Symmetric(S)
    Q12 = L'S
    fun = ControlSystemsBase.isdiscrete(G) ? MatrixEquations.ared : MatrixEquations.arec
    P,_,L2 = fun(G.A, B, Q2, Q1, Q12)
    Q1,Q2,Q12,P,L2
end


struct GMF <: InverseLQRMethod
    γ
    info
end

GMF((K, γ, info)) = GMF(γ, info)

"""
    Q,R,S,P,L2 = inverse_lqr(method::GMF)

Solve the inverse optimal control problem that finds the LQR cost function on the form
```math
x'Qx + u'Ru
```
that leads to an LQR controller approximating a Glover McFarlane controller.
Using this method, `S = 0`.


Ref: Rowe and Maciejowski, "Tuning MPC using H∞ Loop Shaping" .

# Example
```julia
disc = (x) -> c2d(ss(x), 0.01)
G = tf([1, 5], [1, 2, 10]) |> disc # Plant model
W1 = tf(1,[1, 0])          |> disc # Loop shaping weight
gmf2 = glover_mcfarlane(G; W1) # Design Glover McFarlane controller
Q,R,S,P,L = inverse_lqr(GMF(gmf2)) # Get cost function

using Test
L3 = lqr(G*W1, Q, R) # Test equivalence to LQR
@test L3 ≈ -L2
```
"""
function inverse_lqr(method::GMF)
    @unpack γ, info = method
    @unpack Z,X,Gs,W = info
    A,B,C,D = ssdata(Gs)
    strictly_proper = iszero(info.L)

    XW = X/W
    X∞ = γ^2*XW
    # if strictly_proper
    #     Q = C'C
    # else
    # end
    Q = Symmetric(X∞ - A'*((I + X∞*B*B')\X∞)*A) # Equivalent between Iglesias and Merl paper but slightly different formulations
    R = I(Gs.nu)

    fun = ControlSystemsBase.isdiscrete(Gs) ? MatrixEquations.ared : MatrixEquations.arec
    P,_,L2 = fun(A, B, R, Q)
    L2 .*= -1


    Q,R,0*I,P,L2
end