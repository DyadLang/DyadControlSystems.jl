import DynamicPolynomials
using DynamicPolynomials: @polyvar, monomials
using LinearAlgebra
using Symbolics # For build_function
using Kronecker
using ChebyshevApprox
using FillArrays

struct DecomposedDynamics
    x
    u
    dx
end


function decompose_poly(f, nx, nu)
    @polyvar x[1:nx]
    @polyvar u[1:nu]
    dx = f(x, u)
    dyn = DecomposedDynamics(x, u, dx)
    decompose_poly(dyn, nx, nu)
end

function decompose_poly(dyn::DecomposedDynamics, nx, nu)
    x, u, dx = dyn.x, dyn.u, dyn.dx
    @assert length(x) == nx
    @assert length(u) == nu
    function countmap(c)
        d = Dict{eltype(c), Int}()
        for ci in c
            d[ci] = get(d, ci, 0) + 1
        end
        d
    end
    # vars = [x; u]
    # monos_x = monomials(x, 1:2)
    A = map(Iterators.product(1:nx, 1:nx)) do (i,j)
        DynamicPolynomials.coefficient(dx[i], x[j])
    end
    B = map(Iterators.product(1:nx, 1:nu)) do (i,j)
        DynamicPolynomials.coefficient(dx[i], u[j])
    end
    high_monos_u = monomials(u, 2:(nu > 3 ? 2 : 5))
    for mon in high_monos_u
        for i = 1:nx
            DynamicPolynomials.coefficient(dx[i], mon) == 0 ||
                error("All inputs u must enter linearly (input affine dynamics)")
        end
    end
    maxdeg = maximum(maximum(DynamicPolynomials.degree.(monomials(dx))) for dx in dx)
    As = map(2:maxdeg) do d
        all_monos = kron(fill(x, d)...)
        cm = countmap(all_monos)
        # ns = fill(1:nx, d)
        reshape(map(Iterators.product(1:nx, all_monos)) do (i, xx)
            1/cm[xx]*DynamicPolynomials.coefficient(dx[i], xx)
        end, nx, :)
    end
    A, B, As
end


struct SuperKronecker{T, MT} <: AbstractMatrix{T}
    X::MT
    deg::Int
end

SuperKronecker(X::AbstractMatrix{T}, deg::Integer) where T = SuperKronecker{T, typeof(X)}(X, deg)

Base.eltype(A::SuperKronecker{T}) where T = eltype(T)
function Base.size(A::SuperKronecker)
    n,m = size(A.X)
    d = A.deg
    (m^(d-1)*n, m^d)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::SuperKronecker, x::AbstractVecOrMat)
    d = A.deg
    X = A.X
    n = size(X, 2)
    local out
    for i = 1:d
        preI = Eye(i > 1 ? n^(i-1) : 0)
        postI = Eye(i < d ? n^(d-i) : 0) # i+1:d
        if i == 1
            out = kronecker(X, postI)*x
            y .= out
        elseif d == i
            mul!(out, kronecker(preI, X), x)
            y .+= out
        else
            mul!(out, kronecker(preI, X, postI), x)
            y .+= out
        end
    end
    y
end

Base.:(*)(A::SuperKronecker, B::AbstractMatrix) = mul!(similar(B, size(A, 1), size(B, 2)), A, B)

using IterativeSolvers

function Base.:\(A::SuperKronecker, B::AbstractVecOrMat)
    gmres(A,B) # Also works
    # d = A.deg
    # X = A.X
    # T, U = schur(X)
    # n = size(X, 2)
    # B = kronecker(U, d) \ B
    # LT = SuperKronecker(T, d)
    # out = gmres(LT, B)#, abstol=1e-10, reltol=1e-6)
    # kronecker(U', d) \ out
end


function LL(X, d)
    d > 1 || error("d must be > 1")
    SuperKronecker(X, d)
end

function LLnaive(X, d) # For testing purposes
    d > 1 || error("d must be > 1")
    n = size(X, 2)
    local out
    X = sparse(X)
    for i = 1:d
        preI = I(i > 1 ? n^(i-1) : 0)
        postI = I(i < d ? n^(d-i) : 0) # i+1:d
        if i == 1
            out = kron(X, postI)
        elseif d == i
            out += kron(preI, X)
        else
            out += kron(preI, X, postI)
        end
    end
    out
end

struct PQRSolution{T}
    Ks::Vector{Matrix{T}}
    Vs::Vector{Vector{T}}
    A
    B
    As
    Q
    R
end

"""
    sol = pqr(f, Q, R, deg = 3; verbose)
    sol = pqr(A, B, As, Q, R, deg = 3; verbose)

Polynomial-quadratic control design for systems of the form

    x' = f(x, u) = Ax + Bu + N[1]*kron(x, x) + N[2]*kron(x, x, x) + ...

where `N[i]` is a matrix. The cost function is

    J = x'Qx + u'Ru

Ref: "On Approximating Polynomial-Quadratic Regulator Problems", Jeff Borggaard, Lizette Zietsman

# Arguments:
- `f`: Function `f(x, u)` for the dynamics. IF a function is supplied, it will be decomposed into the required Kronecker form automatically using `decompose_poly`.
- `A`: Linear dynamics matrix
- `B`: Control input matrix
- `As`: Array of matrices for nonlinearities
- `Q`: State cost matrix
- `R`: Control input cost matrix
- `deg`: Desired degree of the controller
- `verbose`: Print progress. Defaults to true for systems with more than 10 states.

# Returns
A `PQRSolution` object with fields
- `Ks`: Array of matrices `K[i]` for the controller
- `Vs`: Array of matrices `V[i]` for the Lyapunov function

This solution object supports the following functions:
- [`build_K_function`](@ref)
- [`predicted_cost`](@ref)

# Example
Controlled Lorenz Equations, example 5.1 from the paper referenced above
```julia
using DyadControlSystems, Test

nx = 3 # Number of state variables
nu = 1 # Number of control inputs
A = [
    -10 10 0
    28 -1 0
    0 0 -8/3
]
B = [1; 0; 0;;]
A2 = zeros(nx, nx^2)
A2[2,3] = A2[2,7] = -1/2
A2[3,2] = A2[3,4] = 1/2

Q = diagm(ones(3)) # State cost matrix
R = diagm(ones(1)) # Control input cost matrix

f = (x, u) -> A * x + B * u + A2 * kron(x, x) # Dynamics for simulation
l = (x, u) -> dot(x, Q, x) + dot(u, R, u)     # Cost function for simulation

function closed_loop(xc, (f,l,K), t)          # Closed-loop dynamics for simulation
    x = xc[1:end-1] # Last value is integral of cost
    u = K(x)
    dx = f(x, u)
    dc = l(x, u)
    [dx; dc]        # Return state and cost derivatives
end

x0 = Float64[10, 10, 10, 0] # Initial state, last value is integral of cost
tspan = (0.0, 50.0)
prob = ODEProblem(closed_loop, x0, tspan, (f, l, x->x))

# Design controller of degree 2
pqrsol = pqr(A, B, [A2], Q, R, 2) # It's possible to pass in f instead of A, B, A2
K, _ = build_K_function(pqrsol)
c = predicted_cost(pqrsol, x0[1:end-1])
@test c ≈ 7062.15 atol=0.02 # Compare to paper
sol = solve(prob, Rodas5P(), p=(f,l,K), reltol=1e-8, abstol=1e-8);
c = sol.u[end][end]
@test c ≈ 6911.03 rtol=1e-2 # Compare to paper

# Design controller of degree 4
pqrsol = pqr(A, B, [A2], Q, R, 4)
K, _ = build_K_function(pqrsol)
c = predicted_cost(pqrsol, x0[1:end-1])
@test c ≈ 6924.27 atol=0.02
sol = solve(prob, Rodas5P(), p=(f,l,K), reltol=1e-8, abstol=1e-8);
c = sol.u[end][end]
@test c ≈ 6906.21 rtol=1e-2
```
"""
function pqr(A, B, As, Q, R, deg::Integer=3; verbose = size(A, 1) > 10 || deg >= 6)
    nx = size(A, 1)
    nu = size(R, 1)
    verbose && @info("Solving ARE")
    k1, V2 = safe_lqr(Continuous, Matrix(A), B, Matrix(Q), R)
    @. k1 = -k1 # Positive feedback
    Ac = A + B*k1
    v2 = vec(V2)
    r2 = vec(R)

    v = [zeros(0), v2]
    k = [k1]

    degN = length(As)+1

    N = [As[1], As...]
    BKN = similar(N, deg)

    bb = -(LL(N[2]', 2)*v[2])
    lhs = LL(Ac', 3)
    vd1 = lhs \ bb
    push!(v, vd1)

    res = zeros(nx^2,nu)
    for i=1:nu
      res[:,i] = -(LL(B[:,i]', 3)*v[3])
    end
    kd = 0.5*(R\res')

    push!(k, kd)

    for d = 3:deg
        verbose && @info("Computing Lyapunov coefficients of degree $(d+1)")
        vd = v[end]

        if degN>d-1
            bb = -(LL(N[d]', 2)*v[2])
        else
            bb = zeros(nx^(d+1))
        end

        if degN > d-2
            BKN[d-1] = B*k[d-1]+N[d-1]
        else
            BKN[d-1] = B*k[d-1]
        end

        for i=3:d
            bb .-= LL(BKN[d+2-i]', i)*v[i]
        end

        for i=2:(d ÷ 2)
            tmp = k[i]'*R*k[d+1-i]
            bb  .-= vec(tmp) + vec(tmp')
        end

        if mod(d,2) != 0 # if d is odd
            tmp = k[(d+1) ÷ 2]'*R*k[(d+1) ÷ 2];
            bb .-= vec(tmp)
        end

        lhs = LL(Ac', d+1)
        vd1 = lhs \ bb
        push!(v, vd1)

        verbose && @info("Computing feedback gain for degree $d")
        res = zeros(nx^d,nu)
        for i=1:nu
          res[:,i] = -(LL(B[:,i]', d+1)*v[d+1])
        end
        kd = 0.5*(R\res')
        push!(k, kd)
        
    end
    PQRSolution(k, v[2:end], A, B, As, Q, R)
end

Kfun(sol::PQRSolution, args...) = Kfun(sol.Ks, args...)
function Kfun(Ks, d=length(Ks))
    function (x)
        X = x
        local u
        for i = 1:d
            if i == 1
                u = Ks[i]*X
            else
                mul!(u, Ks[i], X, true, true)
            end
            if i < d
                X = kron(X, x)
            end
        end
        u
    end
end

# Convenience function for decomposing a function into the required form
function pqr(f, Q, R, deg::Integer=2)
    nx = LinearAlgebra.checksquare(Q)
    nu = LinearAlgebra.checksquare(R)
    local A, B, As
    try
        A, B, As = decompose_poly(f, nx, nu)
    catch e
        @error("Failed to aumtomatically decompose the dynamics function as a polynomial-affine system. Make sure that the dynamics is polynomial in the state, and linear in the control inputs and is a function with signature f(x, u).")
        rethrow()
    end
    isempty(As) && error("No higher-order terms found, use lqr/are instead")
    pqr(A, B, As, Q, R, deg)
end

"""
    K_oop, K_ip = build_K_function(sol::PQRSolution)
    K_oop, K_ip = build_K_function(Ks)

Build a function for the controller `K(x)` obtained from [`pqr`](@ref) for the given `Ks` or `PQRSolution`.

Optionally, the degree of the controller can be specified as the second argument, provided that the chosen degree is no higher than the degree for which the solution `sol` was obtained from [`pqr`](@ref).

Keyword arguments are forwarded to `Symbolics.build_function`. Enabling common-subexpression elimination through `cse=true` might be useful for large controllers.
"""
function build_K_function(Ks, deg=length(Ks); simplify = deg < 8, kwargs...)
    K = Kfun(Ks, deg)
    nx = size(Ks[1], 2)
    Symbolics.@variables x[1:nx]
    x = collect(x)
    u = K(x)
    simplify && (u = Symbolics.simplify.(u))
    Symbolics.build_function(u, x; expression=Val{false}, force_SA=true, kwargs...)
end
build_K_function(sol::PQRSolution, args...; kwargs...) = build_K_function(sol.Ks, args...; kwargs...)

"""
    predicted_cost(sol::PQRSolution, x)

Compute the infinite-horizon cost from state `x` predicted using the approximate value function (an approximate Lyapunov function) `V(x)` obtained from [`pqr`](@ref).
"""
predicted_cost(sol::PQRSolution, x) = predicted_cost(sol.Vs, x)
function predicted_cost(Vs, x)
    X = kron(x, x)
    c = 0.0
    for i = eachindex(Vs)
        c += Vs[i]'*X
        if i < length(Vs)
            X = kron(X, x)
        end
    end
    c
end

"""
    poly_approx(f, deg, l, u; N = 1000)

Approximate scalar function `f` in the least-squares sense
on the interval `[l, u]` using a polynomial of degree `deg`.

Returns a vector of polynomial coefficients of length `deg + 1` starting at order 0.

# Arguments:
- `f`: Scalar function
- `deg`: Polynomial degree
- `l`: Lower bound of the domain
- `u`: Upper bound of the domain
- `N`: Number of discretization points

# Example
The following creates 5:th order approximations of sin and cos on the interval `[0, 2pi]`, and extends the valid domain to all real numbers using periodicity (`mod`).
```
using DynamicPolynomials
const sinpol = tuple(DyadControlSystems.poly_approx(sin, 5, 0, 2pi)...)

function sin5(x)
    x = mod(x, 2pi) # Extend the domain to all real numbers
    evalpoly(x, sinpol)
end

sin5(x::Union{PolyVar, Polynomial}) = evalpoly(x, sinpol) # When evaluated using a polynomial variable, we do not include the call to mod

function cos5(x)
    sin5(x+pi/2)
end
```
"""
function poly_approx(f, deg, l, u; N = 1000)
    t = nodes(N, :chebyshev_nodes, [float(u), float(l)]).points
    A = t .^ (0:deg)'
    x = A \ f.(t)
    n = norm(x)
    x[abs.(x) .< 10eps(n)] .= 0
    x
end


"""
    poly_approx(f; deg = 3, lx::AbstractVector, ux::AbstractVector, lu, uu, N = 100000, verbose = true)

Polynomial approximation of dynamics `ẋ = f(x, u)` in the least-squares sense.

# Arguments:
- `deg`: An integer or a vector of the same length as `x` specifying the maximum degree of each state variable. Defaults to 3.
- `lx`: Lower bounds of the state domain over which to approximate the dynamics.
- `ux`: Upper bounds of the state domain over which to approximate the dynamics.
- `lu`: Lower bounds of the input domain over which to approximate the dynamics.
- `uu`: Upper bounds of the input domain over which to approximate the dynamics.
- `N`: Number of samples to use for the approximation.
- `verbose`: Print maximum error after fitting?
"""
function poly_approx(f; deg=3, lx::AbstractVector, ux::AbstractVector, lu, uu, N = 100000, verbose = true)
    nx = length(lx)
    nu = length(lu)
    @polyvar x[1:nx] u[1:nu]

    if deg isa AbstractVector
        filter = function (m)
            all(DynamicPolynomials.degree(m, x[i]) <= deg[i] for i in 1:nx)
        end
    else
        filter = m -> true
    end
    monos_x = monomials(x, 0:maximum(deg), filter)
    monos_u = monomials(u, 1)
    monos = [monos_x; monos_u]
    A = ones(N, length(monos))
    uinds = (1:nu) .+ length(monos_x)
    chebs_x = [shuffle(nodes(N,:chebyshev_nodes, [float(u), float(l)]).points) for (u, l) in zip(ux, lx)]
    chebs_u = [shuffle(nodes(N,:chebyshev_nodes, [float(u), float(l)]).points) for (u, l) in zip(uu, lu)]
    ru = zeros(nu)
    rx = zeros(nx)
    y = zeros(N, nx)
    # dx = ux - lx
    # du = uu - lu
    for i = 1:N
        # rand!(rx)
        # rx .= rx .* dx .+ lx
        # rand!(ru)
        # ru .= ru .* du .+ lu
        for j = 1:nx
            rx[j] = chebs_x[j][i]
        end
        for j = 1:nu
            ru[j] = chebs_u[j][i]
        end
        y[i,:] = f(rx, ru)'
        Threads.@threads for j = 1:length(monos_x)
            A[i,j] = monos_x[j](rx)
        end
        A[i,uinds] .= reverse(ru) # The reverse is due to the Lexical ordering introduced in DynamicPolynomials v0.5
    end
    @assert count(==(1), A) == N
    w = \(A, y)
    verbose && @info "Maximum approximation error over domain: $(maximum(abs, A*w - y)))"
    n = norm(w)
    w[abs.(w) .< 1eps(n)] .= 0
    dx = w'monos
    DecomposedDynamics(x, u, dx)
end