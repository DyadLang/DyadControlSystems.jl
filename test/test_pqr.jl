using DyadControlSystems
using LinearAlgebra
using Plots
using OrdinaryDiffEq
using StaticArrays
##


## Test poly_approx
@test DyadControlSystems.poly_approx(sin, 3, -pi/12, pi/12) ≈ [0, 1, 0, -0.166] atol=1e-3
@test DyadControlSystems.poly_approx(cos, 3, -pi/12, pi/12) ≈ [1, 0, -0.498, 0] atol=1e-3

f = (x, u) -> [
    sin(x[1]) + u[1]
    cos(x[2]) + u[2]
    sin(x[1])*cos(x[3]) + u[2]
]


deg = 7
nx = 3
nu = 2
ux = fill(pi, nx)
lx = -ux
uu = fill(1, nu)
lu = -uu
dx = ux - lx
du = uu - lu

decomposed = DyadControlSystems.poly_approx(f; deg, ux, lx, uu, lu, verbose=true)
pol = decomposed.dx

residuals = map(1:1000) do i
    rx = rand(nx)
    ru = rand(nu)
    rx .= rx .* dx .+ lx
    ru .= ru .* du .+ lu
    xu = [rx; ru]
    vars = pol[1].x[1].vars
    norm(f(rx, ru) - [pol(xu) for pol in pol])
end

@test maximum(residuals) < 0.2 # NOTE: this test fails on DynamicPolynomials v0.5, couldn't easily figure out why so we remain compatible with 0.4 only for now

# plot([
#     x1->f([x1,0,0], zeros(2))[1],
#     x1->pol[1]([x1,0,0,0,0]),
# ], -pi/2, pi/2)

## 5.1 Controlled Lorenz Equations
nx = 3
nu = 1
A = [
    -10 10 0
    28 -1 0
    0 0 -8/3
]
B = [1; 0; 0;;]
N2 = zeros(nx, nx^2)
N2[2,3] = N2[2,7] = -1/2
N2[3,2] = N2[3,4] = 1/2

Q = diagm(ones(3))
R = diagm(ones(1))


f = (x, u) -> A * x + B * u + N2 * kron(x, x)
l = (x, u) -> dot(x, Q, x) + dot(u, R, u)

@test DyadControlSystems.decompose_poly(f, nx, nu)[1] == A
@test DyadControlSystems.decompose_poly(f, nx, nu)[2] == B
@test DyadControlSystems.decompose_poly(f, nx, nu)[3][1] == N2


function f_cl_pqr(xc, (f,l,K), t)
    x = xc[1:end-1]
    u = K(x)
    dx = f(x, u)
    dc = l(x, u)
    [dx; dc]
end
x0 = Float64[10, 10, 10, 0] # Last value is integral of cost
tspan = (0.0, 50.0)
sol = pqr(A, B, [N2], Q, R, 2)
K = build_K_function(sol)
prob = ODEProblem(f_cl_pqr, x0, tspan, (f, l, K))



# Degree 2
sol = pqr(A, B, [N2], Q, R, 2)
K = DyadControlSystems.Kfun(sol)
c = predicted_cost(sol, x0[1:end-1])
@test c ≈ 7062.15 atol=0.02
sol = solve(prob, Rodas5P(), p=(f,l,K), reltol=1e-8, abstol=1e-8)
c = sol.u[end][end]
@test c ≈ 6911.03 rtol=1e-2

# Degree 3
sol = pqr(A, B, [N2], Q, R, 3)
K = DyadControlSystems.Kfun(sol)
c = predicted_cost(sol, x0[1:end-1])
@test c ≈ 6957.19 atol=0.02
sol = solve(prob, Rodas5P(), p=(f,l,K), reltol=1e-8, abstol=1e-8)
c = sol.u[end][end]
@test c ≈ 6906.45 rtol=1e-2

# Degree 4
sol = pqr(A, B, [N2], Q, R, 4)
K = DyadControlSystems.Kfun(sol)
c = predicted_cost(sol, x0[1:end-1])
@test c ≈ 6924.27 atol=0.02
sol = solve(prob, Rodas5P(), p=(f,l,K), reltol=1e-8, abstol=1e-8)
c = sol.u[end][end]
@test c ≈ 6906.21 rtol=1e-2

# 6.192 ms (112367 allocations: 7.75 MiB)


# Degree 4 with built function
sol = pqr(A, B, [N2], Q, R, 4)
K, _ = build_K_function(sol)
sol = solve(prob, Rodas5P(), p=(f,l,K), reltol=1e-8, abstol=1e-8)
c = sol.u[end][end]
@test c ≈ 6906.21 rtol=1e-2

# 4.856 ms (110688 allocations: 5.25 MiB)


##
@test_skip begin # Until bug in Kronecker.jl is fixed
    f = (x, u) -> -x .+ u .+ 5x.^2
    Q = [1.0;;]
    R = [1.0;;]
    nx = nu = 1
    # DyadControlSystems.decompose_poly(f, nx, nu)
    pqrsol = pqr(f, Q, R, 3)
    K, _ = build_K_function(pqrsol)


    x0 = Float64[10, 0] # Last value is integral of cost
    tspan = (0.0, 10.0)
    prob = ODEProblem(f_cl_pqr, x0, tspan, (f, l, K))

    c = predicted_cost(pqrsol, x0[1:end-1])
    sol = solve(prob, Rodas5P(), p=(f,l,K), reltol=1e-8, abstol=1e-8)
    c2 = sol.u[end][end]
    # plot(sol, layout=nx+1)
    @test abs(sol.u[end][1]) < 1e-5
end

## Test scaling
nx = 10
nu = 4
A = randn(nx,nx)
B = randn(nx,nu)
A2 = randn(nx,nx^2)
Q = diagm(ones(nx))
R = diagm(ones(nu))
@time "large system" pqrsol = pqr(A,B,[A2], Q, R, 4); # nx = 10, nu = 4, degree = 4 used to cause OOM before gmres


## Test SuperKronecker
using DyadControlSystems: SuperKronecker, LLnaive, LL
X = randn(2,2) - 5I
L = LL(X, 2);
B = randn(4)
@test L * B ≈ DyadControlSystems.LLnaive(X, 2) * B

using IterativeSolvers
x = gmres(L, B)
@test x ≈ DyadControlSystems.LLnaive(X, 2) \ B rtol=1e-6

@test L \ B ≈ DyadControlSystems.LLnaive(X, 2) \ B

X = randn(10,10) - 5I # Works for 20x20 as well but takes about 30 seconds
L = SuperKronecker(X, 5);
B = randn(100000)
@time L \ B 

## Test size
X = randn(2,3)
L = LL(X, 2);
L2 = LLnaive(X, 2)
@test size(L) == size(L2)



## Burgers equation
using SparseArrays
to_sparse(A) = sparse(first.(A[:,1]), last.(A[:,1]), identity.(A[:,2]))

function test_burgers()
A_ = to_sparse([
    
   (1,1)     -40.0000
   (2,1)      20.0000
  (20,1)      20.0000
   (1,2)      20.0000
   (2,2)     -40.0000
   (3,2)      20.0000
   (2,3)      20.0000
   (3,3)     -40.0000
   (4,3)      20.0000
   (3,4)      20.0000
   (4,4)     -40.0000
   (5,4)      20.0000
   (4,5)      20.0000
   (5,5)     -40.0000
   (6,5)      20.0000
   (5,6)      20.0000
   (6,6)     -40.0000
   (7,6)      20.0000
   (6,7)      20.0000
   (7,7)     -40.0000
   (8,7)      20.0000
   (7,8)      20.0000
   (8,8)     -40.0000
   (9,8)      20.0000
   (8,9)      20.0000
   (9,9)     -40.0000
  (10,9)      20.0000
   (9,10)     20.0000
  (10,10)    -40.0000
  (11,10)     20.0000
  (10,11)     20.0000
  (11,11)    -40.0000
  (12,11)     20.0000
  (11,12)     20.0000
  (12,12)    -40.0000
  (13,12)     20.0000
  (12,13)     20.0000
  (13,13)    -40.0000
  (14,13)     20.0000
  (13,14)     20.0000
  (14,14)    -40.0000
  (15,14)     20.0000
  (14,15)     20.0000
  (15,15)    -40.0000
  (16,15)     20.0000
  (15,16)     20.0000
  (16,16)    -40.0000
  (17,16)     20.0000
  (16,17)     20.0000
  (17,17)    -40.0000
  (18,17)     20.0000
  (17,18)     20.0000
  (18,18)    -40.0000
  (19,18)     20.0000
  (18,19)     20.0000
  (19,19)    -40.0000
  (20,19)     20.0000
   (1,20)     20.0000
  (19,20)     20.0000
  (20,20)    -40.0000
])


B_ = [
    0.025000000000000                   0   0.025000000000000
    0.050000000000000                   0                   0
    0.050000000000000                   0                   0
    0.050000000000000                   0                   0
    0.050000000000000                   0                   0
    0.050000000000000                   0                   0
    0.046960870783050   0.003039129216950                   0
    0.010150240328061   0.039849759671939                   0
                    0   0.050000000000000                   0
                    0   0.050000000000000                   0
                    0   0.050000000000000                   0
                    0   0.050000000000000                   0
                    0   0.050000000000000                   0
                    0   0.039849759671939   0.010150240328061
                    0   0.003039129216950   0.046960870783050
                    0                   0   0.050000000000000
                    0                   0   0.050000000000000
                    0                   0   0.050000000000000
                    0                   0   0.050000000000000
                    0                   0   0.050000000000000
]

A2_ = to_sparse([
 
(2,1)      0.166666666666667
(20,1)     -0.166666666666667
 (1,2)     -0.333333333333333
 (2,2)     -0.166666666666667
 (1,20)     0.333333333333333
(20,20)     0.166666666666667
 (1,21)     0.166666666666667
 (2,21)     0.333333333333333
 (1,22)    -0.166666666666667
 (2,22)     0.000000000000000
 (3,22)     0.166666666666667
 (2,23)    -0.333333333333333
 (3,23)    -0.166666666666667
 (2,42)     0.166666666666667
 (3,42)     0.333333333333333
 (2,43)    -0.166666666666667
 (3,43)    -0.000000000000000
 (4,43)     0.166666666666667
 (3,44)    -0.333333333333333
 (4,44)    -0.166666666666667
 (3,63)     0.166666666666667
 (4,63)     0.333333333333333
 (3,64)    -0.166666666666667
 (4,64)    -0.000000000000000
 (5,64)     0.166666666666667
 (4,65)    -0.333333333333333
 (5,65)    -0.166666666666667
 (4,84)     0.166666666666667
 (5,84)     0.333333333333333
 (4,85)    -0.166666666666667
 (5,85)    -0.000000000000000
 (6,85)     0.166666666666667
 (5,86)    -0.333333333333333
 (6,86)    -0.166666666666667
 (5,105)    0.166666666666667
 (6,105)    0.333333333333333
 (5,106)   -0.166666666666667
 (6,106)   -0.000000000000000
 (7,106)    0.166666666666667
 (6,107)   -0.333333333333333
 (7,107)   -0.166666666666667
 (6,126)    0.166666666666667
 (7,126)    0.333333333333333
 (6,127)   -0.166666666666667
 (7,127)   -0.000000000000000
 (8,127)    0.166666666666667
 (7,128)   -0.333333333333333
 (8,128)   -0.166666666666667
 (7,147)    0.166666666666667
 (8,147)    0.333333333333333
 (7,148)   -0.166666666666667
 (8,148)   -0.000000000000000
 (9,148)    0.166666666666667
 (8,149)   -0.333333333333333
 (9,149)   -0.166666666666667
 (8,168)    0.166666666666667
 (9,168)    0.333333333333333
 (8,169)   -0.166666666666667
 (9,169)   -0.000000000000000
(10,169)    0.166666666666667
 (9,170)   -0.333333333333333
(10,170)   -0.166666666666667
 (9,189)    0.166666666666667
(10,189)    0.333333333333333
 (9,190)   -0.166666666666667
(10,190)   -0.000000000000000
(11,190)    0.166666666666667
(10,191)   -0.333333333333333
(11,191)   -0.166666666666667
(10,210)    0.166666666666667
(11,210)    0.333333333333333
(10,211)   -0.166666666666667
(11,211)   -0.000000000000000
(12,211)    0.166666666666667
(11,212)   -0.333333333333333
(12,212)   -0.166666666666667
(11,231)    0.166666666666667
(12,231)    0.333333333333333
(11,232)   -0.166666666666667
(12,232)    0.000000000000000
(13,232)    0.166666666666667
(12,233)   -0.333333333333333
(13,233)   -0.166666666666667
(12,252)    0.166666666666667
(13,252)    0.333333333333333
(12,253)   -0.166666666666667
(13,253)    0.000000000000000
(14,253)    0.166666666666667
(13,254)   -0.333333333333333
(14,254)   -0.166666666666667
(13,273)    0.166666666666667
(14,273)    0.333333333333333
(13,274)   -0.166666666666667
(14,274)    0.000000000000000
(15,274)    0.166666666666667
(14,275)   -0.333333333333333
(15,275)   -0.166666666666667
(14,294)    0.166666666666667
(15,294)    0.333333333333333
(14,295)   -0.166666666666667
(15,295)    0.000000000000000
(16,295)    0.166666666666667
(15,296)   -0.333333333333333
(16,296)   -0.166666666666667
(15,315)    0.166666666666667
(16,315)    0.333333333333333
(15,316)   -0.166666666666667
(16,316)    0.000000000000000
(17,316)    0.166666666666667
(16,317)   -0.333333333333333
(17,317)   -0.166666666666667
(16,336)    0.166666666666667
(17,336)    0.333333333333333
(16,337)   -0.166666666666667
(17,337)    0.000000000000000
(18,337)    0.166666666666667
(17,338)   -0.333333333333333
(18,338)   -0.166666666666667
(17,357)    0.166666666666667
(18,357)    0.333333333333333
(17,358)   -0.166666666666667
(18,358)    0.000000000000000
(19,358)    0.166666666666667
(18,359)   -0.333333333333333
(19,359)   -0.166666666666667
(18,378)    0.166666666666667
(19,378)    0.333333333333333
(18,379)   -0.166666666666667
(19,379)    0.000000000000000
(20,379)    0.166666666666667
(19,380)   -0.333333333333333
(20,380)   -0.166666666666667
 (1,381)   -0.166666666666667
(20,381)   -0.333333333333333
(19,399)    0.166666666666667
(20,399)    0.333333333333333
 (1,400)    0.166666666666667
(19,400)   -0.166666666666667
(20,400)    0.000000000000000
])


M = to_sparse([
    (1,1)      0.033333333333333
    (2,1)      0.008333333333333
   (20,1)      0.008333333333333
    (1,2)      0.008333333333333
    (2,2)      0.033333333333333
    (3,2)      0.008333333333333
    (2,3)      0.008333333333333
    (3,3)      0.033333333333333
    (4,3)      0.008333333333333
    (3,4)      0.008333333333333
    (4,4)      0.033333333333333
    (5,4)      0.008333333333333
    (4,5)      0.008333333333333
    (5,5)      0.033333333333333
    (6,5)      0.008333333333333
    (5,6)      0.008333333333333
    (6,6)      0.033333333333333
    (7,6)      0.008333333333333
    (6,7)      0.008333333333333
    (7,7)      0.033333333333333
    (8,7)      0.008333333333333
    (7,8)      0.008333333333333
    (8,8)      0.033333333333333
    (9,8)      0.008333333333333
    (8,9)      0.008333333333333
    (9,9)      0.033333333333333
   (10,9)      0.008333333333333
    (9,10)     0.008333333333333
   (10,10)     0.033333333333333
   (11,10)     0.008333333333333
   (10,11)     0.008333333333333
   (11,11)     0.033333333333333
   (12,11)     0.008333333333333
   (11,12)     0.008333333333333
   (12,12)     0.033333333333333
   (13,12)     0.008333333333333
   (12,13)     0.008333333333333
   (13,13)     0.033333333333333
   (14,13)     0.008333333333333
   (13,14)     0.008333333333333
   (14,14)     0.033333333333333
   (15,14)     0.008333333333333
   (14,15)     0.008333333333333
   (15,15)     0.033333333333333
   (16,15)     0.008333333333333
   (15,16)     0.008333333333333
   (16,16)     0.033333333333333
   (17,16)     0.008333333333333
   (16,17)     0.008333333333333
   (17,17)     0.033333333333333
   (18,17)     0.008333333333333
   (17,18)     0.008333333333333
   (18,18)     0.033333333333333
   (19,18)     0.008333333333333
   (18,19)     0.008333333333333
   (19,19)     0.033333333333333
   (20,19)     0.008333333333333
    (1,20)     0.008333333333333
   (19,20)     0.008333333333333
   (20,20)     0.033333333333333
])


zInit = [
    -0.004164118339155
    0.039892266734870
    0.170471265062191
    0.329747293635168
    0.459014939778335
    0.508316733588974
    0.459014939778335
    0.329747293635168
    0.170471265062191
    0.039892266734870
   -0.004164118339155
    0.001115799665464
   -0.000299080322702
    0.000080521625343
   -0.000023006178669
    0.000011503089335
   -0.000023006178669
    0.000080521625343
   -0.000299080322702
    0.001115799665464
]

α   = 0.3
ϵ = 0.005

A = ϵ*(M\Matrix(A_))  + α*I
B = M\B_
A2 = M\Matrix(A2_)

nu = size(B, 2)
Q = Matrix(M ./ 2)
R = I(nu) / 2

@time "Burgers" pqrsol = pqr(A, B, [A2], Q, R, 3)
cp = predicted_cost(pqrsol, zInit)

@time "Build K Burgers" K, _ = build_K_function(pqrsol; simplify=false) # simplify takes very long time on CI
f = (x, u) -> A * x + B * u + A2 * kron(x, x)
l = (x, u) -> dot(x, Q, x) + dot(u, R, u)


function f_cl_burgers(xc, (f,l,K), t)
    # @show t
    x = xc[1:end-1]
    u = K(x)
    dx = f(x, u)
    dc = l(x, u)
    [dx; dc]
end
x0 = [zInit; 0]
tspan = (0.0, 200.0)
prob = ODEProblem(f_cl_burgers, x0, tspan, (f, l, K))

@time "Solve Burgers" sol = solve(prob, Tsit5(), reltol=1e-5, abstol=1e-5)
c = sol.u[end][end]

(; cp, c, sol, pqrsol)
end

(; cp, c, sol, pqrsol) = test_burgers()

@test c ≈ cp rtol=0.1
@test norm(sol.u[end][1:end-1]) < 1e-4



# ==============================================================================
## === Quadrotor from SumOfSquares.jl ===
# ==============================================================================
# A much longer example: This example will demonstrate
# - How to use ModelingToolkit to model the system
# - How to handle non-polynomial terms
# - How to handle non-affine inputs

using DyadControlSystems
using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkitStandardLibrary: Blocks
using ControlSystemsMTK

sin3(x) = -0.166 * x^3 + x # DyadControlSystems.poly_approx(sin, 3, -pi/12, pi/12)'*(x .^ (0:3))
cos3(x) = -0.498 * x^2 + 1 # DyadControlSystems.poly_approx(cos, 3, -pi/12, pi/12)'*(x .^ (0:3))
gravity = 9.81
gain_u1 = 0.89 / 1.4
d0 = 70
d1 = 17
n0 = 55

@named motor_dynamics = Blocks.FirstOrder(T = 0.001)

# x0 = [1.7, 0.85, 0.8, 1, π/12, π/2]
x0 = [0.85, 1, π/12, π/2]

@parameters t
@variables  u(t)[1:2]=0
u = collect(u)
@variables y(t)=0.85 v(t)=1 ϕ(t)=π/12 ω(t)=π/2


D = Differential(t)

eqs = [
    # D(x[1]) ~ x[3],
    D(y) ~ v,
    # D(x[3]) ~ gain_u1 * sin3(x[5])*motor_dynamics.output.u,
    D(v) ~ -gravity + gain_u1 * cos3(ϕ)*(motor_dynamics.output.u + gravity/gain_u1),
    D(ϕ) ~ ω,
    D(ω) ~ -d0 * ϕ - d1 * ω + n0 * u[2],
    motor_dynamics.input.u ~ u[1]
]

@named model = ODESystem(eqs, t; systems=[motor_dynamics])

quadrotor = FunctionSystem(model, u, [y, ϕ])
(; nx, nu) = quadrotor
##
R = diagm([1.0, 10.0])
Q = ControlSystemsMTK.build_quadratic_cost_matrix(model, u,
    [y => 10, v => 10, ϕ => 1, ω => 0.01]
)

tspan = (0.0, 10.0)

p = float.([ModelingToolkit.defaults(model)[p] for p in quadrotor.p])
x0sim = float.([ModelingToolkit.defaults(model)[p] for p in quadrotor.x])


f = (x,u)->quadrotor(x,u,p,nothing)
l = (x, u) -> dot(x, Q, x) + dot(u, R, u)

A,B,As = DyadControlSystems.decompose_poly(f, nx, nu)
pqrsol = pqr(f, Q, R, 3) 

K, _ = build_K_function(pqrsol)
cl = (x, (f,K), t)->f(x, K(x))
prob = ODEProblem(cl, SVector{5}(x0sim), tspan, (f, K))
sol = solve(prob, Tsit5())
@test norm(sol.u[end]) < 1e-4
time = range(tspan..., 100)
uv = map(time) do t
    x = sol(t)
    K(x)
end
um = reduce(hcat, uv)'

Vt = [predicted_cost(pqrsol, sol(t)) for t in time]
@test maximum(diff(Vt)) < 0 # Lyapunov function decreasing along system trajectories
if isinteractive()
    plot(sol, layout=8, title=permutedims(state_names(quadrotor)))
    plot!(time, um, title="u", sp=(1:2)' .+ nx)
    plot!(time, Vt, title="V(x)", sp=8, yscale=:log10)
end


# ==============================================================================
## === Cartpole ================================================================
# ==============================================================================

# using DynamicPolynomials
# using Symbolics
# const sinpol = tuple(DyadControlSystems.poly_approx(sin, 5, 0, 2pi)...)
# function Base.sin(x::Union{PolyVar, Polynomial})
#     evalpoly(x, sinpol)
# end
# function Base.cos(x::Union{PolyVar, Polynomial})
#     sin(x+pi/2)
# end

function cartpole(x, u, p, t)
    T = promote_type(eltype(x), eltype(u))
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81
    τ = 0.001

    ϕ  = x[2] - π
    qd = x[SA[3, 4]]
    xf = x[5] # Filter state
    s = sin(ϕ)
    c = cos(ϕ)

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = [0 -mp*qd[2]*l*s; 0 0]
    G = [0, mp * g * l * s]
    B = [1, 0]
    qdd = -H \ (C * qd + G - B * xf)
    dxf = (-xf + u[1]) / τ
    return [qd; qdd; dxf]
end

nx = 5
nu = 1


# The cartpole system can be written on the form f(x) + g(x)u, not f(x) + Bu. We can thus not hope to create a good polynomial-affine approximation unless we filter the input

f = (x, u) -> cartpole(x, u, 0, 0)
l = (x, u) -> dot(x, Q, x) + dot(u, R, u)

deg = [1, 5, 2, 2, 1]
ux = [4, 0.4pi, 5, 5, 100]
lx = -ux
uu = fill(10, nu)
lu = -uu
@time "poly approx cartpole" cartpole_pol = DyadControlSystems.poly_approx(f; deg, ux, lx, uu, lu)

R = diagm([0.2])
Q = diagm(Float64[2,1,0.1,1,1e-6])

xr = [0, 0, 0, 0, 0]
A,B,As = DyadControlSystems.decompose_poly(cartpole_pol, nx, nu)
sys = ss(DyadControlSystems.linearize(cartpole, xr, zeros(1), 0, 0)..., I, 0)
L = lqr(sys, Q, R)

@test A ≈ sys.A rtol = 1e-1
@test B ≈ sys.B rtol = 1e-1

@time "PQR Cartpole" pqrsol = pqr(cartpole_pol, Q, R, maximum(deg))

@test pqrsol.Ks[1] ≈ -L rtol=1e-1
# NOTE: The linear term does not correspond to the linearization around a non-zero point


topi(x::Number) = mod(x+pi, 2pi)-pi
topi(x) = [x[1], topi(x[2]), x[3], x[4], x[5]]
##
function f_cl_cartpole(xc, (f,l,K), t)
    x = xc[1:end-1] # Last value is integral of cost
    Δx = topi(x)
    th = 100
    Δu = clamp.(K(Δx), -th, th) # K operates on Δx
    # Δu = -L *Δx # K operates on Δx
    u = Δu #+ ur
    dx = f(x, u)
    dc = l(Δx, Δu)
    [dx; dc]        # Return state and cost derivatives
end
x0 = [
    0
    pi-deg2rad(15)
    0
    0
    0
    0
]
cp = predicted_cost(pqrsol, x0[1:end-1])
tspan = (0.0, 9.0)

prob = ODEProblem(f_cl_cartpole, x0, tspan, (f, l, x->x))

fig = plot(layout=nx+1)
for d = [3, 5]
    @time "build_k_function deg=$d" K, _ = build_K_function(pqrsol, d, simplify=false)
    @time "solve deg=$d" sol = solve(prob, Tsit5(), reltol=1e-5, abstol=1e-5, p=(f, l, K))
    @show c = sol.u[end][end]
    @test norm(topi(sol.u[end][1:end-1])) < 1

    isinteractive() && plot!(sol, layout=nx+1, title=["x" "ϕ" "dx" "dϕ" "u" "cost"], lab="Deg $d")
end
fig

# ==============================================================================
## Handle input saturation =====================================================
# ==============================================================================
# We can "handle" input saturation by using the `clamp` function in the dynamics that is approximated by polynomials. To do this, it's important that we clamp the input filter state and not the input itself
using DyadControlSystems, OrdinaryDiffEq

function cartpole_sat(x, u, p, t)
    T = promote_type(eltype(x), eltype(u))
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81
    τ = 0.001
    th = p

    ϕ  = x[2] - π
    qd = x[SA[3, 4]]
    xf = clamp(x[5], -th, th) # Filter state
    s = sin(ϕ)
    c = cos(ϕ)

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = [0 -mp*qd[2]*l*s; 0 0]
    G = [0, mp * g * l * s]
    B = [1, 0]
    qdd = -H \ (C * qd + G - B * xf)
    dxf = (-xf + u[1]) / τ
    if xf == th
        dxf = min(dxf, zero(dxf)) # Only non-zero derivative if it's decreasing
    elseif xf == -th
        dxf = max(dxf, zero(dxf)) # Only non-zero derivative if it's increasing
    end
    return [qd; qdd; dxf]
end

nx = 5
nu = 1
th = 40 # Input saturation threshold

f = (x, u) -> cartpole_sat(x, u, th, 0)
l = (x, u) -> dot(x, Q, x) + dot(u, R, u)

deg = [1, 5, 4, 2, 5]
ux = [4, 0.4pi, 5, 5, 1.2th]
lx = -ux
uu = fill(10, nu)
lu = -uu
@time "poly approx cartpole_sat" cartpole_sat_pol = DyadControlSystems.poly_approx(f; deg, ux, lx, uu, lu)

R = diagm([2])
Q = diagm(Float64[2,1,0.1,1,1e-6])

xr = [0, 0, 0, 0, 0]
A,B,As = DyadControlSystems.decompose_poly(cartpole_sat_pol, nx, nu)
sys = ss(DyadControlSystems.linearize(cartpole_sat, xr, zeros(1), 0, 0)..., I, 0)
L = lqr(sys, Q, R)

@test A ≈ sys.A rtol = 1e-1
@test B ≈ sys.B rtol = 1e-1

@time "PQR Cartpole" pqrsol = pqr(cartpole_sat_pol, Q, R, 5)

@test pqrsol.Ks[1] ≈ -L rtol=1e-1
# NOTE: The linear term does not correspond to the linearization around a non-zero point


topi(x::Number) = mod(x+pi, 2pi)-pi
topi(x) = [x[1], topi(x[2]), x[3], x[4], x[5]]
##
function f_cl_cartpole_sat(xc, (f,l,K), t)
    x = xc[1:end-1] # Last value is integral of cost
    Δx = topi(x)
    Δu = clamp.(K(Δx), -th, th) # K operates on Δx
    # Δu = -L *Δx # K operates on Δx
    u = Δu #+ ur
    dx = f(x, u)
    dc = l(Δx, Δu)
    [dx; dc]        # Return state and cost derivatives
end
x0 = [
    0
    pi+deg2rad(15) # Avoid the initial chattering close to the downward equilibrium
    0
    0
    0
    0
]
@show cp = predicted_cost(pqrsol, x0[1:end-1])
tspan = (0.0, 9.0)

prob = ODEProblem(f_cl_cartpole_sat, x0, tspan, (f, l, x->x))

fig = plot(layout=nx+1)
for d = [3,5]
    @time "build_k_function deg=$d" K, _ = build_K_function(pqrsol, d, simplify=false)
    @time "solve deg=$d" sol = solve(prob, Tsit5(), reltol=1e-5, abstol=1e-5, p=(f, l, K))
    @show c = sol.u[end][end]
    @test norm(topi(sol.u[end][1:end-1])) < 1

    isinteractive() && plot!(sol, layout=nx+1, title=["x" "ϕ" "dx" "dϕ" "u" "cost"], lab="Deg $d")
end
fig

# ==============================================================================
## Duffing system from https://juliacontrol.github.io/ControlSystemsMTK.jl/dev/batch_linearization/#Gain-scheduling
# ==============================================================================
using DyadControlSystems, OrdinaryDiffEq, DataInterpolations, Statistics
function duffing(state, u, p, t)
    x, v = state
    k, k3, c = p
    [
        v
        -k * x - k3 * x^3 - c * v + 10u[]
    ]
end

p = (k=10, k3=2, c=1)
ω = 15
ζ = 1
F = tf(ω^2, [1, 2ζ*ω, ω^2])
t = 0:0.001:8
r = (0.5sign.(sin.(2pi/6*t)) .* (t .> 1) .+ 0.5)
# plot(r)
rf = lsim(F, r', t).y[:]
# plot(rf)
rf_int = CubicSpline(rf, t)

Q = diagm(Float64[100, 0.1])
R = diagm(Float64[0.001])

pqrsol = pqr((x, u)->duffing(x,u,p,0), Q, R, 3)
K, _ = build_K_function(pqrsol)
function f_cl_duffing(x, p, t)
    xr = [rf_int(t), 0]
    u = K(x - xr)
    duffing(x, u, p, t)
end
x0 = zeros(2)
tspan = (0.0, 8.0)
prob = ODEProblem(f_cl_duffing, x0, tspan, p)
sol = solve(prob, Tsit5())


ue = map(t) do t
    x = sol(t)
    e = x - [rf_int(t), 0]
    K(e)[], e[1]
end
u = first.(ue)
e = last.(ue)
if isinteractive()
    plot(sol, idxs=1, layout=2)
    plot!(t, rf, lab="r", sp=1)
    plot!(t, u, lab="u", sp=2)
    plot!(t, e, lab="e", sp=1)
end
@test mean(abs2, e) < 0.02

# ==============================================================================
## Simple quadrotor with s = sin, c = cos ======================================
# ==============================================================================
using DyadControlSystems, OrdinaryDiffEq
function polyquad(state, u, p, t)
    g = 9.81
    Ku = 0.89 / 1.4
    d0 = 70
    d1 = 17
    n0 = 55
    τ = 0.001
    xf, y, v, ϕ, ω, s, c = state
    [
        (-xf + u[1]) / τ
        v
        -g + Ku * c * xf
        ω
        -d0 * ϕ - d1 * ω + n0 * u[2]
        c*ω
        -s*ω
    ]
end

function quad(state, u, p, t)
    g = 9.81
    Ku = 0.89 / 1.4
    d0 = 70
    d1 = 17
    n0 = 55
    τ = 0.001
    xf, y, v, ϕ, ω = state
    [
        (-xf + u[1]) / τ
        v
        -g + Ku * cos(ϕ) * xf
        ω
        -d0 * ϕ - d1 * ω + n0 * u[2]
    ]
end

ϵ = 1e-6
nx = 5
nu = 2
Q = diagm(Float64[ϵ, 10, 10, 1, 0.01, ϵ, ϵ])
R = diagm(Float64[1, 100])
xr = [0,0,0,0,0,sin(0), cos(0)]
ur = [9.81/(0.89 / 1.4), 0]
fr = (Δx,Δu) -> polyquad(Δx + xr, Δu + ur, 0, 0) # Operates in Δx, Δu coordinates
pqrsol = pqr(fr, Q, R, 3)

# A, B = DyadControlSystems.linearize(polyquad, xr, zeros(nu), 0, 0)
# sys = ss(A,B,I,0)
# L = lqr(sys, Q, R)

function f_cl_polyquad(xc, K, t)
    x = xc[1:end-1]
    xe = [x; sin(x[4]); cos(x[4])]
    Δu = K(xe - xr)
    u = Δu + ur
    dx = quad(x, u, 0, t)
    dc = dot(xe - xr, Q, xe - xr) + dot(Δu, R, Δu)
    [dx; dc]
end
x0 = 5*[0, 0.85, 1, π/12, π/2, 0]
tspan = (0.0, 8.0)
prob = ODEProblem(f_cl_polyquad, x0, tspan)

fig = plot(layout=nx+3)
for deg in [1,3]
    @show deg
    K, _ = build_K_function(pqrsol, deg)
    sol = solve(prob, Tsit5(), p=K)
    @test norm(sol.u[end][2:5] - xr[2:5]) < 0.02

    t = range(tspan..., 300)
    u = map(t) do t
        x = sol(t)[1:end-1]
        xe = [x; sin(x[4]); cos(x[4])]
        K(xe-xr)
    end
    u = reduce(hcat, u)'
    if isinteractive()
        lab = "deg=$deg"
        plot!(sol; title=["xf" "y" "v" "ϕ" "ω" "cost"], lab)
        plot!(t, u; sp=(1:nu)' .+ (nx+1), title=["u1" "u2"], lab)
    end
end
fig
