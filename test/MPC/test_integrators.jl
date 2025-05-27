using DyadControlSystems
using DyadControlSystems.MPC
using DyadControlSystems.LowLevelParticleFilters
using OrdinaryDiffEq
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Plots

function quadtank(h,u,p=nothing,t=nothing)
    kc = 0.5
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a1, a3, a2, a4= 0.03, 0.03, 0.03, 0.03
    γ1, γ2 = 0.3, 0.3

    ssqrt(x) = √(max(x, zero(x)) + 1e-3)
    # ssqrt(x) = sqrt(x)
    xd = @inbounds SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end


nx = 4
nu = 2
Ts = 2
x0 = SA[1.0,1,1,1]
u0 = SVector(zeros(nu)...)
p = nothing

discrete_dynamics_rk = MPC.rk4(quadtank, Ts)
alg = RK4()
discrete_dynamics = MPC.MPCIntegrator(quadtank, ODEProblem, alg; Ts, nx, nu, p, maxiters=typemax(Int), dt=Ts, adaptive=false, kwargshandle=KeywordArgError)

x1_rk = discrete_dynamics_rk(x0, u0, p, 0)
x1 = discrete_dynamics(x0, u0, p, 0)
@test norm(x1 - x1_rk) < 1e-12

x1 = discrete_dynamics(x0, u0)
@test norm(x1 - x1_rk) < 1e-12



# @btime discrete_dynamics($x0, $u0, 0, 0);
# 1.533 μs (69 allocations: 6.47 KiB)
# 593.511 ns (19 allocations: 1.70 KiB) # StaticArray from dynamics
# 230.032 ns (2 allocations: 96 bytes)  # StaticArray x0
# 165.412 ns (3 allocations: 128 bytes) # dt=Ts, adaptive=false


# @btime $(discrete_dynamics_rk)($x0, $((u0)), 0, 0);
# 327.743 ns (15 allocations: 1.41 KiB)
# 80.872 ns (0 allocations: 0 bytes) # StaticArray

## Rolllout
N = 10
u = rand(nu, N)
x = ones(nx, N+1)
x_rk = ones(nx, N+1)
MPC.rollout!(discrete_dynamics_rk, x_rk, u, p, 0)
MPC.rollout!(discrete_dynamics, x, u, p, 0)

if isinteractive()
    plot(x_rk', lab="RK4", layout=nx)
    plot!(x', lab="Tsit5", layout=nx)
end

@test norm(x - x_rk) < 1e-12

# @btime MPC.rollout!($discrete_dynamics_rk, $x_rk, $u, 0); # 866.485 ns (10 allocations: 480 bytes)
# @btime MPC.rollout!($(Val(nx)), $(Val(nu)), $discrete_dynamics_rk, $x_rk, $u, 0);# 870.393 ns (4 allocations: 160 bytes)
# @btime MPC.rollout!($discrete_dynamics, $x, $u, 0); # 1.397 μs (1 allocation: 32 bytes)
# 37.721 μs (485 allocations: 29.19 KiB) if dt=Ts, adaptive=false

## Linearize

Ark,Brk = MPC.linearize(discrete_dynamics_rk, x0, u0, p, 0)
A,B = MPC.linearize(discrete_dynamics, x0, u0, p, 0)

@test norm(A-Ark) < 1e-6
@test norm(B-Brk) < 1e-6

# @btime MPC.linearize($discrete_dynamics_rk, $x0, $u0, 0, 0) # 408.275 ns (0 allocations: 0 bytes)
# @btime MPC.linearize($discrete_dynamics, $x0, $u0, 0, 0) # 24.837 μs (296 allocations: 21.58 KiB)
# 22.102 μs (355 allocations: 40.28 KiB)
# 4.778 μs (92 allocations: 13.38 KiB) # set_u! instead of reinit!
# 1.002 μs (14 allocations: 2.56 KiB) with dt=Ts, adaptive=false
# 1.214 μs (10 allocations: 1.98 KiB) with dt=Ts, adaptive=false and SVector states


# @btime MPC.Ajac!($A, $discrete_dynamics, $x0, $u0, 0, 0);
# @btime MPC.Bjac!($B, $discrete_dynamics, $x0, $u0, 0, 0);

## DAE
using OrdinaryDiffEq
using ForwardDiff
function stiffdyn(x, u, p, t)
    y₁, y₂, y₃ = x
    u1, u2 = u
    k₁, k₂, k₃ = p
    SA[
        -k₁*y₁ + k₃*y₂*y₃ + u1
        k₁*y₁ - k₃*y₂*y₃ - k₂*y₂^2 + u2
        y₁ + y₂ + y₃ - 1
    ]
end
function daedyn(x, p, t)
    y₁, y₂, y₃ = x
    u1 = u2 = 0
    k₁, k₂, k₃ = p
    SVector(
        -k₁*y₁ + k₃*y₂*y₃ + u1,
        k₁*y₁ - k₃*y₂*y₃ - k₂*y₂^2 + u2,
        y₁ + y₂ + y₃ - 1,
    )
end

nu = 2
nx = 3
x0 = [1.0, 0, 0]
u0 = zeros(nu)
M = [1. 0  0
     0  1. 0
     0  0  0]

Ts2 = 0.01
p = [0.04, 3e7, 1e4]

stifffun = ODEFunction(stiffdyn, mass_matrix = M)
# stifffun = ODEFunction(MPC.control_input_wrapper(stiffdyn, nu), mass_matrix = M)



ddyn5 = MPC.MPCIntegrator(stifffun, ODEProblem, Rodas5P(); Ts=Ts2, nx, nu, p, unstable_check=(args...)->false)
# this PR https://github.com/SciML/OrdinaryDiffEq.jl/pull/1783/files introduced additional error checks that fail and cannot easily be ignored. The tests still pass though

stiffdyn(x0, u0, p, 0)
ddyn5(x0, u0, p, 0)


N        = 200
x        = zeros(3, N+1)
x[:, 1] .= x0
u        = zeros(nu, N)

MPC.rollout!(ddyn5, x, u, p)

function solvedae(x0)
    tspan2    = (0.0, N*Ts2)
    stifffun  = ODEFunction(daedyn, mass_matrix = M)
    probstiff = ODEProblem(stifffun, x0, tspan2, p)
    sol       = solve(probstiff, Rodas5(), saveat = Ts2, abstol=1e-8, reltol=1e-8)
    sol
end

sol = solvedae(x0)
if isinteractive()
    plot(sol, layout=nx)
    plot!(sol.t, x', l=:dash)
    # plot!(u', sp=(nx+1:nx+nu))
end

@test Array(sol) ≈ x


## DAE Diff
using FiniteDiff
A,B = MPC.linearize(ddyn5, x0, u0, p, 0)

Afd = FiniteDiff.finite_difference_jacobian(x->ddyn5(x, u0, p, 0), x0)
Bfd = FiniteDiff.finite_difference_jacobian(u->ddyn5(x0, u, p, 0), u0)

Afd = FiniteDiff.finite_difference_jacobian(x->ddyn5(x, u0, p, 0), x0)
Bfd = FiniteDiff.finite_difference_jacobian(u->ddyn5(x0, u, p, 0), u0)

@test A ≈ Afd atol=1e-2
@test B ≈ Bfd rtol=1e-2



## Pendulum DAE
# using DyadControlSystems
# using DyadControlSystems.MPC
# using OrdinaryDiffEq

# # https://courses.seas.harvard.edu/courses/am205/g_act/DAE_slides.pdf
# function pend(state, f, p=0, t=0)
#     x,y,u,v,λ = state
#     g = 9.82
#     SA[
#         u
#         v
#         -λ*x + f[1]
#         -λ*y - g + f[2]
#         # x^2 + y^2 - 1
#         x*u + y*v
#         # u^2 + v^2 − λ*(x^2 + y^2) − g*y + x*f[1] + y*f[2]
#     ]
# end
# function pend_nocontrol(state, p=0, t=0)
#     x,y,u,v,λ = state
#     g = 9.82
#     SA[
#         u
#         v
#         -λ*x
#         -λ*y - g
#         # x^2 + y^2 - 1
#         x*u + y*v
#         # u^2 + v^2 − λ*(x^2 + y^2) − g*y
#     ]
# end

# nu = 2
# nx = 5
# ny = 2
# # x0 = [1.0, 0, 0, 0, 0]
# x0 = [1/sqrt(2) , -1/sqrt(2), 0, 0, 6.98]
# # x0 = [0.0, -1, 0, 0, 0]
# u0 = zeros(nu)
# M = Matrix(1.0*I(nx))
# M[end] = 0
# Ts = 0.05

# pendfun = ODEFunction(pend, mass_matrix = M)
# pendfun_nc = ODEFunction(pend_nocontrol, mass_matrix = M)

# dprob = ODEProblem(pendfun_nc, x0, (0.0, 3))
# sol = solve(dprob, Rodas5())#, force_dtmin=true, dtmin=1e-2)
# plot(sol, layout=5)



# p = nothing
# dpend = MPC.MPCIntegrator(pendfun, ODEProblem, Rodas5(); Ts, nx, nu, force_dtmin=true, dtmin=1e-3)

# # dpend(x0, u0, 0, 0)


# state_ns = [:x, :y, :u, :v, :λ]
# measurement = (x,u,p,t)->SA[x[1], x[2]]
# discrete_dynamics = FunctionSystem(dpend, measurement, Ts; x=state_ns, u=:u^2, y=state_ns[1:2], p)


# # Control limits
# umin = -10 * ones(nu)
# umax = 10 * ones(nu)

# # State limits (state constraints are soft by default)
# xmin = nothing 
# xmax = nothing
# constraints = NonlinearMPCConstraints(; umin, umax, xmin, xmax)

# xr = zeros(nx)
# xr[2] = -1
# xr[5] = 9.82
# ur = zeros(nu)

# N = 10
# Q1 = 10.0I(nx)
# Q1[end] = 0 # Do not penalize the algebraic state
# Q2 = I(nu)
# qs = 100
# qs2 = 100000

# R1 = 1e-5*I(nx)
# R2 = I(ny)

# # QN,A,B = MPC.calc_QN_AB(Q1,Q2,nothing,dpend,xr,ur)

# solver = OSQPSolver(
#     eps_rel = 1e-6,
#     eps_abs = 1e-6,
#     max_iter = 15000,
#     check_termination = 5,
#     sqp_iters = 1, 
#     dynamics_interval = 1,
#     verbose=true,
#     polish=false, 
# )



# kf = let
#     A,B = DyadControlSystems.linearize(dpend, xr, ur, p, 0)
#     C,D = DyadControlSystems.linearize(measurement, xr, ur, p, 0)
#     KalmanFilter(A, B, C, D, R1, R2)
# end
# # ekf = ExtendedKalmanFilter(
# #     kf,
# #     discrete_dynamics.dynamics,
# #     discrete_dynamics.measurement,
# # )
# prob = QMPCProblem(discrete_dynamics;Q1,Q2,qs,qs2,constraints,N,xr,ur,solver,p,QN=1Q1);

# @time hist = MPC.solve(prob; x0, T = 50, verbose = false, noise=0, dyn_actual=discrete_dynamics);#, sqp_callback)
# plot(hist, plot_title="DAE MPC with known state")


# X,E,R,U,Y,UE = reduce(hcat, hist)

# @test norm((X[:, end] - xr)[1:4]) < 0.02


# ## using DAEUnscentedKalmanFilter

# function g_(x, z, u, p, t)
#     x,y,u,v,λ = state
#     g = 9.82
#     SA[
#         x*u + y*v
#     ]
# end

# get_x_z(xz) = xz[1:4], xz[5:5]
# build_xz(x, z) = [x; z]
# xz0 = x0


# ukf = UnscentedKalmanFilter(discrete_dynamics, R1, R2)
# daeukf = LowLevelParticleFilters.DAEUnscentedKalmanFilter(ukf; g=g_, get_x_z, build_xz, xz0, nu)

# prob = QMPCProblem(discrete_dynamics; observer=daeukf, Q1,Q2,qs,qs2,constraints,N,xr,ur,solver);

# @time hist = MPC.solve(prob; x0, T = 50, verbose = false, noise=0, dyn_actual=discrete_dynamics);
# plot(hist, plot_title="DAE MPC with estimated state")


# X,E,R,U,Y,UE = reduce(hcat, hist)

# @test norm((X[:, end] - xr)[1:4]) < 0.02
