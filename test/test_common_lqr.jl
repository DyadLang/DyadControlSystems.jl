using ControlSystemsBase, DyadControlSystems, Test, StableRNGs

# double-integrator model
A = [0 1.0; 0 0]
B = [0; 1]
C = [1 0]
D = [0]
sys = ss(A,B,C,D)

# sys = ssrand(1,1,2, proper=true)

Q1 = [1 0; 0 1]
Q2 = [1;;]
K = lqr(sys, Q1, Q2)

C = [1 0; 0 1; 0 0]
D = [0;0;1;;]

systems = [sys, sys] # If we pass the same system twice, we should get the same result as standard LQR

K2, P2, cost = common_lqr(systems, Q1, Q2; ϵ = 1e-6, silent_solver=true)
@test K ≈ K2 rtol=1e-3


## Test how well it scales
rng = StableRNG(0)
systems = [sys*(1 + 0.1rand(rng)) for _ in 1:1000]
@time K2, P2, cost = common_lqr(systems, Q1, Q2; ϵ = 1e-6, silent_solver=true)
@test K ≈ K2 rtol=1e-2
# This takes [ Info: ALMOST_OPTIMAL 5.775526 seconds (17.51 M allocations: 540.408 MiB, 1.21% gc time)


A2 = [0 1.0; 0 0.1]
B2 = [0; 1]
C2 = [1 0]
D2 = [0]
sys2 = ss(A2,B2,C2,D2)
systems = [sys, sys2]
K2,_ = common_lqr(systems, Q1, Q2; ϵ = 1e-5, silent_solver=true)

## Test MCM interface
import MonteCarloMeasurements as MCM
import MonteCarloMeasurements: ±, Uniform, unsafe_comparisons
A = [0 1.0; 0 0 ± 0.01]
B = [0; 1 ± 0.01]
C = [1 0]
D = [0]
usys = ss(A,B,C,D)

K3 = lqr(usys, Q1, Q2)
@test K2 ≈ K3 rtol=2e-1

sys_cl = ss(A - B*K3, B, C, D)
P1, _ = DyadControlSystems.common_lyap(sys_cl, Q1)
@test all(eigvals(P1) .> 0)

systems = RobustAndOptimalControl.sys_from_particles(sys_cl)
P2, _ = DyadControlSystems.common_lyap(systems, Q1)
@test P1 ≈ P2

P3 = lyap(Continuous, (sys.A - sys.B*K)', Q1)
@test P1 ≈ P3 rtol = 0.1


sysd = c2d(sys, 0.1)
Kd = lqr(sysd, Q1, Q2)
Acl = (sysd.A - sysd.B*Kd)
P3d = lyap(Discrete, Acl', Q1)

P2d, _ = DyadControlSystems.common_lyap(Discrete, [Acl', Acl'], Q1)
@test P2d ≈ P3d rtol = 0.01


##
N = 6
P = tf(1.0, [1, 1, 1, 0]) |> ss
ω = MCM.Particles(N, Uniform(0.5, 1.5))
Pu = tf(1.0, [1, ω, ω^2, 0]) |> ss # Create a model with uncertainty in spring stiffness k ~ U(80, 120)
unsafe_comparisons(true) # For the Bode plot to work
Q = diagm(ones(P.nx))
R = [1.0;;]

L = lqr(P, Q, R)
Lu = lqr(Pu, Q, R)

K  = kalman(P, Q, R)
Ku = kalman(Pu, Q, R)

C = observer_controller(P, L, K)
Cu = observer_controller(P, Lu, Ku)


G = extended_gangoffour(Pu, C)
Gu = extended_gangoffour(Pu, Cu)
w = exp10.(LinRange(-2, 2, 200))

for i = 1:N
    @test isstable(RobustAndOptimalControl.sys_from_particles(Gu, i))
end

if isinteractive()
    kwargs = (; ri=false, N)
    figb = bodeplot(G, w; plotphase=false, lab="Nominal design", title=["S" "PS" "CS" "T"], c=1, kwargs...)
    bodeplot!(Gu, w; plotphase=false, lab="Uncertainty-aware design", legend=:bottomleft, c=2, kwargs...)

    kwargs = (; points=true, ylims=(-3,1), xlims=(-2.5,1), markeralpha=0.7, markerstrokewidth=[0.0 1])
    fign = nyquistplot(Pu*C, w; lab="Nominal design", c=1, Ms_circles=[2], unit_circle=true, kwargs...)
    nyquistplot!(Pu*Cu, w; lab="Uncertainty-aware design", legend=:bottomleft, c=2, kwargs...)
    plot(fign, figb)
end


del = [-0.5, 0.5]
As = [[0 3/4 + d; 3/4 - d 0] for d in del]
Q = I(2)
sol = common_lyap(Discrete, As, Q)
@test Int(sol.termination_status) == 2 # INFEASIBLE