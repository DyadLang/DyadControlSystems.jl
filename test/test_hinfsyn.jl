using DyadControlSystems
import RobustAndOptimalControl as ROC
using Hypatia # Hypatia performs better than both SCS and M on this problem.

@testset "Difficult Hinf numerics" begin
    @info "Testing Difficult Hinf numerics"
    ## Test higher prec hinfsyn
    # Example from "A robust numerical method for the γ-iteration in H∞ control"
    a = b = d = n = e2 = 1.0
    e1 = 0
    A = -1.0I(2)
    B1 = [e1 0; 0 e2]
    B2 = ones(2)
    C1 = diagm([a, b])
    C2 = [d n]
    D11 = diagm([0.5, 0.5])
    D12 = [0; 1]
    D21 = [0 1]
    D22 = [0;;]
    P = ss(A,B1,B2,C1,C2,D11,D12,D21,D22)

    K2, γ2 = hinfsyn_lmi(P, opt=Hypatia.Optimizer)
    @test γ2 ≈ 0.806 atol=1e-2
end

## Manually built hinf problem
@testset "Manual Hinf problem" begin
    @info "Testing Manual Hinf problem"
    
    P = named_ss(ss(tf(1, [1, 0.1, 1])), :P)
    We = named_ss(makeweight(10, 1, 0.1),  :We, u=:y, y=:e)  # We care a lot about the error on low frequencies
    Wu = named_ss(0.01makeweight(1e-3, 10, 10), :Wu, u=:Wu, y=:uw) # Above ω=100, we want to limit the control effort
    # Wd = named_ss(makeweight(1, 1, 1e-3),  :Wd, u=:do, y=:d) # d is low frequency
    Wd = named_ss(ss(1), :Wd, u=:do, y=:d)
    sumP = sumblock("y = Py + d")
    split_u = splitter(:u, 2)

    connections = [
        :u1 => :Wu # splitter to input of Wu
        :u2 => :Pu # splitter to input of P
        :Py => :Py # P output to first input of sumblock
        :d => :d   # output of Wd to second input of sumblock
        :y => :y   # output of sumblock to input of We
    ];

    w1 = [ # External inputs
        :do, :u
    ]
    z1 = [ # External outputs
        :e, :uw, :y
    ];

    G = ROC.connect([P,We,Wu,Wd,sumP,split_u], connections; z1, w1)

    Gsyn = ROC.partition(G, u = [:u], y = [:y]) # You can provide either u or w, and either y or z
    K, γ, info = hinfsynthesize(Gsyn, γrel=1.001, interval = (0.1, 5), transform=true)
    K2, γ2 = hinfsyn_lmi(Gsyn, γrel = 1.001, opt=Hypatia.Optimizer)

    @test γ ≈ 0.3148 atol=1e-2 # value by slicot
    @test γ2 ≈ 0.3148 atol=1e-2 # value by slicot


    @test hinfnorm2(lft(Gsyn, K))[1] ≈ γ atol=1e-2
    @test hinfnorm2(lft(Gsyn, K2))[1] ≈ γ2 atol=1e-2

end


@testset "Numerically hard problems 1" begin
    ## Numerically difficult problem instance, Riccati-based solver performs better than SCS and Mosek. Hypatia together with balance_statespace performs best. 
    # The failure is detectable by hinfnorm2(lft(Gsyn, K))
    Gsyn3A = [0.0 1.0 0.0; -1.2000000000000002e-6 -0.12000999999999999 0.0; -11.2 -0.0 -2.0e-7]
    Gsyn3B = [0.0 0.0; 0.0 1.0; 1.0 0.0]
    Gsyn3C = [-7.466666666666666 -0.0 19.999999866666666; 0.0 0.0 0.0; -11.2 -0.0 0.0]
    Gsyn3D = [0.6666666666666666 0.0; 0.0 1.0; 1.0 -0.0]
    Gsyn3 = ss(Gsyn3A, Gsyn3B, Gsyn3C, Gsyn3D)
    Gsyn = partition(Gsyn3, 1, 2)
    K, γ = hinfsynthesize(Gsyn, ftype=Float64)[1:2]
    K2, γ2 = DyadControlSystems.hinfsyn_lmi(Gsyn, opt=()->Hypatia.Optimizer(iter_limit=100_000), ftype=Float64, ϵ=1e-3, γrel = 1.0001, balance=false)

    @test_broken γ2 ≈ 4.4825150 atol=1e-2 # value by slicot

    @test hinfnorm2(lft(Gsyn, K))[1] ≈ γ atol=1e-3
    @test hinfnorm2(lft(Gsyn, K2))[1] ≈ γ2 atol=1e-2

    # With balancing transform
    K2b, γ2b = DyadControlSystems.hinfsyn_lmi(Gsyn, opt=()->Hypatia.Optimizer(iter_limit=100_000, tol_rel_opt=1e-5, verbose=false), ftype=Float64, ϵ=1e-3, γrel = 1.0001, balance=true, perm=true)
    @test γ2b ≈ 4.4825150 atol=1e-1 # value by slicot
    @test hinfnorm2(lft(Gsyn, K2b))[1] ≈ 4.4825150 atol=1e-1
end

@testset "Numerically hard problems 2" begin
    # The spinning satellite
    a = 10
    A = [0 a; -a 0]
    B = I(2)
    C = [1 a; -a 1]
    D = 0
    Gnom = ss(A,B,C,D)
    WS = makeweight(1e3,1,1e-1)
    WU = let
        tempA = [-3535.46761115006 3535.46761115006; -3535.46761115006 -3535.46761115006]
        tempB = [0.0; 10725.385484097349;;]
        tempC = [-46.28786141317 -6546.0921383757795]
        tempD = [10000.0;;]
        ss(tempA, tempB, tempC, tempD)
    end
    WT = makeweight(0.5,20,100)

    isinteractive() && bodeplot([WS,WU,WT], lab=["wS" "wKS" "wT"])

    Gsyn = hinfpartition(Gnom,WS,WU,WT)
    K1,γ = hinfsynthesize(Gsyn, check=false)
    K2,γ2 = hinfsyn_lmi(Gsyn, opt=()->Hypatia.Optimizer{Float64}(iter_limit=100_000), balance=true, perm=false, ftype=Float64, verbose=false, γrel=1.01) # using Double64 instead of Float64 makes the optimal value slightly better, note also the γrel=1.01, one can also improve it slightly by lowering ϵ to 1e-4

    @test_broken γ ≈ 1.1491 atol=1e-3 # value by slicot
    @test_broken γ2 ≈ 1.1491 atol=1e-3 # value by slicot

    @test_skip γ2 ≈ 1.1491 atol=5e-2 # this test passes locally, but fails on CI. Might be related to https://github.com/chriscoey/Hypatia.jl/issues/801

    @test_broken hinfnorm2(lft(Gsyn, K1))[1] ≈ 1.1491 atol=1e-3 # value by slicot
    @test_broken hinfnorm2(lft(Gsyn, K2))[1] ≈ 1.1491 atol=1e-3 # value by slicot

    # T = feedback(Gnom*K2)
    # plot(step(T, 2))
end



## Passivity

@testset "passivity" begin
    @info "Testing passivity"
  

G = tf(1,[1,1]) |> ss
@test ispassive_lmi(G)

G = tf([1,1],[1,2]) |> ss
@test ispassive_lmi(G)

G = DemoSystems.resonant()
@test !ispassive_lmi(G)


## SPR design
using RobustAndOptimalControl, ControlSystems
ζ = 1/100
α = 1:5
w2 = α .^ 4
As = [
    [0 1; -w2 -2ζ*sqrt(w2)] for w2 in w2
]

A = cat(As..., dims=(1,2))
l = pi
pa = 0.55*l
ba = sin.(α.*pa)
C2s = [
    [0 ba] for ba in ba
]

C2 = reduce(hcat, C2s)
B2 = C2'
B1 = [B2 0B2]
D12 = [0; 1.9]
D21 = D12'

pe = 0.7*l
C1s = [
    [0 sin(α*pe)
    0 0] for α in α
]
C1 = reduce(hcat, C1s)

P0 = ss(A,B1,B2,C1,C2; D12,D21)
K_trad, Gcl_trad = h2synthesize(P0)
@test h2norm(Gcl_trad) ≈ 1.9843 rtol=1e-3 # Trad. H2 table 1

K, Gcl, sν = DyadControlSystems.spr_synthesize(P0, verbose=true, silent_solver=true, balance=false, ϵ=1e-6)
h2norm(Gcl)
@test h2norm(Gcl) <= 1.01 * 2.9096 # we actually get a better result than the paper reports
@test sν <= 1.01 * 4.7746 
@test ispassive(-K)

@test K_trad.B ≈ K.B rtol=1e-10


# w = exp10.(LinRange(-2, 2, 2000))
# bodeplot([Gcl_trad, Gcl], w)
# bodeplot(-[K_trad, K], w)
# nyquistplot(-[K_trad, K], w, xlims=(-0.05,0.25), ylims=(-0.2, 0.2))
# passivityplot(-[K_trad, K], w, yscale=:identity)


## Example 4.2
C1s = [
    [sin(α*pe) 0
    0 0] for α in α
]
C1 = reduce(hcat, C1s)
D12 = [0; 0.5]
# D21 = D12'

P0 = ss(A,B1,B2,C1,C2; D12,D21)
K_trad, Gcl_trad = h2synthesize(P0)
@test h2norm(Gcl_trad) ≈ 1.2375684 rtol=1e-3 # Trad. H2 table 2


K, Gcl, sν = spr_synthesize(P0; verbose=true, silent_solver=true, balance=false, ϵ=1e-6)
h2norm(Gcl)
@test h2norm(Gcl) <= 1.01 * 1.25009
@test sν <= 1.01 * 2.2925 
@test ispassive(-K)
@test K_trad.B ≈ K.B rtol=1e-10


# w = exp10.(LinRange(-2, 3, 2000))
# bodeplot([Gcl_trad, Gcl], w)
# # bodeplot(-[K_trad, K], w)
# nyquistplot(-[K_trad, K], w, xlims=(-0.05,0.25), ylims=(-0.2, 0.2))
# passivityplot(-[K_trad, K], w, yscale=:identity)



## Example 4.3
C1s = [
    [0 sin(α*pe)
    0 0] for α in α
]
C1 = reduce(hcat, C1s)
D12 = [0; 0.1]
D21 = [0 5]

P0 = ss(A,B1,B2,C1,C2; D12,D21)
K_trad, Gcl_trad = h2synthesize(P0)
@test h2norm(Gcl_trad) ≈ 2.10249834 rtol=1e-3 # Trad. H2 table 3


K, Gcl, sν = spr_synthesize(P0; verbose=true, silent_solver=true, balance=false, ϵ=1e-6)
h2norm(Gcl)
@test h2norm(Gcl) <= 1.05 * 2.1882
@test ispassive(-K)



## Example 4.4
ps = 0.625pi
C2s = [
    [0 sin(α*ps)] for α in α
]
C2 = reduce(hcat, C2s)

C1s = [
    [sin(α*pe) 0
    0 0] for α in α
]
C1 = reduce(hcat, C1s)

D12 = [0; 0.1]
D21 = [0 5]

P0 = ss(A,B1,B2,C1,C2; D12,D21)
K_trad, Gcl_trad = h2synthesize(P0)
@test h2norm(Gcl_trad) ≈ 1.7893387 rtol=1e-3 # Trad. H2 table 3


K, Gcl, sν = spr_synthesize(P0; verbose=true, silent_solver=true, balance=false, ϵ=1e-6)
h2norm(Gcl)
@test h2norm(Gcl) <= 1.8749 # Here we're also a bit better
@test ispassive(-K)


end