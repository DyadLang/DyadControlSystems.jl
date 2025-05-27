using DyadControlSystems, RobustAndOptimalControl, LinearAlgebra, ControlSystems
M = ComplexF64[-0.2054746602260128 - 0.050986439231433714im -2.0547466022601277 - 0.5098643923143371im; 2.0547466022601277 + 0.5098643923143371im -0.2054746602260128 - 0.050986439231433714im]
# M = cat(M,M,M,M,M,M, dims=(1,2))

# @btime mussv($M, tol=1e-3)
# 6.410 ms (58609 allocations: 3.20 MiB)
# The LMI method is not very nice when there are many frequencies to calculate at since it can not be warm started. Also, most of the time is spent on problem setup in Convex, perhaps a single large problem can be setup instead?
# @btime structured_singular_value($M)
# 7.469 ms (74078 allocations: 12.54 MiB)

@test structured_singular_value(M) ≈ 2.127619 atol=1e-3
@test mussv(M, tol=1e-4) ≈ 2.127619 atol=1e-3
@test mussv(M .* ones(1,1,10), tol=1e-4) ≈ 2.127619 * ones(10) atol=1e-3
@test mussv_tv(M .* ones(1,1,10), tol=1e-3) ≈ 2.127619 atol=1e-3


## this time mimo real
delta = uss([δr(), δr()])
a = 1
P = ss([0 a; -a -1], I(2), [1 a; 0 1], 0)* (ss(1.0*I(2)) + delta)
K = ss(I(2))
blocks = [
    [-1,0],
    [-1,0]
]

G = lft(P, -K)
@test_skip mussv(G.M, blocks) ≈ 1/sqrt(2) # not yet supported
@test norm(G.M, Inf) ≈ 1/1.3416 atol=0.001 # with no blocks, equal to hinfnorm

blocks, M0 = RobustAndOptimalControl.blocksort(G)

hn = norm(M, Inf)

w = exp10.(LinRange(-5, 1, 100))
M = freqresp(M0, w)
mu = mussv_DG(M)
# mu2 = mussv(M[:,:,1], blocks) # not supported here yet
maximum(mu)
# maximum(structured_singular_value(M))
@test 1/maximum(mu) ≈ √(2) atol=0.01


## Diagonal complex
delta = uss([δc(), δc()])
a = 1
P = ss([0 a; -a -1], I(2), [1 a; 0 1], 0)* (I(2) + delta)
K = ss(I(2))
G = lft(P, -K)
@test mussv(G) ≈ 0.7439 atol=1e-2
@test mussv_tv(G) >= mussv(G)



## Full complex (note, the system has changed)
delta = δss(2,2)
P = ss([0 1; 0 0], I(2), [1 0], 0) * (I(2) + delta)
K = ss([1;1])
G = lft(P, -K)
blocks, M0 = RobustAndOptimalControl.blocksort(G)
M = freqresp(M0, 5)
mu1 = opnorm(M)
mu2, Q = mussv(M, blocks, shortcut=false, retQ=true)
@test mu1 ≈ mu2 rtol=1e-2
@test abs(Q[1,2]) < 1e-6
@test abs(Q[2,1]) < 1e-6
@test abs(Q[1,1] - Q[2,2]) < 1e-6 # these should be the same, the matrix should be q*I


# statespace version with full complex
@test mussv(M0, blocks) ≈ norm(M0, Inf) atol=1e-2 



## Diagonal complex
delta = uss([δc(), δc()])
P = ss([0 1; 0 0], I(2), [1 0], 0) * (I(2) + delta)
K = ss([1;1])
G = lft(P, -K)
blocks, M0 = RobustAndOptimalControl.blocksort(G)
M = freqresp(M0, w)
mu = mussv(M, blocks, shortcut=false)


# statespace version with diagonal complex
@test mussv(M0, blocks) > maximum(mu) # since mu is gridded
@test mussv(M0, blocks) ≈ maximum(mu) atol=1e-1 





##
P = ss([0 1; 0 0], I(2), [1 0], 0) * ((I(2)) + delta)
K = ss([1;1])
G = lft(P, -K)
w = exp10.(LinRange(-5, 1, 100))
M = freqresp(G.M, w)
mu_d = DyadControlSystems.mussv_tv(M)
mu2_d = DyadControlSystems.mussv_tv(G)

@test mu_d < mu2_d
@test mu_d ≈ mu2_d atol=0.1

##

## 
w = exp10.(LinRange(-2, 2, 100))
M0 = [
    tf(1,[2,1]) 1 tf([1, -2],[2,4])
    -1 tf([1, 0], [1,1,1]) tf(1,[1,1])^2
    tf([3,0], [1,5]) tf(-1,[4,1]) 1
]
M = freqresp(M0, w)
blocks = [
    [1,0],
    [2,2]
]

norms = vec(mapslices(opnorm, M, dims=(1,2)))
mu, Q = mussv(M[:,:,end], blocks, shortcut=false, retQ=true)
@test mu ≈ 2.2074 atol=0.01
mu, Q = mussv(M[:,:,1], blocks, shortcut=false, retQ=true)
@test mu ≈ 2.16798 atol=0.01

if isinteractive()
    mu = mussv(M, blocks, shortcut=false)
    specrads = mapslices(M, dims=(1,2)) do M
        RobustAndOptimalControl.ρ(M)
    end |> vec
    plot(w, [specrads mu norms], xscale=:log10) |> display
    # should look like the figure 47 in https://www.imng.uni-stuttgart.de/mst/files/RC.pdf
end




##
M = [
    -0.419172+1.34694im     0.920658-0.418172im   -0.597287+0.328288im    0.875252+0.242921im   -0.373638-0.661918im
    0.117772-1.48963im     0.521134-0.697134im     0.92823-0.912084im     1.08485-0.729275im    0.245678-0.624385im
   -0.716686+0.124978im   0.0764972+0.454298im   -0.315827-0.0593509im    1.55683-0.689675im    0.317717+0.281817im
     1.10179-0.0483083im  0.0345058-0.553636im   -0.897137+1.03713im    -0.156598+0.842033im  -0.0605758+0.0125149im
    0.626008-0.858766im    0.284181+0.0317958im   0.300803-0.0233927im   -1.54014-0.209006im  -0.0507538-0.0693233im
]


# Full
@test opnorm(M) ≈ 3.1942 atol=2e-3

# diagonal complex uncertainties
@test mussv(M, tol=1e-4) ≈ 2.9675 atol=1e-2

# Diagonal real bounds by mu-tools: [2.0659, 1.5083]
@test mussv_DG(M) ≈ 2.0659 atol=1e-2
@test_skip mussv(M, [[-1, 0] for _ in 1:5], shortcut=false) ≈ 2.0659 atol=1e-2 # not yet supported

# using RobustAndOptimalControl.IntervalArithmetic
# D = Diagonal([Interval(-1,1) for _ in 1:5])
# 1/RobustAndOptimalControl.bisect_a(M, D)

# Identity complex
mu = mussv(M, [[5, 0]], shortcut=false) # large block δc*I
@test mu ≈ 2.1252 atol=1e-2


## The benchmark below exposes numerical problems for SCS,
# but M and Hypatia solves it well. Hypatia is twice as fast as M
# a = 10
# P = ss([0 a; -a 0], I(2), [1 a; -a 1], 0)
# K = ss(I(2))
# w = exp10.(LinRange(-5, 1, 100))
# delays = 0.01:0.02:0.5
# @time mus1 = map(delays) do d
#     @show d
#     Pd = P * (I(2) + neglected_delay(d).*I(2)*uss([δc(), δc()]))
#     mussv(lft(Pd, -K))
# end

# mus2 = map(delays) do d
#     @show d
#     Pd = P * (I(2) + neglected_delay(d).*I(2)*uss([δc(), δc()]))
#     maximum(structured_singular_value(lft(Pd, -K), w))
# end

# @time mus3 = map(delays) do d
#     @show d
#     Pd = P * (I(2) + neglected_delay(d).*I(2)*uss([δc(), δc()]))
#     mussv(lft(Pd, -K), optimizer=DyadControlSystems.Hypatia.Optimizer)
# end

# plot(delays, mus1, lab="SCS")
# plot!(delays, mus2, lab="Naive")
# plot!(delays, mus3, lab="Hypatia", l=:dash)