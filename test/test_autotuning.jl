using DyadControlSystems, FiniteDiff, Plots, LinearAlgebra
using DyadControlSystems: IpoptSolver

## test numerical scaling


# s = 10
# p0 = rand(3)
# C = pid(p0)*tf(1, [0.001, 1])
# sC = time_scale(C, s)
# C2 = pid([p0[1], p0[2]*s, p0[3]/s])*tf(1, [0.001/s, 1])

# w = exp10.(LinRange(-5, 5, 200))
# bodeplot([C, sC, C2], w)



w = exp10.(LinRange(-2, 1, 200))
P = ssrand(1,1,3, proper=true)
P = sign(dcgain(P)[])*P
p = rand(4)

z = 1 / sqrt(2)
Tf = p[4]
# F = tf([one(Tf)], [Tf^2, 2*z*Tf, one(Tf)])
F = tf([one(Tf)], [Tf, one(Tf)])
C = pid(p[1:3])*F
L1 = P*C # Loop gain without scaling

P2, p2, w2, scale_info = DyadControlSystems._scale_numerics(P, p, w)
@test_broken w2[end ÷ 2] ≈ 1 # deactivated due to bug with satisfying Mks when rescaling frequency
Tf2 = p2[4]
F2 = tf([one(Tf2)], [Tf2, one(Tf2)])
C2 = pid(p2[1:3])*F2

@test hinfnorm2(G_CS(P, C))[1] * scale_info.g_scale ≈ hinfnorm2(G_CS(P2, C2))[1] rtol=1e-6


L2 = time_scale(P2*C2, scale_info.ωc)

@test tf(L2) ≈ tf(L1) rtol=1e-2
# bodeplot([L1, L2], w)
# bodeplot([C, C2], w)


p3 = DyadControlSystems._unscale_numerics(p2, scale_info)
@test p3 ≈ p
# @test tf(P3) ≈ tf(P) rtol=1e-4



## Test that freqresp is differentiable
w = exp10.(LinRange(-2, 2, 20))
function testdiff(A)
    B = I(2)
    C = I(2)
    D = 0
    sys = ss(A,B,C,D)
    R = freqresp(sys, w)
    sum(abs, R)
end
@test_nowarn DyadControlSystems.ForwardDiff.gradient(testdiff, randn(2,2))

##

# Process model (continuous time LTI SISO system).
T = 4 # Time constant
L = 1 # Delay
K = 1 # Gain
P = tf(K, [T, 1.0])#*delay(L) # process dynamics

# Robustness consraints
Ms = 1.2 # Maximum allowed sensitivity function magnitude
Mt = Ms  # Maximum allowed complementary sensitivity function magnitude
Mks = 10.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
w = 2π .* exp10.(LinRange(-2, 2, 100))
# w = exp10.(LinRange(-4, 3, 200))

p0 = Float64[1, 1, 0, 0]
prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.1, Tf=25.0, disc=:tustin)


## Test inner functions
P, Ms, Mt, Mks, w = prob.P, prob.Ms, prob.Mt, prob.Mks, prob.w
pC = p0[1:3]
pF = p0[4]
c0 = pC[1]
pC ./= c0
P = P * c0
Tmin = 1 / w[end]
pF = max(pF, Tmin)
p = [pC; pF]
pC = p -> p[1:3]
pF = p -> p[4]
p[4] = max(p[4], Tmin)
fr = x -> DyadControlSystems.faster_freqresp(x, w, im .* w) # frequency response over grid
Pw = fr(P) # Frequency response
@test size(Pw, 1) == length(w) 

# @inferred DyadControlSystems.TFS(P, pC, pF)

tfs = DyadControlSystems.TFS(P, pC, pF)
@inferred tfs.S(p0)
@inferred tfs.PS(p0)
@inferred tfs.F(p0)
@inferred tfs.K(p0)
@inferred tfs.C(p0)
@inferred tfs.∇C_pC(p0)
@inferred tfs.∇F_pF(p0)
@inferred tfs.∇K_p(p0 .+ 0.1) # add small to have proper tf

@test ss(tfs.K(p)).nx == 3
@test ss(tfs.K(p)).ny == 1
@test ss(tfs.K(p)).nu == 1

@test tfs.F(p) == tf([1], [p[4]^2, 2*p[4]*1/sqrt(2), 1])
@test hinfnorm2(tfs.S(p) - feedback(1, tfs.F(p)*tfs.C(p)*P))[1] < 1e-10
# c,∇c = DyadControlSystems.autotuning_cost_grad(P, p, prob, tfs)

# @btime DyadControlSystems.autotuning_cost_grad($p, $prob, $tfs);
# 356.914 μs (5068 allocations: 1.49 MiB)
# 238.331 μs (1356 allocations: 1.18 MiB)
# 116.660 μs (524 allocations: 162.88 KiB)
# 186.502 μs (1231 allocations: 935.11 KiB) PS(p)
# 155.203 μs (1100 allocations: 463.58 KiB)
# 149.092 μs (1015 allocations: 370.14 KiB) remove call to minreal
# 152.909 μs (1029 allocations: 285.25 KiB) remove another call to minreal

# changing to PS(p) from P*S(p) actually made timings of autotuning_cost_grad and solve worse, even though it results in a smaller system. One reason might be the repeated calls to minreal before dstep, maybe hey simplified a bit too much before? Anyway, we get more iterations now as well so it seems worse overall. Since we incur more allocations, maybe we've fucked up types?


# @btime DyadControlSystems.dstep($(prob.P), 0.01, 0:0.01:10);
# 62.798 μs (36 allocations: 25.95 KiB)

##
p0 = DyadControlSystems.init_strategy_loopshapingPI(prob, true)
res = solve(prob, p0[3], alg=IpoptSolver(verbose=false, exact_hessian=false))
@time res = solve(prob, alg=IpoptSolver(verbose=false, exact_hessian=false))
# @test res.p ≈ [9.669035919442452, 13.56702987024118, 0.0, 0.0015915494309189533] rtol=0.1
# @test res.cost < 1.12*0.07872394756135513 # with scaling, the cost is different, the parameter vector is the same though
@test res.cost < 1.1*1.27

@time res = solve(prob, p0, verbose=false);
@test res.cost < 1.1*1.27
# got 0.07938124087455262 at [9.676885983501608, 13.508941737282864, 0.0, 0.0015915494309189533] after 57 iterations (returned XTOL_REACHED)
# 0.300013 seconds (235.17 k allocations: 46.177 MiB)
# 4.687873 seconds (1.13 M allocations: 520.030 MiB, 0.94% gc time)
# 4.817748 seconds (602.39 k allocations: 354.160 MiB) # Julia v1.8
# 4.704849 seconds (602.10 k allocations: 353.215 MiB, 2.67% gc time) jacobian cache

# got 0.07872394083663733 at [9.699190790223808, 13.65561813774254, 0.010933114457068285, 0.0015915494309189533] after 94 iterations (returned XTOL_REACHED)
#   3.009864 seconds (582.42 k allocations: 344.649 MiB, 1.24% gc time)

# got 0.07872125834135947 at [9.699376652124002, 13.656161908564258, 0.011001610598615527, 0.0015915494309189533] after 105 iterations (returned XTOL_REACHED)
#   2.832125 seconds (560.83 k allocations: 562.399 MiB, 0.98% gc time)# some static systems
# 3.520682 seconds (506.53 k allocations: 559.376 MiB, 20.95% gc time) making P in prob static made allocations go up, maybe need to convert all controller function to static as well for this to improve

# got 0.07872062553793753 at [9.699435906181943, 13.656170449437091, 0.010930107092270028, 0.0015915494309189533] after 76 iterations (returned XTOL_REACHED)
#   2.330960 seconds (342.36 k allocations: 418.432 MiB, 1.29% gc time) # Precomputed ∇F_pF

# got 0.07871576586766227 at [9.699808434957298, 13.656893756640967, 0.010864032633928034, 0.0015915494309189533] after 77 iterations (returned XTOL_REACHED)
#   1.380147 seconds (153.85 k allocations: 94.931 MiB, 2.86% gc time) # Custom static freqresp_nohess
# 1.360514 seconds (154.16 k allocations: 94.941 MiB) Polyester.@batch in freqresp_nohess (makes standard evaluation much faster so we keep it

# got 0.07871576586766227 at [9.699808434957298, 13.656893756640967, 0.010864032633928034, 0.0015915494309189533] after 77 iterations (returned XTOL_REACHED)
#   1.377822 seconds (299.94 k allocations: 108.353 MiB, 1.30% gc time)


# Note, the algorithm used below is much faster for this problem
# @btime solve($prob, $p0, alg = :LD_SLSQP, xtol_rel=1e-4);
# got 0.07872393531722481 at [9.69919102982491, 13.655619629391973, 0.010933457576454721, 0.0015915494309189533] after 28 iterations (returned XTOL_REACHED)
# 8.197 ms (99134 allocations: 17.01 MiB)

# got 0.0894055398566879 at [9.084258344007731, 12.072709574066161, 0.08249875974771005, 0.00919383957593824] with Ipopt and approximate hessians
#   352.479 ms (282674 allocations: 38.48 MiB)

plot(res)


## Test with delay
L = 1 # Delay
P = tf(K, [T, 1.0])*delay(L) # process dynamics

prob = AutoTuningProblem(; P, Ms=1.5, Mt, Mks, w, Ts=0.1, Tf=25.0)
p0 = DyadControlSystems.init_strategy_loopshapingPI(prob, true)
res = solve(prob, alg=IpoptSolver(verbose=false, exact_hessian=false))
@time res = solve(prob, p0, verbose=false)#, xtol_rel=1e-5, alg = :LD_SLSQP)
@test res.cost < 1.1*1.2615936400830905
 # can reach 1.2615931929274606 at [1.3706552040409923, 0.5607532007314091, 0.5572093850898701, 0.07504773018548977] after 35 iterations (returned XTOL_REACHED)
#  1.332080 seconds (377.02 k allocations: 202.190 MiB)
# 191.604 ms (739966 allocations: 86.16 MiB) with Ipopt

plot(res)


## pidIE
using DyadControlSystems
T = 4 # Time constant
L = 1 # Delay
K = 1 # Gain
P = tf(K, [T, 1.0])#*delay(L) # process dynamics

## Robustness consraints
Ms = 1.2 # Maximum allowed sensitivity function magnitude
Mt = Ms  # Maximum allowed complementary sensitivity function magnitude
Mks = 10.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
w = 2π .* exp10.(LinRange(-2, 2, 50))
# w = exp10.(LinRange(-4, 3, 200))

prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.1, Tf=25.0, metric=:IE)
p0 = DyadControlSystems.init_strategy_loopshapingPI(prob, true)

res = DyadControlSystems.solve(prob, p0, maxiter=100, tol=1e-2, eps=1e-6, verbose=false)
# @test res.p ≈ [9.942537186250703, 14.310860918557086] rtol=0.1
@test res.p[1:3] ≈ [9.942537186250703, 14.310860918557086, 0.0006995434080803242] rtol=0.1
@test res.cost > 0.9*13.915054200338265


plot(res)

## test with PI only
p0 = DyadControlSystems.init_strategy_loopshapingPI(prob)
res = DyadControlSystems.solve(prob, p0, maxiter=100, tol=1e-2, eps=1e-3, verbose=false)
@test res.p ≈ [9.646554785027194, 13.914377250002708] rtol=0.1
@test res.cost > 0.9*13.877106653823152 # maximization

## Combined
prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.1, Tf=25.0, metric=:IEIAE)
p0 = DyadControlSystems.init_strategy_loopshapingPI(prob, true)
res = DyadControlSystems.solve(prob, p0, maxiter=100, tol=1e-2, eps=1e-3, verbose=false)
# got 0.07780300484349312 at [0.9993701928929265, 1.439965753440338, 0.0011661552744589457, 0.0015915494309189533] after 11 iterations (returned XTOL_REACHED)

plot(res)
# @test res.p ≈ [9.654196112579122, 13.784550161835654, 0.01140229268762217, 0.0015915494309189533] rtol=0.1
cost_unconstrained = res.cost
@test_broken cost_unconstrained < 0.08 # 0.0783473347536410


## Test pmax
# limit ki and test that the constraint is honored
prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.1, Tf=25.0, metric=:IEIAE, pmax=[Inf, 10, Inf, Inf])
p0 = Float64[0,0,0]
p0 = DyadControlSystems.init_strategy_loopshapingPI(prob, true)
res = DyadControlSystems.solve(prob, p0, maxiter=100, tol=1e-2, eps=1e-3, verbose=false)
plot(res)
@test res.p[2] <= 1.01*10
@test res.cost < 0.12 # 0.10936834501262285

# test the same thing for the IE solver
prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.1, Tf=25.0, metric=:IE, pmax=[Inf, 10, Inf, Inf])
# p0 = Float64[0,0,0]
p0 = DyadControlSystems.init_strategy_loopshapingPI(prob, true)
res = DyadControlSystems.solve(prob, p0, maxiter=100, tol=1e-2, eps=1e-3, verbose=false)
plot(res)
@test res.p[2] ≈ 10 atol=0.1
@test res.cost > 0.9*9.999995929991218 # maximization



## Another test case
# This test case with this initial guess fails to find a stabilizing controller after the first iteration.
# This is used to test the heuristic to increase Mks to find a stabilizing controller and then lower it again
P = let
    PA = [0.0 1.0; -1.2000000000000002e-6 -0.12000999999999999]
    PB = [0.0; 1.0;;]
    PC = [11.2 0.0]
    PD = [0.0;;]
    ss(PA, PB, PC, PD)
end

Ms = 1.2 # Maximum allowed sensitivity function magnitude
Mt = 1.2  # Maximum allowed complementary sensitivity function magnitude
Mks = 100.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
w = 2π .* exp10.(LinRange(-2, 2, 100))
p0 = Float64[0.1, 0.0, 0, 0.001]
prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.01, Tf=55.0, metric=:IE)

res = DyadControlSystems.solve(prob, p0, maxiter=10, tol=1e-3, eps=1e-6, verbose=false)
@test res.cost > 0 # 0 if it fails
@test res.cost > 0.9*0.017160412672000032


isinteractive() && plot(res)


prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.01, Tf=55.0, metric=:IE)
p0 = DyadControlSystems.init_strategy_loopshapingPID(prob)
res = DyadControlSystems.solve(prob, p0, maxiter=10, tol=1e-3, eps=1e-6, verbose=false)
@test res.cost > 0 # 0 if it fails
@test res.cost > 0.9*0.017160412672000032 # maximization



## Unstable process
# Unstable processes require finding an initial stabilizing controller
# This example comes from the paper

P = ss(zpk([], [1,-10], 10))

Ms = 1.4 # Maximum allowed sensitivity function magnitude
Mt = 1.4  # Maximum allowed complementary sensitivity function magnitude
Mks = 100.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
w = exp10.(LinRange(-3, 3, 100))
p0 = Float64[6, 1, 0, 0.001]
# p0 = DyadControlSystems.init_strategy_loopshapingPID(prob)
prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.01, Tf=10.0, metric=:IE)
res = DyadControlSystems.solve(prob, p0, maxiter=20, tol=1e-3, eps=1e-6, verbose=false, random_start=false)

@test res.cost > 0 # 0 if it fails
@test res.cost  > 0.9*5.498170764749871

if isinteractive()
    f1 = plot(res)
    K = pid(p0)*tf(1, [1/maximum(w), 1])
    f2 = bodeplot(feedback(P*res.K))
    f3 = RobustAndOptimalControl.gangoffourplot(P, ss(K), w, Ms_lines=[prob.Ms])
    plot(f1,f2,f3)
end

## No initial parameter vector

ps = DyadControlSystems.init_strategy_loopshapingPID(prob)
@test !isempty(ps)
res = DyadControlSystems.solve(prob, maxiter=20, tol=1e-3, eps=1e-6, verbose=false, random_start=false, mapfun=map)
@test res.cost > 0.9*5.49801898432824

plot(res)


nyquistplot(P, w)



## Double mass model

P = ControlSystems.DemoSystems.double_mass_model()

Ms = 1.3 # Maximum allowed sensitivity function magnitude
Mt = 1.5  # Maximum allowed complementary sensitivity function magnitude
Mks = 300.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
w = 2π .* exp10.(LinRange(-2, 2, 100))

prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.01, Tf=3.0, metric=:IE)
res = DyadControlSystems.solve(prob, maxiter=10, tol=1e-3, eps=1e-6, verbose=false);
plot(res)
@test res.cost > 0 # 0 if it fails
@test res.cost > 0.9*0.33987494582757033

prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.01, Tf=3.0, metric=:IEIAE)
res = DyadControlSystems.solve(prob, maxiter=10, tol=1e-3, eps=1e-6, verbose=false);
plot(res)
@test res.cost > 0 # 0 if it fails
@test res.p[2] > 0.9*0.33987494582757033


## Resonant

P = ControlSystems.DemoSystems.resonant()

Ms = 1.5 # Maximum allowed sensitivity function magnitude
Mt = 1.5  # Maximum allowed complementary sensitivity function magnitude
Mks = 30.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
w = 2π .* exp10.(LinRange(-3, 2, 100))

@time prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.02, Tf=5.0, metric=:IE)
@time res = DyadControlSystems.solve(prob, maxiter=10, tol=1e-3, eps=1e-6, verbose=false);
@time plot(res)
@test res.cost > 0 # 0 if it fails
@test res.cost > 0.9*0.21294688558042313

C,kp,ki,kd,fig = loopshapingPID(P, 1; Mt, doplot=isinteractive(), Tf=0.05, ϕt=45, form=:parallel); fig
p0 = [kp, ki, kd, 0.05]
prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.02, Tf=5.0, metric=:IAE)
res = DyadControlSystems.solve(prob, p0, maxiter=100, tol=1e-3, eps=1e-6, verbose=false);
plot(res)
# @test res.p ≈ [4.558635176141274, 3.6479468637612387, 2.536135482667003, 0.0883471093795396] rtol=1e-1

res = DyadControlSystems.solve(prob, maxiter=100, tol=1e-3, eps=1e-6, verbose=false);
plot(res)
@test res.cost > 0 # 0 if it fails
@test res.p[2] > 0.9*3.625298

##
P = let
    room_modelA = [-0.00011642997747379028 5.1799884706927967e-5 0.0002684021348927724 5.1169383946991395e-6; 5.179988470692794e-5 -3.41397520076659e-5 -0.00035441923907489386 -5.254813802825365e-6; -0.00026840213489277257 0.00035441923907489456 -0.001752734842810621 -7.537118265801767e-5; 5.116938394698816e-6 -5.254813802824936e-6 7.537118265801547e-5 -1.961609636486536e-5]
    room_modelB = [-0.03359011837012772; 0.009518365953452339; -0.03579627464436649; 0.000740247597244907;;]
    room_modelC = [-0.033590118370127715 0.00951836595345235 0.03579627464436649 0.0007402475972449485]
    room_modelD = [0.0;;]
    ss(room_modelA, room_modelB, room_modelC, room_modelD)
end

one_week = 7*24*60*60 # seconds
w = exp10.(LinRange(log10(1/one_week), -1, 150))
Tf = 1/0.005

# C, kp, ki, kd, fig, CF = loopshapingPID(P, 1w[end÷2]; Mt, ϕt=45, form=:parallel, doplot=true, Tf); fig

F = tf(1, [Tf^2, 2*Tf/sqrt(2), 1])

Ms = 1.5 # Maximum allowed sensitivity function magnitude
Mt = 1.5  # Maximum allowed complementary sensitivity function magnitude
Mks = 20.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
@time prob = AutoTuningProblem(; P=P, Ms, Mt, Mks, w, Ts=2, Tf=2*60*60, metric=:IEIAE)
# p0 = ControlSystems.convert_pidparams_from_standard(1,2000,600, :parallel)
# p0 = [p0..., 1/0.005]
@time res = DyadControlSystems.solve(prob, maxiter=10, tol=1e-3, eps=1e-6, verbose=false);
plot(res)




# ## Manual loop shaping for the system above
# Tf = 1/0.005
# F = tf(1, [Tf^2, 2*Tf/sqrt(2), 1])
# C = pid(1, 2000, 600) * F
# plot(
#     marginplot(P*C, w),
#     gangoffourplot(P, C, w, Ms_lines=[Ms], Mt_lines=[Mt])
# )
# # bodeplot(P*C, w)
# plot(step(extended_gangoffour(P, C), 2*60*60))


##
T = 4 # Time constant
L = 1 # Delay
K = 0.9 # Gain
P = tf(K, [T, 1.0])#*delay(L) # process dynamics
Ms = 1.2
Mt = Ms
Mks = 10.0
w = 2π .* exp10.(LinRange(-2, 2, 100))
prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=0.1, Tf=25.0)
p0 = DyadControlSystems.init_strategy_loopshapingPI(prob, true)
res = solve(prob, p0, verbose=false)
plot(res)

@test DyadControlSystems.satisfied(prob, prob.P, res.p, prob.w, prob.Mks)
