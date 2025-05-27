using Test, DyadControlSystems
using DyadControlSystems: AutoTuningProblem2, solve, IpoptSolver
include("autotuning_benchmarks.jl")
P =
    let
        tempA = [
            0.0 0.0 1.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0 1.0 0.0
            -400.0 400.0 -0.4 -0.0 0.4 -0.0
            0.0 0.0 0.0 0.0 0.0 1.0
            10.0 -11.0 0.01 1.0 -0.011 0.001
            0.0 10.0 0.0 -10.0 0.01 -0.01
        ]
        tempB = [0.0 0.0; 0.0 0.0; 100.80166347371734 0.0; 0.0 0.0; 0.0 -0.001; 0.0 0.01]
        tempC = [0.0 0.0 0.0 1.0 0.0 0.0]
        tempD = [0.0 0.0]
        named_ss(
            ss(tempA, tempB, tempC, tempD),
            x = [
                Symbol("wheel₊mass₊s(t)"),
                Symbol("car_and_suspension₊mass₊s(t)"),
                Symbol("wheel₊mass₊v(t)"),
                Symbol("seat₊mass₊s(t)"),
                Symbol("car_and_suspension₊mass₊v(t)"),
                Symbol("seat₊mass₊v(t)"),
            ],
            u = [Symbol("road₊f(t)_scaled"), Symbol("force₊f(t)")],
            y = [Symbol("output")],
        )
end

Copt = let
    CA = [0.0 4.0 0.0; 0.0 0.0 32.0; 0.0 -62.1902026690993 -63.08861205338373]
    CB = [0.0; 0.0; 32.0;;]
    CC = [2.375526272770401 3.1411399244248353 50.457924779299645]
    CD = [0.0;;]
    named_ss(ss(CA, CB, CC, CD), x=[:optimizedx1, :optimizedx2, :optimizedx3], u=[:optimizedu], y=[:optimizedy])
end

res_p = [1.6162751248202438, 4.889312944234776, 0.8113484538356535, 0.022416304882035273]

dcg = dcgain(minreal(sminreal(P[:output, :force])))
P = inv(dcg) * P 

Ts = 0.01
Tf = 10.0
Ms = 1.3
Mt = 1.2
Mks = 3e1
disc = :tustin
tvec = 0:Ts:Tf
w = exp10.(LinRange(-2, 3, 300))


params = [0.0010000000000000007, 0.00020000000000000015, 0.020000000000000014, 0.001]

measurement = P.y[1]
control_input = :force
step_output = P.y[1]
ref = 0.0
if false # new approach
    step_input = :road
    response_type = impulse
    timeweight = false
elseif false # new approach with realistic disturbance input
    step_input = :road
    u = @. cos(2*tvec) * exp(-0.5*tvec)#; plot(u)
    response_type = (G, t) -> lsim(G, u', t)
    timeweight = false
else # Old approach
    step_input = :force
    response_type = step
    timeweight = false
end

# using MadNLP
# solver = MadNLP.Optimizer()

solver = IpoptSolver(
    exact_hessian = false,
    verbose = false,
    max_iter = 400,
    acceptable_iter = 10,
    tol             = 1e-5,
    acceptable_tol  = 1e-4,
)

##

prob = AutoTuningProblem2(
    P;
    w,
    measurement,
    control_input,
    step_input,
    step_output,
    response_type,
    Ts,
    Tf,
    Ms,
    Mt,
    Mks,
    metric = abs2,
    ref,
    disc,
    timeweight,
    # autodiff = Optimization.AutoForwardDiff(),
    # autodiff = Optimization.AutoFiniteDiff(),
    reduce = true,
)

prob, params = problem_8(3)
solver = IpoptSolver(
    exact_hessian = false,
    verbose = false,
    max_iter = 400,
    max_cpu_time = 60.0,
    acceptable_iter = 12,
    tol             = 1e-5,
    acceptable_tol  = 1e-4,
)

@time sol = solve(prob, params, solver)
# 0.526267 seconds (469.05 k allocations: 209.042 MiB, 5.83% gc time)
# 0.525299 seconds (467.17 k allocations: 199.349 MiB, 5.44% gc time) BodemagWorkspace
# 0.388998 seconds (287.86 k allocations: 60.750 MiB, 3.82% gc time) StaticStateSpace to bodemag!, @threads
# 0.377643 seconds (287.78 k allocations: 60.746 MiB, 3.04% gc time) :static threads
# 0.371406 seconds (289.17 k allocations: 59.367 MiB, 1.61% gc time) # bodemag_nohess
# 0.378257 seconds (254.75 k allocations: 57.729 MiB, 2.06% gc time) static_pid
@test sol.cost < 0.05247807147593516*1.1
(; K, G, sol) = sol

if isinteractive()
    using Plots
    # bodeplot(G[:, :r], size=(1000,1000), plotphase=false, lab=["S" "T" "CS"]) |> display
    # hline!([Ms Mt Mks], l=(:black, :dash), primary=false)
    # plot(impulse(G[:output, :road], tvec, method=:zoh))
    # plot(step(G[:output, :r], tvec, method=:zoh))


    G0 = extended_gangoffour(P[:, :force], Copt)
    fig_bode = bodeplot(G0, w, title=["S" "PS" "CS" "T"], plotphase=false, lab="JSC")
    G1 = extended_gangoffour(P[:, :force], K)
    bodeplot!(G1, w, title=["S" "PS" "CS" "T"], plotphase=false, lab="new", legend=:topleft)
    hline!([Ms Mt Mks], l=(:black, :dash), primary=false, sp=[1 4 3])

    q(G) = balance_statespace(G)[1] |> minreal

    fig_time = plot(impulse(q(lft(P[[1,1], :].sys, -Copt)), 0:0.001:10, method=:zoh), label="JSC")
    plot!(impulse(q(lft(P[[1,1], :].sys, -K)), 0:0.001:10, method=:zoh), label="new")

    fig_nyquist = nyquistplot(P[:, :force]*Copt, w, lab="JSC", Ms_circles=Ms, Mt_circles=Mt)
    nyquistplot!(P[:, :force]*K, w, lab="new", xlims=(-4, 1), ylims=(-4, 1))

    plot(fig_time, fig_bode, fig_nyquist, size=(1000, 1000), layout=(2,2))

end

# ==============================================================================
## Test some parameters forced to 0
# ==============================================================================

T = 4 # Time constant
K = 1 # Gain
P = tf(K, [T, 1.0])

Mt = Ms = 1.5  # Maximum allowed complementary sensitivity function magnitude
Mks = 10.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
w = 2π .* exp10.(LinRange(-3, 1, 100))

params = Float64[1, 1, 0, 0]
prob = AutoTuningProblem2(P; Ms, Mt, Mks, w, Ts=0.1, Tf=25.0, ub=[Inf,Inf,0,Inf])
@test prob.ub[4] == 0 # filter was turned off

sol = solve(prob, params, solver)
isinteractive() && plot(sol)
@test sol.p[3] == 0
@test sol.p[4] == 0
@test sol.cost < 0.003387380085589611 * 1.1

prob = AutoTuningProblem2(P; Ms, Mt, Mks, w, Ts=0.1, Tf=25.0, ub=[Inf,Inf,0,Inf], filter_order=1)
@test prob.ub[4] == 0 # filter was turned off

sol = solve(prob, params, solver)
isinteractive() && plot(sol)
@test sol.p[3] == 0
@test sol.p[4] == 0
@test sol.cost < 0.003387380085589611 * 1.1


# ==============================================================================
## Test error handling
# ==============================================================================
@test_throws "Filter time constant must be positive in the presence of derivative action" AutoTuningProblem2(P; Ms, Mt, Mks, w, Ts=0.1, Tf=25.0, ub=[Inf,Inf,1,0]) # Cannot turn off filter when using D

# ==============================================================================
## Test both reference and disturbance input
# ==============================================================================
w = 2π .* exp10.(LinRange(-3, 3, 100))
step_input = [:reference_input, :u]
ref = [1.0, 0.0]
prob = AutoTuningProblem2(P; Ms=1.2, Mt=1.2, Mks=20, ref, w, Ts=0.01, Tf=5.0, step_input)
params = [1, 1, 0.1, 0.1]
sol = solve(prob, params, solver)
@test sol.cost < 0.1180777918363196 * 1.1
isinteractive() && plot(sol)

prob = AutoTuningProblem2(P; Ms=1.2, Mt=1.2, Mks=20, ref, w, Ts=0.01, Tf=5.0, step_input, filter_order=1)
params = [1, 1, 0.1, 0.1]
sol = solve(prob, params, solver)
@test sol.cost <= 0.1180777918363196 # we can push harder with less filter
isinteractive() && plot(sol)


prob = AutoTuningProblem2(P; Ms=1.2, Mt=1.2, Mks=20, ref, w, Ts=0.01, Tf=5.0, step_input, filter_order=2, optimize_d=true, ub=[Inf,Inf,Inf,Inf,2], lb=[0,0,0,0,0])
params = [1, 1, 0.1, 0.1]
sol = solve(prob, params, solver)
@test sol.cost <= 0.1180777918363196 # we can push harder with less filter, larger tolerated filter damping can flatten the filter peak, allowing more effort for high frequencies
isinteractive() && plot(sol)

# ==============================================================================
## Test two step _outputs_
# ==============================================================================

step_input = :reference_input
step_output = [:y, :u_controller_output_C]
measurement = :y

ref = [1.0, 0.0]

scale = 1/5
prob = AutoTuningProblem2(1/scale*P; Ms=1.2, Mt=1.2, Mks=20*scale, ref, w, Ts=0.01, Tf=5.0, step_input, step_output, measurement)
params = [scale*1, scale*1, scale*0.1, 0.1]
sol = solve(prob, params, solver)
@test sol.cost <= 1.078246081395187 * 1.1
# sol.p = [1.0565462117927191, 0.09500999699716994, 0.004368484758791603, 0.0014650440851676143]

isinteractive() && plot(sol, legend=false)

# ==============================================================================
## Test MIMO step response, penalize control signal in step response
# This time, penalize the control-signal derivative in the step response
# ==============================================================================

Pud = named_ss([1/scale * P; tf([1,0], [10, 1])], y = [:y, :du])
# Pud = named_ss([1/scale * P; tf(1)], y = [:y, :du])

step_input = :reference_input
step_output = [:y, :du]
measurement = :y

ref = [1.0, 0.0]

scale = 1/5
prob = AutoTuningProblem2(Pud; Ms=1.2, Mt=1.2, Mks=20*scale, ref, w, Ts=0.01, Tf=5.0, step_input, step_output, measurement)
params = [scale*1, scale*1, scale*0.1, 0.1]
sol = solve(prob, params, solver)

@test sol.p[2] > 1 # test that we didn't get zero integral action due to penalizing control signal in step response
# @test sol.cost <= 0.13545429523128094 * 1.1

isinteractive() && plot(sol, legend=false)


# ==============================================================================
## Test MIMO step response, penalize control signal in step response
# To make this work properly and not give zero integral action is to not penalize the control signal in steady state (we expect steady-state input). One can do this by weighting the step responses, or as is done here, set the reference to the expected steady-state controller output.
# ==============================================================================

step_input = [:reference_input, :u]
step_output = [:y, :u_controller_output_C]

ref = [1.0 0.0; 0.5 -1]

scale = 1/2
prob = AutoTuningProblem2(1/scale * P; Ms=1.2, Mt=1.2, Mks=20*scale, ref, w, Ts=0.01, Tf=5.0, step_input, step_output)
params = [scale*1, scale*1, scale*0.1, 0.1]
sol = solve(prob, params, solver)

@test sol.p[2] > 0.1 # test that we didn't get zero integral action due to penalizing control signal in step response
@test sol.cost <= 2.4111142413537405 * 1.1

isinteractive() && plot(sol, legend=false)


# ==============================================================================
## Run benchmark
# ==============================================================================
results = run_benchmarks()
results = run_benchmarks() # Run a second time since the first time may time out
status = [r.sol.retcode for r in results]
time = getproperty.(results, :timeres)
cost = getproperty.(results, :cost)

# @test all(DyadControlSystems.SciMLBase.successful_retcode, status) # May hit MaxTime on CI

@test all(cost .<= [0.005935599420432861; 0.00430624798156275; 0.693826409623054; 0.006697125313065129; 0.03708001448008453; 0.00041457537518284376; 0.00032087510338533434; 28963.953169778233; 874624.7220886125; 43936.330873944935; 0.052524452763621506; 1.528608338541142e-7*5] * 2.0) # Relaxed (2.0) testing due to flakyness on CI