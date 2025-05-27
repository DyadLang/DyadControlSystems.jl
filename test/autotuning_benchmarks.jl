# This file benchmarks pidfIAE only
# using DyadControlSystems
using LinearAlgebra, PrettyTables, StaticArrays



function problem_1(; Ms = 1.2 )
    T = 4 # Time constant
    L = 1 # Delay
    K = 1 # Gain
    P = tf(K, [T, 1.0])#*delay(L) # process dynamics

    # Robustness consraints
    # Maximum allowed sensitivity function magnitude
    Mt = Ms  # Maximum allowed complementary sensitivity function magnitude
    Mks = 10.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
    w = 2π .* exp10.(LinRange(-2, 2, 100))

    p0 = Float64[1, 1, 0, 0]
    prob = AutoTuningProblem2(P; Ms, Mt, Mks, w, Ts=0.1, Tf=25.0)
    prob, p0
end



## Test with delay
function problem_2()
    T = 4 # Time constant
    K = 1 # Gain
    L = 1 # Delay
    P = tf(K, [T, 1.0])*delay(L) # process dynamics

    # Robustness consraints
    Ms = 1.2
    Mt = Ms  # Maximum allowed complementary sensitivity function magnitude
    Mks = 10.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
    w = 2π .* exp10.(LinRange(-2, 2, 100))

    p0 = 0.8*Float64[2, 1, 0.5, 0.1]
    prob = AutoTuningProblem2(P; Ms, Mt, Mks, w, Ts=0.1, Tf=25.0)
    prob, p0
end


## Test ub
function problem_3()
    T = 4 # Time constant
    K = 1 # Gain
    P = tf(K, [T, 1.0])

    # Robustness consraints
    Ms = 1.2 # Maximum allowed sensitivity function magnitude
    Mt = Ms  # Maximum allowed complementary sensitivity function magnitude
    Mks = 10.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
    w = 2π .* exp10.(LinRange(-2, 2, 100))

    prob = AutoTuningProblem2(P; Ms, Mt, Mks, w, Ts=0.1, Tf=25.0, ub=[Inf, 10, Inf, Inf])
    p0 = Float64[1,0,0,0]
    prob, p0
end



## Another test case
# This test case with this initial guess fails to find a stabilizing controller after the first iteration.
# This is used to test the heuristic to increase Mks to find a stabilizing controller and then lower it again
function problem_4()
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
    prob = AutoTuningProblem2(P; Ms, Mt, Mks, w, Ts=0.01, Tf=55.0)
    prob, p0
end



## Unstable process
# Unstable processes require finding an initial stabilizing controller
# This example comes from the paper

function problem_5()
    P = ss(zpk([], [1,-10], 10))

    Ms = 1.4 # Maximum allowed sensitivity function magnitude
    Mt = 1.4  # Maximum allowed complementary sensitivity function magnitude
    Mks = 100.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
    w = exp10.(LinRange(-3, 3, 100))
    p0 = Float64[6, 1, 0, 0.001]
    prob = AutoTuningProblem2(P; Ms, Mt, Mks, w, Ts=0.01, Tf=10.0)
    prob, p0
end

function problem_6()
    P = DemoSystems.double_mass_model()
    Ms = 1.3 # Maximum allowed sensitivity function magnitude
    Mt = 1.3  # Maximum allowed complementary sensitivity function magnitude
    Mks = 1000.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
    w = exp10.(LinRange(-3, 4, 200))
    # p0 = Float64[10, 0.1, 10, 0.001]
    p0 = Float64[1, 0, 1, 0.001]
    prob = AutoTuningProblem2(P; Ms, Mt, Mks, w, Ts=0.001, Tf=10.0)
    prob, p0
end

function problem_7()
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
            
    Ms = 1.5 # Maximum allowed sensitivity function magnitude
    Mt = 1.5  # Maximum allowed complementary sensitivity function magnitude
    Mks = 20.0 # Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.
    timeweight = true
    p0 = [ControlSystemsBase.convert_pidparams_from_standard(1,2000,1, :parallel)..., Tf]
    AutoTuningProblem2(P; Ms, Mt, Mks, w, Ts=2, Tf=2*60*60, timeweight), p0
end

function problem_8(method=3)
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

    # Copt = let
    #     CA = [0.0 4.0 0.0; 0.0 0.0 32.0; 0.0 -62.1902026690993 -63.08861205338373]
    #     CB = [0.0; 0.0; 32.0;;]
    #     CC = [2.375526272770401 3.1411399244248353 50.457924779299645]
    #     CD = [0.0;;]
    #     named_ss(ss(CA, CB, CC, CD), x=[:optimizedx1, :optimizedx2, :optimizedx3], u=[:optimizedu], y=[:optimizedy])
    # end

    dcg = dcgain(minreal(sminreal(P[:output, :force])))
    P = inv(dcg) * P 

    Ts = 0.01
    Tf = 10.0
    Ms = 1.3
    Mt = 1.2
    Mks = 3e1
    disc = :tustin
    tvec = 0:Ts:Tf
    w = exp10.(LinRange(-1, 3, 300))


    params = [0.0010000000000000007, 0.00020000000000000015, 0.020000000000000014, 0.001]

    measurement = P.y[1]
    control_input = :force
    step_output = P.y[1]
    ref = 0.0
    if method == 1 # new approach
        step_input = :road
        response_type = impulse
        timeweight = false
    elseif method == 2 # new approach with realistic disturbance input
        step_input = :road
        u = @. cos(2*tvec) * exp(-0.5*tvec)#; plot(u)
        response_type = (G, t) -> lsim(G, u', t)
        timeweight = false
    else method == 3 # Old approach
        step_input = :force
        response_type = step
        timeweight = false
    end

    # using MadNLP
    # solver = MadNLP.Optimizer()

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
    prob, params
end

function problem_9()
    # Active suspension model
    # poorly scaled, dcgain 1e-3
    # TODO: it would be nice if this problem could be solved well without providing the lb that forces a large integrator gain

    P = let
        tempA = [0.0 0.0 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 0.0; -1010.0 1000.0 -1001.0 -0.0 1.0 -0.0; 0.0 0.0 0.0 0.0 0.0 1.0; 10.0 -11.0 0.01 1.0 -0.011 0.001; 0.0 10.0 0.0 -10.0 0.01 -0.01]
        tempB = [0.0; 0.0; 0.0; 0.0; -0.001; 0.01;;]
        tempC = [0.0 0.0 0.0 1.0 0.0 0.0]
        tempD = [0.0;;]
        named_ss(ss(tempA, tempB, tempC, tempD), x=[Symbol("wheel₊mass₊s(t)"), Symbol("car_and_suspension₊mass₊s(t)"), Symbol("wheel₊mass₊v(t)"), Symbol("seat₊mass₊s(t)"), Symbol("car_and_suspension₊mass₊v(t)"), Symbol("seat₊mass₊v(t)")], u=[:u], y=[:y])
    end
    w = exp10.(LinRange(-2, 3, 300))
    prob = AutoTuningProblem2(P;
        w,
        measurement = :y,
        control_input = :u,
        step_input = :u,
        step_output = :y,
        ref = 0.0,
        Ts = 0.01,
        Tf = 10.0,
        Ms = 1.2,
        Mt = 1.1,
        Mks = 3e4,
        lb = [0.0, 2000, 0, 0],
    )
    p0 = [100, 2000, 200, 0.01] # DyadControlSystems.initial_guess(prob)
    prob, p0
end


## Run =========================================================================

function run_benchmarks()
    algs = [
        IpoptSolver(
            exact_hessian = false,
            verbose = false,
            max_iter = 1000,
            acceptable_iter = 10,
            tol             = 1e-5,
            acceptable_tol  = 1e-3,
            constr_viol_tol = 1e-3,
            acceptable_constr_viol_tol = 2e-2,
        )
    ]

    problems = [
        problem_1
        ()->problem_1(; Ms = 1.5)
        problem_2# delay system
        problem_3
        problem_4
        problem_5
        problem_6
        problem_7
        ()->problem_8(1)
        ()->problem_8(2)
        ()->problem_8(3)
        problem_9
    ]


    results = map(Iterators.product(problems, algs)) do (problem, alg)
        @show problem
        prob, p0 = problem()
        # @show p0 = initial_guess(prob)
        res = solve(prob, p0, alg; maxtime=10.0)
    end

    stringtype(x) = string(typeof(x))

    status = [r.sol.retcode for r in results]
    # pretty_table(status, header = stringtype.(algs), title="Status")

    time = getproperty.(results, :timeres)
    # pretty_table(time, header = stringtype.(algs), title="Elapsed time")

    cost = getproperty.(results, :cost)
    # pretty_table(cost, header = stringtype.(algs), title="Cost")
    results
end

#=
julia> run_benchmarks()
[ Info: Approximating delay with a Pade approximation of order 3
Status
┌─────────────────┐
│ Ipopt.Optimizer │
├─────────────────┤
│         Success │
│         Success │
│         Success │
│         Success │
│         Success │
│         Success │
│         Success │
│         Success │
│         Success │
│         Success │
│         Success │
└─────────────────┘
Elapsed time
┌─────────────────┐
│ Ipopt.Optimizer │
├─────────────────┤
│        0.112013 │
│       0.0777813 │
│       0.0494965 │
│        0.100023 │
│          0.2935 │
│       0.0860699 │
│         0.45735 │
│        0.476434 │
│        0.593155 │
│         1.65548 │
│        0.329027 │
└─────────────────┘
Cost
┌─────────────────┐
│ Ipopt.Optimizer │
├─────────────────┤
│      0.00648008 │
│      0.00702586 │
│        0.693797 │
│      0.00675276 │
│       0.0366094 │
│     0.000323861 │
│     0.000185111 │
│         28964.0 │
│       8.74625e5 │
│         43936.3 │
│       0.0524781 │
└─────────────────┘

=#