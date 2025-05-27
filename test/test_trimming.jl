using ModelingToolkit, DyadControlSystems, OrdinaryDiffEq, Optimization, Test, Symbolics
using OptimizationOptimJL
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq.SciMLBase: successful_retcode

Ind_ref = 2.0
Res_ref = 200.0
weight = 0.01

@parameters t
# Includes functions to build various ODESystems
# Function to be used with testset 1
function OilBore(;name)
    @info "Oil bore direct setup"
    pars = @parameters begin
        Aperture
        Area
        PowerCoefficient
    end
    vars = @variables begin 
        Column(t)
        Power(t)
    end
    D = Differential(t)
    eqs = [
        D(Column) ~ (-Aperture/Area)*sqrt(Column) + (PowerCoefficient/Area)*Power
    ]
    ODESystem(eqs, t, vars, pars; defaults = Dict(Power => 0.8, Aperture => 2.0, Area => 20.0, PowerCoefficient => 5.0), name = name)
end

# Function to be used with testset 2
function ProtonAttractor_Direct(;name)
    @info "ProtonAttractor - direct set-up"
    ps = @parameters g=9.81 attraction_factor=0.1*weight
    @variables t
    sts = @variables radius(t) i(t) V(t)=1600*weight
    D=Differential(t)
    eqns = [
    D(D(radius)) ~ g - (attraction_factor*i^2)/(0.1*radius)
    D(i) ~ V/(Ind_ref * weight) - i*((Res_ref*weight)/(Ind_ref * weight))
    ]
    ODESystem(eqns, t, sts, ps; name = name)
end

# Functions below are for test sets 3 - 5
function ProtonAttractor_Single(;name)
    @info "Proton single ind"
    V = 1400*weight
    D = Differential(t)
    @named R1 = Resistor(; R = Res_ref*weight)
    @named L1 = Inductor(; L = Ind_ref * weight)
    @named voltage = Voltage()
    @named input = RealInput()
    @named output = RealOutput()
    @named ground = Ground()
    rl_source_eqns = [
            Symbolics.connect(input, voltage.V) 
            Symbolics.connect(voltage.p, R1.p)
            Symbolics.connect(R1.n, L1.p)
            Symbolics.connect(L1.n, voltage.n)
            Symbolics.connect(ground.g, voltage.n)
            output.u ~ L1.i 
            ]
    ODESystem(rl_source_eqns, t, systems=[R1, L1, voltage, input, output, ground]; name = name)
end
function ProtonAttractor_Double(;name)
    @info "Proton double Ind"
    V = 1400*weight
    D = Differential(t)
    @named R1 = Resistor(; R = Res_ref*weight)
    @named L1 = Inductor(; L = 0.5*Ind_ref * weight)
    @named L2 = Inductor(; L = 0.5*Ind_ref * weight )
    @named voltage = Voltage()
    @named input = RealInput()
    @named output = RealOutput()
    @named ground = Ground()
    rl_source_eqns = [
            Symbolics.connect(input, voltage.V) 
            Symbolics.connect(voltage.p, R1.p)
            Symbolics.connect(R1.n, L1.p)
            Symbolics.connect(L1.n, L2.p)
            Symbolics.connect(L2.n, voltage.n)
            Symbolics.connect(ground.g, voltage.n)
            output.u ~ L1.i 
            ]
    ODESystem(rl_source_eqns, t, systems=[R1, L1, L2, voltage, input, output, ground]; name = name)
end
"""
    2nd order ODESystem representing movement of proton.
"""
function Proton(; name, radius_start=0.0)
    @named input = RealInput()
    @named output = RealOutput() 
    ball_vars = @variables radius(t)=radius_start vs(t)=0.0
    ball_params = @parameters g=9.81 attraction_factor=0.1*weight
    D = Differential(t)
    ball_eqn = [
        vs ~ D(radius)
        0.1*D(vs) ~ 0.1*g - attraction_factor*(input.u)^2/radius
        output.u ~ radius
    ]
    compose(ODESystem(ball_eqn, t, ball_vars, ball_params; name), [input, output]) # Notation
end 

# Testsets below: 

@testset "Testset1: Oil Bore nonlinear" begin
    @parameters t
    D = Differential(t)
    @named oil_bore = OilBore()
    @named full_sys = ODESystem([], t, systems = [oil_bore])
    test_sol_dict = Dict(oil_bore.Column => 4.0, oil_bore.Power => 0.8, D(oil_bore.Column) => 0.0)
    
    @info "Setting desired states"
    desired_states = Dict(oil_bore.Column => 4.0)
    inputs = [oil_bore.Power]
    sol, trim_states = trim(full_sys; desired_states = desired_states, inputs = inputs)
    @test SciMLBase.successful_retcode(sol)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 5e-2, atol = 1e-6)
    end

    @info "Imposing one hard-constraint"
    hard_eq_cons =  [
        oil_bore.Power ~ 1.0
    ]
    sol, trim_states = trim(full_sys; hard_eq_cons = hard_eq_cons, desired_states = desired_states, inputs = inputs)
    @test SciMLBase.successful_retcode(sol)
    test_sol_dict = Dict(oil_bore.Power => 1.0, D(oil_bore.Column) => 0.0)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 5e-2, atol = 1e-6)
    end

    @info "Two hard-constraints including derivative. Should not be over-constrained"
    hard_eq_cons =  [
        oil_bore.Power ~ 1.0
        D(oil_bore.Column) ~ 4.7*weight
        ]
    sol, trim_states = trim(full_sys; hard_eq_cons = hard_eq_cons, desired_states = desired_states, inputs = inputs)
    @test SciMLBase.successful_retcode(sol)
    test_sol_dict = Dict(oil_bore.Power => 1.0, D(oil_bore.Column) => 4.7*weight)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 5e-2, atol = 1e-6)
    end

    @info "Testing with inequality constraints"
    hard_ineq_cons =  [
        oil_bore.Power - 0.9 ≲ 0.0
    ]
    soft_ineq_cons =  [
        -oil_bore.Column + 6 ≲ 0.0 
    ]
    sol, trim_states = trim(full_sys; hard_ineq_cons = hard_ineq_cons, soft_ineq_cons = soft_ineq_cons, desired_states = Dict(oil_bore.Column => 6.0, oil_bore.Power => 0.9), inputs = inputs, abstol = 0.0, solver = IPNewton())
    test_sol_dict = Dict(oil_bore.Column => 5.06, oil_bore.Power => 0.9)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 2e-1, atol = 1e-6) # Weird solutions from optimizer. 
    end

    @info "Testing with equality and hard & soft inequality constraints"
    desired_states = Dict(oil_bore.Column => 4.0, D(oil_bore.Column) => 4.7*weight)
    hard_eq_cons =  [
        oil_bore.Power ~ 1.0
        
        ]
    hard_ineq_cons =  [
        oil_bore.Power - 1.1 ≲ 0.0
    ]
    soft_ineq_cons =  [
        -oil_bore.Column + 6 ≲ 0.0 
    ]
    penalty_multipliers = Dict(:desired_states => 1.0, :trim_cons => 5.0, :sys_eqns => 10.0)
    dualize = true
    sol, trim_states = trim(full_sys; desired_states = desired_states, hard_eq_cons = hard_eq_cons, hard_ineq_cons = hard_ineq_cons, soft_ineq_cons = soft_ineq_cons, inputs = inputs, dualize = dualize, penalty_multipliers = penalty_multipliers, abstol = 0.0, solver = IPNewton())
    test_sol_dict = Dict(oil_bore.Column => 6.0, oil_bore.Power => 1.0)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 2e-1, atol = 1e-6) # Weird solutions from optimizer. 
    end
end

@testset "Testset 2: Proton attractor direct" begin
    @parameters t
    D = Differential(t)
    @named proton_attractor_1 = ProtonAttractor_Direct()
    @named full_sys = ODESystem([], t, systems = [proton_attractor_1])
    inputs = [proton_attractor_1.V]
    test_sol_dict = Dict(proton_attractor_1.radius => 5*weight, proton_attractor_1.V => 14, D(proton_attractor_1.radius) => 0.0)

    @info "Setting desired states"
    desired_states = Dict(proton_attractor_1.radius => 5*weight, proton_attractor_1.V => 14.5)
    penalty_multipliers = Dict(:desired_states => 1000.0) # Note this high penalty is important to get a suitable local optimum with IPNewton()
    sol, trim_states = trim(full_sys; desired_states = desired_states, inputs = inputs, penalty_multipliers = penalty_multipliers)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 5e-2, atol = 1e-6)
    end

    test_sol_dict = Dict(proton_attractor_1.radius => 10.0, proton_attractor_1.V => 198.0)
    desired_states = Dict(proton_attractor_1.radius => 10.0, proton_attractor_1.V => 200.0)
    sol, trim_states = trim(full_sys; desired_states = desired_states, inputs = inputs, penalty_multipliers = penalty_multipliers)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 5e-2, atol = 1e-6)
    end

    @info "Testing trim with hard and soft inequality constraints"
    hard_ineq_cons = [
        proton_attractor_1.V - 12 ≲ 0.0
    ]
    soft_ineq_cons = [
        proton_attractor_1.i - 6 ≲ 0.0
    ]
    desired_states = Dict(proton_attractor_1.radius => 5*weight, proton_attractor_1.V => 12.0)
    dualize = true
    penalty_multipliers = Dict(:desired_states => 1000.0, :trim_cons => 100, :sys_eqns => 1000)
    sol, trim_states = trim(full_sys; desired_states, inputs, hard_ineq_cons, soft_ineq_cons, dualize, penalty_multipliers, abstol = 0.0, solver = IPNewton())
    test_sol_dict = Dict(proton_attractor_1.V => 12.0)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 5e-2, atol = 1e-6)
    end
end

@testset "Testset3: Proton-Attractor Single composable" begin
    @parameters t
    @named rl_model1 = ProtonAttractor_Single()
    @named proton_single = Proton(radius_start = 5*weight)
    radius_set = 5*weight
    @named ref_radius_comp = Constant(; k=radius_set) 
    @named sat_pid = LimPID(k = 3.0, Ti = 0.5, Td = 100.0, u_max = 10.0, u_min = -10.0, Ni = 14.1)
    control_eqns = [
        Symbolics.connect(rl_model1.output, proton_single.input)
        Symbolics.connect(ref_radius_comp.output, sat_pid.reference)
        Symbolics.connect(proton_single.output, sat_pid.measurement)
        Symbolics.connect(sat_pid.ctr_output, rl_model1.input)
    ]
    @named full_sys = ODESystem(control_eqns, t, systems = [proton_single, rl_model1, sat_pid, ref_radius_comp])
    @info "Testing with equality constraints"
    hard_eq_cons = [
    proton_single.radius ~ 5*weight
    ]
    desired_states = Dict(rl_model1.L1.i => 700*weight)
    soft_eq_cons = [
        proton_single.vs ~ 0.001
    ]
    dualize = true
    penalty_multipliers = Dict(:desired_states => 100000000.0, :trim_cons => 10.0, :sys_eqns => 1000.0)
    sol, trim_states = trim(full_sys; hard_eq_cons = hard_eq_cons, soft_eq_cons = soft_eq_cons, desired_states = desired_states, dualize = dualize, penalty_multipliers = penalty_multipliers, abstol = 0.0, solver = IPNewton())
    test_sol_dict = Dict(proton_single.radius => 5*weight, rl_model1.L1.i => 700*weight)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 2e-1, atol = 1e-6)
    end
    
    @info "Testing with inequality constraints instead"
    hard_ineq_cons = [
    proton_single.radius ≳ 0.06
    ]
    desired_states = Dict(rl_model1.L1.i => 700*weight, proton_single.radius => 5*weight)
    soft_ineq_cons = [
        proton_single.vs ≳ 0.001
    ] # Note that this constraint won't be feasible physically.
    dualize = true
    penalty_multipliers = Dict(:desired_states => 100000000.0, :trim_cons => 10.0, :sys_eqns => 1000.0)
    sol, trim_states = trim(full_sys; hard_ineq_cons = hard_ineq_cons, soft_ineq_cons = soft_ineq_cons, desired_states = desired_states, penalty_multipliers = penalty_multipliers, dualize = dualize, abstol = 0.0, solver = IPNewton())
    test_sol_dict = Dict(proton_single.radius => 5*weight, rl_model1.L1.i => 700*weight)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 2e-1, atol = 1e-6)
    end
end

@testset "Testset4: Proton Attractor - Double Composable" begin
    @variables t
    @named rl_model2 = ProtonAttractor_Double()
    @named proton_double = Proton(radius_start = 5*weight)
    radius_set = 5*weight
    @named ref_radius_comp = Constant(; k=radius_set) 
    @named sat_pid = LimPID(k = 3.0, Ti = 0.5, Td = 100.0, u_max = 10.0, u_min = -10.0, Ni = 14.1)
    control_eqns = [
        Symbolics.connect(rl_model2.output, proton_double.input)
        Symbolics.connect(ref_radius_comp.output, sat_pid.reference)
        Symbolics.connect(proton_double.output, sat_pid.measurement)
        Symbolics.connect(sat_pid.ctr_output, rl_model2.input)
    ]
    @named full_sys = ODESystem(control_eqns, t, systems = [proton_double, rl_model2, sat_pid, ref_radius_comp])
    desired_states = Dict(proton_double.radius => 5*weight)
    soft_eq_cons = [
        proton_double.vs ~ 0.001
    ] # Note that this constraint won't be feasible physically.
    params = Dict(rl_model2.L2.L => 0.0101)
    dualize = true
    penalty_multipliers = Dict(:desired_states => 1E9, :trim_cons => 1E2, :sys_eqns => 1E4)
    sol, trim_states = trim(full_sys, soft_eq_cons = soft_eq_cons, desired_states = desired_states, params = params, penalty_multipliers = penalty_multipliers, dualize = dualize)
    test_sol_dict = Dict(proton_double.radius => 5*weight, proton_double.vs => 0.0)
    for key in keys(test_sol_dict)
        @test isapprox(trim_states[key], test_sol_dict[key], rtol = 2e-1, atol = 1e-4)
    end
end


