using ModelingToolkit, DyadControlSystems, OrdinaryDiffEq, Optimization, Test, Symbolics
using OptimizationOptimJL
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks
using Plot
using Ipopt
using OptimizationMOI

@parameters t
R = 4
L = 6
V = 12.0
@named R1 = Resistor(; R = R)
@named L1 = Inductor(; L = 0.5*L)
@named L2 = Inductor(; L = 0.5*L )
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
        output.u ~ L2.i 
        ]
@named circuit_dae = ODESystem(rl_source_eqns, t, systems=[R1, L1, L2, voltage, input, output, ground])
inputs = [input.u]
outputs = []
sys, diff_idxs, alge_idxs, input_idxs = ModelingToolkit.io_preprocessing(circuit_dae, inputs, outputs)
full_equations(sys)

tspan = (0.0, 10)
x0 = Dict(
    L1.p.i => 0.0,
    L2.v => 0.0, 
)
u0 = Dict(
    input.u => 12
)
prob = ODEProblem(sys, x0, tspan, u0, jac = true)
sol = solve(prob, Rodas4())
plot(sol, idxs = [R1.i])
plot(sol, idxs = [R1.i])
plot(sol, idxs = [L1.p.i])

desired_states = Dict(L1.i => 7)
sol, trim_states = trim(circuit_dae; inputs, outputs, desired_states)
trim_states


desired_states = Dict(L2.v => 0.1)
sol, trim_states = trim(circuit_dae; inputs, outputs, desired_states)
D = Differential(t)
hard_eq_cons = [
    D(L1.p.i) ~ 4
]

sol, trim_states = trim(circuit_dae; inputs, outputs, hard_eq_cons, desired_states, solver =Ipopt.Optimizer())