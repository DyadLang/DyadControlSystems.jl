#=
TODO: 
- Multivariate analysis points
- Plot problem in callback
- Uncertain models through expanding the operating points
- Loop openings in get_sensititity etc.
- Cascade PID loop example
- Up docs
- Only plot the relevant nonlinear signal from sol depending on the output-analysis point
- ForwardDiff does not work due to MTK.linearize trying to call Float64(Dual)
=#
## Test evaluation
using DyadControlSystems
using ModelingToolkit
using ModelingToolkitStandardLibrary
using ModelingToolkitStandardLibrary.Blocks
using Optimization
# using OptimizationOptimJL
using Optim
using OptimizationMOI
using Ipopt
using OrdinaryDiffEq
using DyadControlSystems.CommonSolve
using DyadControlSystems.MPC
using Printf
using Plots
default(size=(1000, 800), titlefontsize=10)
using Test

# function getdefault(x)
#     x = Symbolics.unwrap(x)
#     p = Symbolics.getparent(x, nothing)
#     p === nothing || (x = p)
#     Symbolics.getmetadata(x, Symbolics.VariableDefaultValue, 0.0)
# end

##

w = exp10.(LinRange(0, 3, 200))
t = 0:0.001:0.21

sys = DyadControlSystems.ControlDemoSystems.dcmotor()
syss = structural_simplify(sys)
sysc = complete(sys)

# prob = ODEProblem(syss, [], (0.0, 10.0))



tunable_parameters = [
    sysc.pi_controller.gainPI.k => (1e-9, 100.0)
    sysc.pi_controller.int.k    => (2.0, 1e2)
]

defs = Dict(sysc.L1.p.i => 0.0, sysc.pi_controller.u_max => Inf) # The inf is to avoid the controller saturating at time t=0 due to the step input.
operating_points = [ # Can be one or several operating points
    defs
]

op = operating_points[1]

# m2 =  get_sensitivity(sys, :y; op)[1]
# bodeplot(sminreal(ss(m2...)))

##
WS    = tf([1.5, 0], [1, 50])
ω     = 2pi*20.0
Gref  = tf(ω^2, [1, 2ω, ω^2]) * tf(1, [0.001, 1])
sto   = StepTrackingObjective(reference_model=Gref, tolerance=0.05, input=:r, output=:y)
mso   = MaximumSensitivityObjective(WS, :y)
oo    = OvershootObjective(1.05, :r, :y)
seto  = SettlingTimeObjective(; final_value = 1.0, time = 0.025, tolerance = 0.09, input=:r, output=:y)
rto   = RiseTimeObjective(min_value = 0.91, time = 0.025, input=:r, output=:y)
gmo   = GainMarginObjective(2, :y)
pmo   = PhaseMarginObjective(45, :y)

objectives = [
    sto,
    seto,
    mso,
    oo,
    rto,
    gmo,
    pmo,
    # StepRejectionObjective(tf([1, 0], [1,2,1])),
    # GainMarginObjective(2),
    # PhaseMarginObjective(45),
]


# plot(step(Gref, t))
# plot(objectives, layout=length(objectives))

prob = DyadControlSystems.StructuredAutoTuningProblem(sys, w, t, objectives, operating_points, tunable_parameters)
plot(prob)

x0 = [1.0, 10]
x0 = [2.649585, 11.7084]
x0 = [1.0, 20]
# x0 = [1.7953184486861322, 9.337518201151585]
# callback = (x, l)->(@printf("loss = %4.4g, \n", l); false)

@time res = solve(prob, x0,
    MPC.IpoptSolver(verbose=isinteractive(), exact_hessian=false, acceptable_iter=4, tol=1e-3, acceptable_tol=1e-2, max_iter=100);
    # outer_iters = 2,
    # NelderMead();
    # ParticleSwarm();
    # callback,
)
# 5.989143 seconds (56.83 M allocations: 2.724 GiB, 4.76% gc time)
# 2.061692 seconds (19.74 M allocations: 961.547 MiB, 4.12% gc time) with cache
# 0.348771 seconds (2.84 M allocations: 124.540 MiB, 10.33% gc time) cache linearization function

plot(res)

obj_vals = [last(s) for s in res.objective_status[1]]
@test all(obj_vals .< 0.5)


## Use MaximumSensitivity and MaximumTransfer for CS
WS    = tf([1.4, 0], [1, 50])
mso   = MaximumSensitivityObjective(WS, :y)
mto   = MaximumTransferObjective(tf(1.2), :y, :u) # CS

objectives = [mso, mto]
prob = DyadControlSystems.StructuredAutoTuningProblem(sys, w, t, objectives, operating_points, tunable_parameters)

# plot(prob)
#
x0 = [1.0, 20]
@time res = solve(prob, x0,
    MPC.IpoptSolver(verbose=isinteractive(), exact_hessian=false, acceptable_iter=4, tol=1e-4, acceptable_tol=1e-3, max_iter=300, mu_strategy="monotone");
    outer_iters = 1,
    # NelderMead()
)
obj_vals = [last(s) for s in res.objective_status[1]]
@test all(obj_vals .< 0.2)

isinteractive() && plot(res)


# ==============================================================================
## Adding a P controller for the position ======================================
# ==============================================================================

# First we try without velocity feedforward so that we may test the sensitivity calculations with loop openings
sys_inner = DyadControlSystems.ControlDemoSystems.dcmotor(ref=nothing)
@named ref = Blocks.Step(height = 1, start_time = 0)
@named p_controller = Blocks.Gain(10.0)
@named outer_feedback = Blocks.Feedback()

connect = ModelingToolkit.connect
connections = [
    connect(ref.output, :r, outer_feedback.input1)
    connect(sys_inner.angle_sensor.phi, :yp, outer_feedback.input2)
    connect(outer_feedback.output, :ep, p_controller.input)
    connect(p_controller.output, :up, sys_inner.feedback.input1)
]


@named closed_loop = ODESystem(connections, ModelingToolkit.get_iv(sys_inner); systems = [sys_inner, ref, p_controller, outer_feedback])
cc = complete(closed_loop)
op2 = Dict(cc.dcmotor.L1.i => 0, cc.dcmotor.pi_controller.u_max => Inf)
m1 = get_sensitivity(closed_loop, closed_loop.dcmotor.y; loop_openings=[:yp], op=op2)[1]
m2 =  get_sensitivity(sys, :y; op)[1]
@test sminreal(ss(m1...)) == sminreal(ss(m2...))


## =============================================================================
# We now add velocity feedforward

@named ref            = Blocks.Step(height = 1, start_time = 0)
@named ref_diff       = Blocks.Derivative(T=0.1)
@named add            = Blocks.Add()
@named p_controller   = Blocks.Gain(10.0)
@named outer_feedback = Blocks.Feedback()
@named id = Blocks.Gain(1.0) # a trivial identity element to allow us to place the analysis point :r in the right spot

connect = ModelingToolkit.connect
connections = [
    connect(ref.output, :r, id.input)
    connect(id.output, outer_feedback.input1, ref_diff.input)
    connect(ref_diff.output, add.input1)
    connect(add.output, sys_inner.feedback.input1)
    connect(p_controller.output, :up, add.input2)
    connect(sys_inner.angle_sensor.phi, :yp, outer_feedback.input2)
    connect(outer_feedback.output, :ep, p_controller.input)
]

@named closed_loop = ODESystem(connections, ModelingToolkit.get_iv(sys_inner); systems = [sys_inner, ref, id, ref_diff, add, p_controller, outer_feedback])
cl = complete(closed_loop)


simprob = ODEProblem(structural_simplify(cl), [cl.dcmotor.L1.i => 0], (t[1], 1))
sol = solve(simprob, Tsit5())
plot(sol)
plot!(sol, idxs=[
    cl.dcmotor.pi_controller.ctr_output.u
    sys.angle_sensor.phi.u
    ])



##
tunable_parameters = [
    cl.dcmotor.pi_controller.gainPI.k => (1e-1, 10.0)
    cl.dcmotor.pi_controller.int.k    => (2.0, 1e2)
    cl.p_controller.k    => (1e-2, 1e2)
]

operating_points = [ # Can be one or several operating points
    Dict(cl.dcmotor.L1.i => 0.0, cl.dcmotor.pi_controller.u_max => Inf)
]

ωp     = 2pi*2.0
Pref  = tf(ωp^2, [1, 2ωp, ωp^2])
stp   = StepTrackingObjective(reference_model=Pref, tolerance=0.05, input=:r, output=:yp)
mso   = MaximumSensitivityObjective(weight=WS, output=closed_loop.dcmotor.y, loop_openings=[:yp])
objectives = [
    stp,
    mso,
]


w = exp10.(LinRange(0, 3, 200))
t = 0:0.005:2
prob = DyadControlSystems.StructuredAutoTuningProblem(closed_loop, w, t, objectives, operating_points, tunable_parameters)

x0 = [1.0, 20, 0.1]
callback = (x, l)->(@printf("loss = %4.4g, \n", l); false)
@time res = solve(prob, x0,
    MPC.IpoptSolver(verbose=isinteractive(), exact_hessian=false, acceptable_iter=4, tol=1e-3, acceptable_tol=1e-2, max_iter=300);
    # NelderMead();
    # ParticleSwarm();
    # callback,
)

obj_vals = [last(s) for s in res.objective_status[1]]
@test all(obj_vals .< [0.15, 0.05])

plot(res)

##

op = res.op[1]
Gv  = named_ss(closed_loop, :r, closed_loop.dcmotor.y; op)
Gp  = named_ss(closed_loop, :r, :yp; op)
Gvo = named_ss(closed_loop, :r, closed_loop.dcmotor.y; op, loop_openings=[:yp])
Gpo = named_ss(closed_loop, :r, :yp; op, loop_openings=[closed_loop.dcmotor.y])

@test Gv != Gvo
@test Gp != Gpo

w = exp10.(LinRange(-2, 2, 200))
bodeplot([Gv, Gp, Gvo, Gpo], w, lab=["V" "P" "VO" "PO"], plotphase=false)

op = res.op[1]
bodeplot([
    ss(get_sensitivity(closed_loop, closed_loop.dcmotor.y; op)[1]...),
    ss(get_sensitivity(closed_loop, closed_loop.dcmotor.y; op, loop_openings=[:yp])[1]...)
    ], plotphase = false,
    lab    = ["Position loop closed" "Position loop open"],
    title  = "Sensitivity at velocity output", size=(500, 300),
    legend = :bottomright
)

# ==============================================================================
## Test uncertain operating points =============================================
# ==============================================================================
import DyadControlSystems.RobustAndOptimalControl.MonteCarloMeasurements
if !isdefined(Main, :MCM)
    MCM = MonteCarloMeasurements
end
sys = DyadControlSystems.ControlDemoSystems.dcmotor()
syss = structural_simplify(sys)
sysc = complete(sys)

N = 10 # Number of samples for the uncertain parameters
op = Dict()
op[sysc.inertia.J] = MCM.Particles(N, MCM.Uniform(0.01, 0.03))
op[sysc.L1.i] = 0.0
op[sysc.pi_controller.u_max] = 1000.0
op[sysc.pi_controller.u_min] = -1000.0
operating_points = [op]
# operating_points = DyadControlSystems.expand_uncertain_operating_points(op)
# operating_points[1][sysc.inertia.J]

w = exp10.(LinRange(0, 3, 200))
t = 0:0.001:0.21

tunable_parameters = [
    sysc.pi_controller.gainPI.k => (1e-9, 100.0)
    sysc.pi_controller.int.k    => (2.0, 1e2)
    # sysc.pi_controller.T    => (1/1e2, 1/2.0)
]


##
WS    = tf([1.5, 0], [1, 40])
ω     = 2pi*20.0
Gref  = tf(ω^2, [1, 2ω, ω^2]) * tf(1, [0.001, 1])
sto   = StepTrackingObjective(reference_model=Gref, tolerance=0.05, input=:r, output=:y)
mso   = MaximumSensitivityObjective(WS, :y)
oo    = OvershootObjective(1.06, :r, :y)
seto  = SettlingTimeObjective(; final_value = 1.0, time = 0.07, tolerance = 0.12, input=:r, output=:y)
rto   = RiseTimeObjective(min_value = 0.88, time = 0.07, input=:r, output=:y)

objectives = [
    sto,
    seto,
    mso,
    oo,
    rto,
]

prob = DyadControlSystems.StructuredAutoTuningProblem(sys, w, t, objectives, operating_points, tunable_parameters)

x0 = [1.0, 10]
x0 = [2.649585, 11.7084]
x0 = [1.0, 20]
# x0 = [1.3975369072223867, 21.63274830678775]
# x0 = [1.7953184486861322, 9.337518201151585]
# callback = (x, l)->(@printf("loss = %4.4g, \n", l); false)

@time res = solve(prob, x0,
    MPC.IpoptSolver(verbose=isinteractive(), exact_hessian=false, acceptable_iter=5, tol=1e-3, acceptable_tol=1e-2, max_iter=150);
)

lsys = sminreal(named_ss(sys, :r, :y; op=res.op[1]))
@test lsys.sys != ss(0)

S = ss(get_sensitivity(sys, :y; op=res.op[1])[1]...)

obj_vals = [last(s) for s in res.objective_status[1]]
@test all(obj_vals .< 0.5)
plot(res)

# A good optimum
# pi_controller₊k => 1.0001143079608283
# pi_controller₊int₊k => 20.000008698304956
# objective value     : 0.11820730786474624


# tup = res.cost_functions[2].lin_fun(zeros(4), res.op[1], 0)
# ss(tup.f_x, tup.f_u, tup.h_x, tup.h_u)



# ==============================================================================
## Mixing tank
# ==============================================================================

connect = ModelingToolkit.connect;
t = ModelingToolkit.t_nounits;
D = ModelingToolkit.D_nounits;
rc = 0.25 # Reference concentration 

@mtkmodel MixingTank begin
    @parameters begin
        c0 = 0.8, [description = "Nominal concentration"]
        T0 = 308.5, [description = "Nominal temperature"]
        a1 = 0.2674
        a21 = 1.815
        a22 = 0.4682
        b = 1.5476
        k0 = 1.05e14
        ϵ = 34.2894
    end
    @variables begin
        gamma(t), [description = "Reaction speed"]
        xc(t) = c0, [description = "Concentration"]
        xT(t) = T0, [description = "Temperature"]
        xT_c(t), [description = "Cooling temperature"]
    end
    @components begin
        T_c = RealInput()
        c = RealOutput()
        T = RealOutput()
    end
    begin
        τ0 = 60
        wk0 = k0 / c0
        wϵ = ϵ * T0
        wa11 = a1 / τ0
        wa12 = c0 / τ0
        wa13 = c0 * a1 / τ0
        wa21 = a21 / τ0
        wa22 = a22 * T0 / τ0
        wa23 = T0 * (a21 - b) / τ0
        wb = b / τ0
    end
    @equations begin
        gamma ~ xc * wk0 * exp(-wϵ / xT)
        D(xc) ~ -wa11 * xc - wa12 * gamma + wa13
        D(xT) ~ -wa21 * xT + wa22 * gamma + wa23 + wb * xT_c
        xc ~ c.u
        xT ~ T.u
        xT_c ~ T_c.u
    end
end
begin
    Ftf = tf(1, [(100), 1])^2
    Fss = ss(Ftf)
    # Create an MTK-compatible constructor 
    function RefFilter(; name)
        sys = ODESystem(Fss; name)
        "Compute initial state that yields y0 as output"
        empty!(ModelingToolkit.get_defaults(sys))
        return sys
    end
end
@mtkmodel InverseControlledTank begin
    begin
        c0 = 0.8    #  "Nominal concentration
        T0 = 308.5 #  "Nominal temperature
        x10 = 0.42
        x20 = 0.01
        u0 = -0.0224
        c_start = c0 * (1 - x10) # Initial concentration
        T_start = T0 * (1 + x20) # Initial temperature
        c_high_start = c0 * (1 - 0.72) # Reference concentration
        T_c_start = T0 * (1 + u0) # Initial cooling temperature
    end
    @components begin
        ref = Constant(k = 0.25) # Concentration reference
        ff_gain = Gain(k = 1) # To allow turning ff off
        controller = PI(gainPI.k = 10, T = 500)
        tank = MixingTank(xc = c_start, xT = T_start, c0 = c0, T0 = T0)
        inverse_tank = MixingTank(xc = nothing, xT = T_start, c0 = c0, T0 = T0)
        feedback = Feedback()
        add = Add()
        filter = RefFilter()
        noise_filter = FirstOrder(k = 1, T = 1, x = T_start)
        # limiter = Gain(k=1)
        limiter = Limiter(y_max = 370, y_min = 250) # Saturate the control input 
    end
    @equations begin
        connect(ref.output, :r, filter.input)
        connect(filter.output, inverse_tank.c)
        connect(inverse_tank.T_c, ff_gain.input)
        connect(ff_gain.output, :uff, limiter.input)
        connect(limiter.output, add.input1)
        connect(controller.ctr_output, :u, add.input2)
        connect(add.output, :u_tot, tank.T_c)
        connect(inverse_tank.T, feedback.input1)
        connect(tank.T, :y, noise_filter.input)
        connect(noise_filter.output, feedback.input2)
        connect(feedback.output, :e, controller.err_input)
    end
end;
@named model = InverseControlledTank()
ssys = structural_simplify(model, split=true)
cm = complete(model)

# op = Dict(
#     D(cm.inverse_tank.xT) => 1,
#     cm.tank.xc => 0.65
# )

op = Dict(
    cm.filter.y.u => 0.8 * (1 - 0.42),
    # D(cm.filter.y.u) => 0
)

tspan = (0.0, 1000.0)
prob = ODEProblem(ssys, op, tspan)
sol = solve(prob, Rodas5P())#, initializealg=ShampineCollocationInit())

# plot(sol, layout=8) |> display
@assert SciMLBase.successful_retcode(sol)

plot(sol, idxs=[cm.tank.xc, cm.tank.xT, cm.controller.ctr_output.u], layout=3, sp=[1 2 3])
hline!([ModelingToolkit.getp(prob, cm.ref.k)(prob)], label="ref", sp=1)



begin
	wvec = exp10.(LinRange(-4, 2, 200))
	tvec = 0:5:2000
	
	WS  = tf([1.5, 0], [1, 1/10])
	mso = MaximumSensitivityObjective(WS, :y)
	mto = MaximumTransferObjective(tf(2, [1, 0]), :y, :u)

	objectives = [
		mso,
		mto,
	]
end
using Random, Distributions
begin
	tunable_parameters = [
	    cm.controller.gainPI.k => (1.0, 100.0) # parameter => (lower_bound, upper_bound)
	    cm.controller.int.k    => (1/1000, 1/5)
		cm.noise_filter.T      => (1.0, 10.0)
	]

	N = 10
	op2 = Dict{Any, Any}(copy(op))
	Random.seed!(0)
	op2[cm.tank.xc]  = MCM.Particles(N, Uniform(0.25, 0.65))
	op2[cm.tank.xT]  = MCM.Particles(N, Uniform(300, 320))
	operating_points = [ # Can be one or several operating points
	    op2
	]
end

atprob = StructuredAutoTuningProblem(model, wvec, tvec, objectives, operating_points, tunable_parameters)
# linearize(model, :u, :y; op)

plot(atprob, titlefontsize=10)

begin
	p0 = [10.0, 1/500, 10] # Initial guess
	res = solve(atprob, p0,
	    DyadControlSystems.IpoptSolver(verbose=isinteractive(), exact_hessian=false);
        # outer_iters=2,
	)
end

plot(res, size=(810,500))

obj_vals = [last(s) for ob in res.objective_status for s in ob]
@test all(obj_vals .< 1)


## SimulationObjective
costfun = sol->sum(abs2, sol[cm.tank.xc] .- rc)
prob
solve_args = (Rodas5P(), )
solve_kwargs = (; saveat = 10)
simobj = DyadControlSystems.SimulationObjective(costfun, prob, solve_args, solve_kwargs)

objectives2 = [objectives; simobj]
atprob2 = StructuredAutoTuningProblem(model, wvec, tvec, objectives2, operating_points, tunable_parameters)

plot(atprob2, titlefontsize=10, idxs=cm.tank.xc)


p0 = [10.0, 1/500, 10] # Initial guess
res2 = solve(atprob2, p0,
    DyadControlSystems.IpoptSolver(verbose=isinteractive(), exact_hessian=false);
    # outer_iters=2,
)

plot(
    plot(res),
    plot(res2, size=(810,500), idxs=cm.tank.xc),
)

obj_vals = [last(s) for ob in res2.objective_status for s in ob]
@test all(obj_vals .< 2)
