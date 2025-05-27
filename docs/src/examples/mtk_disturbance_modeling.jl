using ModelingToolkit, OrdinaryDiffEq, Plots, LinearAlgebra
using ModelingToolkitStandardLibrary
using ModelingToolkitStandardLibrary.Mechanical.Rotational
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: connect
using DyadControlSystems
using ControlSystemsMTK
t = ModelingToolkitStandardLibrary.Blocks.t
indexof(sym,syms) = findfirst(isequal(sym),syms)
function wr(sys)
    ODESystem(Equation[], ModelingToolkit.get_iv(sys), systems=[sys], name=:a_wrapper)
end

## Build the system model ======================================================
# Parameters
m1 = 1
m2 = 1
k = 10 # Spring stiffness
c = 3  # Damping coefficient

@named inertia1 = Inertia(; J = m1)
@named inertia2 = Inertia(; J = m2)
@named spring = Spring(; c = k)
@named damper = Damper(; d = c)
@named torque = Torque()

function SystemModel(u=nothing; name=:model)
    eqs = [
        connect(torque.flange, inertia1.flange_a)
        connect(inertia1.flange_b, spring.flange_a, damper.flange_a)
        connect(inertia2.flange_a, spring.flange_b, damper.flange_b)
    ]
    if u !== nothing
        push!(eqs, connect(torque.tau, u.output))
        return @named model = ODESystem(eqs, t; systems = [torque, inertia1, inertia2, spring, damper, u])
    end
    ODESystem(eqs, t; systems = [torque, inertia1, inertia2, spring, damper], name)
end


model = SystemModel() # Model with load disturbance
cmodel = complete(model)

## Design an LQR controller
@named d = Step(start_time=1, duration=10) # Disturbance
model_outputs = [model.inertia1.w, model.inertia2.w, model.inertia1.phi, model.inertia2.phi] # This is the state realization we want to control
inputs = [model.torque.tau.u]
linsys = named_ss(wr(model), inputs, model_outputs)
C = linsys.C

# Design state-feedback gain using LQR
# Define cost matrices
y_costs = [
    model.inertia1.w =>   1
    model.inertia2.w =>   1
    model.inertia1.phi => 1
    model.inertia2.phi => 1
]
Q = C'*diagm(ModelingToolkit.varmap_to_vars(y_costs, model_outputs))*C
@assert size(Q) == (4, 4)
L = lqr(linsys, 100Q, I(1))*C # Post-multiply by `C` to get the correct input to the controller
@named state_feedback = MatrixGain(-L) # Build negative feedback into the feedback matrix
@named add = Add() # To add the control signal and the disturbance

connections = [
    [state_feedback.input.u[i] ~ model_outputs[i] for i in 1:4]
    connect(add.input1, d.output)
    connect(add.input2, state_feedback.output)
    connect(add.output, :u, model.torque.tau)
]
closed_loop = ODESystem(connections, t, systems=[model, state_feedback, add, d], name=:closed_loop)


prob = ODEProblem(structural_simplify(closed_loop), Pair[], (0., 20.))
sol = solve(prob, Rodas5(), dtmax=0.1); # set dtmax to prevent the solver from overstepping the entire disturbance
plot(plot(sol, idxs=[model.inertia1.phi, model.inertia2.phi]), plot(sol, idxs=[state_feedback.output.u], title="Control signal"))

# Large stationary error


## Add disturbance model =======================================================
integrator = tf(1, [1, 1e-3]) # An (almost) integraing disturbance model. The 1e-3 is for numerical robustness in the Riccati-equation solver.
@named dist = ModelingToolkit.DisturbanceModel(model.torque.tau.u, integrator)
(f_oop, f_ip), augsys, dvs, p = ModelingToolkit.add_input_disturbance(wr(model), dist)

@unpack u, d = augsys
matrices, ssys = linearize(augsys, [u, d], model_outputs)
lsys = ss(matrices...)
lsys = named_ss(lsys, x=Symbol.(state_names(ssys)), u=Symbol.(input_names(ssys)), y=Symbol.(model_outputs))



## Design output-feedback controller using LQG =================================


# Define cost and covariance matrices
x_costs = [
    augsys.model.inertia1.w =>   1
    augsys.model.inertia2.w =>   1
    augsys.model.inertia1.phi => 10
    augsys.model.inertia2.phi => 10
    augsys.dist.x[1] => 0 # We do not penalize the disturbance state, it is not controllable
]

x_variances = [
    augsys.model.inertia1.w =>   10
    augsys.model.inertia2.w =>   10
    augsys.model.inertia1.phi => 1
    augsys.model.inertia2.phi => 1
    augsys.dist.x[1] => 100         # This correcponds to the integral gain of the controller
]

Q1 = build_quadratic_cost_matrix(matrices, wr(ssys), x_costs)
Q2 = 1.0 * I(1)
R1 = build_quadratic_cost_matrix(matrices, wr(ssys), x_variances)
R2 = Diagonal([1,1,1,1]) # Order is the same as in model_outputs
lqg = LQGProblem(lsys[:, :u].sys, Q1, Q2, R1, R2)
G_cont = -observer_controller(lqg) # Build negative feedback into the controller
@named cont = Blocks.StateSpace(ssdata(G_cont)...)


@named add = Add() # To add the control signal and the disturbance
@named d = Step(start_time=1, duration=10) # Disturbance

model_outputs = [model.inertia1.w, model.inertia2.w, model.inertia1.phi, model.inertia2.phi]

connections = [
    [cont.input.u[i] ~ model_outputs[i] for i in 1:4]
    connect(cont.output, add.input2)
    connect(add.input1, d.output)
    connect(add.output, :u, model.torque.tau)
]
closed_loop = ODESystem(connections, t, systems=[model, cont, add, d], name=:closed_loop)

# dmodel = @nonamespace augsys.dist
prob = ODEProblem(structural_simplify(closed_loop), Pair[], (0., 20.))
sol = solve(prob, Rodas5(), dtmax=0.1); # set dtmax to prevent the solver from overstepping the entire disturbance

f1 = plot(sol, idxs=[model.inertia1.phi, model.inertia2.phi])
f2 = plot(sol, idxs=[cont.output.u, cont.x[1]], lab=["Control signal" "Estimated disturbance"])


S = ss(get_sensitivity(closed_loop, :u)[1]...) |> minreal
w = exp10.(LinRange(-1.5, 2, 200))
f3 = bodeplot(S, w, plotphase=false, legend=:bottomright, title="Sensitivity")
Ms, ws = hinfnorm2(S)
scatter!([ws], [Ms], label="Ms = $(Ms)", markershape=:circle, markersize=3, markercolor=:red)


plot(f1,f2,f3, layout= @layout([[a;b] c]), size=(800, 600))

##
# Compare controller state with actual state
plot(sol, idxs=[
    model.inertia1.phi;
    model.inertia2.phi;
    cont.x[2:end]...
])



## Construct Kalman filter for state estimation ================================
# The Kalman filter is used to estimate the state of the system from the output.

Ts = 0.01
timevec = 0:Ts:sol.t[end]
y = sol(timevec, idxs=model_outputs)[:]
yn = [y .+ 0.0002 .* [10,10,1,1] .* randn.() for y in y]
u = sol(timevec, idxs=[cont.output.u])[:]


kf = KalmanFilter(ssdata(c2d(lsys[:, 1], Ts))..., R1, R2)

using ModelingToolkit: renamespace
@nonamespace x = augsys.dist.x[1]
filtsol = DyadControlSystems.forward_trajectory(kf, u, yn)
fx = plot(timevec, filtsol, ploty=true, plotu=false, plotx=false, plotxt = false, layout=5)
plot!(timevec, getindex.(filtsol.xt, indexof(renamespace(:dist, x), unknowns(ssys))), sp=5, lab="Estimated disturbance")
plot!(sol, idxs=model_outputs, sp=1:4)
plot!(sol, idxs=d.output.u, sp=5)








## Automatic robustification ===================================================
G = system_mapping(lqg)
W1 = -G_cont
K, γ, info = glover_mcfarlane(G, 1.05; W1)
labels = ["LQG" "GMF"]
gangoffourplot(G, [-G_cont, K], w)
nyquistplot([-G_cont, K] .* Ref(G), w; ylims=(-2,2), xlims=(-5, 0.2), Ms_circles=[Ms], labels)




## Controller reduction ========================================================
Kgmfr, hs = controller_reduction(lqg.sys, -K, 5) # Expects positive-feedback controller
Kgmfr = -Kgmfr # Flip sign again
Kgmfr.D .*= 0.0 # a hack to get better rolloff after reduction

marginplot([-G_cont*G, K*G, Kgmfr*G], w, lab=["LQG" "" "GMF original" "" "GMF Reduced" ""], plotphase=true, size=(1800,900))
nyquistplot([-G_cont*G, K*G, Kgmfr*G], w, ylims=(-2,2), xlims=(-5, 0.2), Ms_circles=[Ms], lab=["LQG" "GMF original" "GMF Reduced"])





## Do the same thing with only linear tools
matrices2, ssys2 = linearize(wr(model), [model.torque.tau.u], model_outputs);
lsys2 = add_low_frequency_disturbance(ss(matrices2...), ϵ=1e-3)

# @test lsys1[:, 1] == lsys2

# C = lsys1.C
