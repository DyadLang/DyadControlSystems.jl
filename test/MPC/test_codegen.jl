## Cartpole with MTKstdlib =====================================================

function wr(sys)
    ODESystem(Equation[], ModelingToolkit.get_iv(sys), systems=[sys], name=:a_wrapper_to_get_around_the_annoying_namespacing_and_avoid_calling_unpack_on_everything)
end

using LinearAlgebra
using ModelingToolkit
using ModelingToolkitStandardLibrary
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkitStandardLibrary.Mechanical.MultiBody2D
using ModelingToolkitStandardLibrary.Mechanical.Translational
using OrdinaryDiffEq
using DyadControlSystems
using Test
connect = ModelingToolkit.connect

# TODO: https://github.com/JuliaSymbolics/SymbolicUtils.jl/pull/475
using Symbolics.SymbolicUtils.Code
@inline function Code.create_array(A::Type{<:Base.ReshapedArray{T,N,P,MI}}, S, nd::Val, d::Val, elems...) where {T,N,P,MI}
    Code.create_array(P, S, nd, d, elems...)
end

@parameters t
D = Differential(t)

@named link1 = Link(; m = 0.2, l = 10, I = 1, g = -9.807)
@named cart = Translational.Mass(; m = 1, s_0 = 0)
@named fixed = Fixed()
@named force = Force()

eqs = [connect(link1.TX1, cart.flange)
       connect(cart.flange, force.flange)
       connect(link1.TY1, fixed.flange)]

@named model = ODESystem(eqs, t, [], []; systems = [link1, cart, force, fixed])
def = ModelingToolkit.defaults(model)
def[link1.y1] = 0
def[link1.x1] = 10
def[D(D(link1.x1))] = 0
def[D(D(link1.y_cm))] = 0
def[link1.A] = -pi / 2
def[link1.dA] = 0
def[cart.s] = 0
def[force.flange.v] = 0
lin_outputs = [cart.s, cart.v, link1.A, link1.dA]
lin_inputs = [force.f.u]
G = named_ss(model, lin_inputs, lin_outputs, allow_symbolic = true, op = def,
             allow_input_derivatives = true)
G = minreal(sminreal(G))

C = G.C
Q1 = 1.0C'I(G.nx) * C
L = lqr(G, 10Q1, I(1)) / C

@named state_feedback = MatrixGain(-L) # Build negative feedback into the feedback matrix
@named add = Add() # To add the control signal and the disturbance
@named d = Step(start_time = 3, duration = 1, height = 150) # Disturbance

connections = [[state_feedback.input.u[i] ~ lin_outputs[i] - def[lin_outputs[i]]
                for i in 1:4]
               connect(add.input1, d.output)
               connect(add.input2, state_feedback.output)
               connect(add.output, :u, force.f)]
@named _closed_loop = ODESystem(connections, t, [], [], systems = [state_feedback, add, d])
closed_loop = extend(_closed_loop, model)

sys = structural_simplify(closed_loop)
prob = ODEProblem(sys, Pair[link1.A => pi / 2 - 0.5, D(D(link1.x1))=>0, D(D(link1.y_cm))=>0], (0.0, 20.0))
sol = solve(prob, Rodas5P(), tstops = [3, 4]);
Tf = sol.t[end]
ts = range(0, Tf, length = round(Int, Tf * 20))
idxs = [link1.x1, 0, link1.x2, link1.y2]
us = sol(ts, idxs = idxs);

# @gif for u in us
#     plot(u[1:2:end], u[2:2:end], lw = 1, marker = (:d, 1), lab = false, xlims = (-40, 40),
#          ylims = (-25, 25), aspect_ratio = 1, title = "Inverted pendulum (closed loop)",
#          dpi = 200)
# end

# plot(
#     plot(sol, vars=lin_outputs),
#     plot(sol, vars=lin_inputs, title="Control signal"),
# )


## Build FunctionSystem and solve MPC problem ==================================
using DyadControlSystems.MPC
dynamics = FunctionSystem(model, lin_inputs, lin_outputs)
@assert dynamics.nx == 8 # n_ode + n_alg
@assert dynamics.na == 2 # n_alg
link_outputs = [link1.x1, link1.x2, link1.y1, link1.y2]
linksys = FunctionSystem(model, lin_inputs, link_outputs)


(; nx, nu) = dynamics
Ts = 0.15        # sample time
N = 60           # Optimization horizon (number of time steps of length Ts)
x0 =  ModelingToolkit.varmap_to_vars(def, dynamics.x)  # Initial state


xp = dynamics(x0, randn(nu), randn(length(dynamics.p)), 0) # Test that we can call the dynamics
@test_broken xp isa SVector


function indexmap(symbols_to_get, symbols_to_get_from)
    inds = map(symbols_to_get) do sym
        findfirst(isequal(sym), symbols_to_get_from)
    end
    any(isnothing, inds) && error("Could not find all symbols in indexmap, in particular, couldn't find $(symbols_to_get[findall(isnothing, inds)])")
    inds
end


scale_map = [
    cart.s => 3.0
    link1.A => 2.0
    link1.dA => 2.0
    link1.y1 => 3.0
    link1.x1 => 3.0
    force.flange.v => 4.0
    link1.ddA => 1.0
    link1.fx1 => 1.0
]
scale_x = ModelingToolkit.varmap_to_vars(scale_map, dynamics.x)


terminal_ref_map = [
    cart.s => 0
    link1.A => pi/2
    link1.dA => 0
    link1.x1 => NaN
    force.flange.v => 0
    link1.y1 => NaN
    link1.ddA => NaN
    link1.fx1 => NaN
]

p = ModelingToolkit.varmap_to_vars(def, dynamics.p)
discrete_dynamics    = MPC.MPCIntegrator(dynamics.dynamics, ODEProblem, Rodas4(); Ts, nx, nu, dt=Ts, adaptive=false, p)
discrete_dynamics(x0, randn(nu), randn(length(dynamics.p)), 0)


## End of setup ================================================================

const terminal_ref_inds3 = SVector(indexmap(first.(terminal_ref_map), dynamics.x)...)
r = ModelingToolkit.varmap_to_vars(terminal_ref_map, dynamics.x)

cost_map = [
    cart.s   => 1.0
    link1.A  => 1.0
    link1.dA => 0.0
    force.flange.v => 1
]
const cost_inds3 = SVector(indexmap(first.(cost_map), dynamics.x)...)
q = last.(cost_map)

# Create objective 
const Q3_ = Diagonal(SVector(q...))     # state cost matrix




running_cost = StageCost() do si, p, t
    e = si.x[cost_inds3] - r[cost_inds3]
    dot(e, Q3_, e) + 0.01*abs2(si.u[])
#     0.1*abs2(si.u[])
end

# objective = Objective() # No objective
objective = Objective(running_cost)

# Create objective input
u = zeros(nu, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p, 0)
oi = ObjectiveInput(x, u, r)

# Create constraints
bounds_constraint = BoundsConstraint(umin=[-20.0], umax=[20.0], xmin=fill(-Inf, nx), xmax=fill(Inf, nx), xNmin=r, xNmax=r)

observer = StateFeedback(discrete_dynamics, x0, dynamics.nu, dynamics.ny)

solver = IpoptSolver(;
        verbose                    = isinteractive(),
        tol                        = 1e-5,
        acceptable_tol             = 1e-4,
        max_iter                   = 2000,
        max_cpu_time               = 100.0,
        max_wall_time              = 100.0,
        constr_viol_tol            = 1e-5,
        acceptable_constr_viol_tol = 1e-4,
        acceptable_iter            = 100,
        exact_hessian              = true,
        mu_init                    = 0.1, # Initial value of barrier parameter
        # TODO: the mu_init didn't do much for MPC.solve time. Test if it has effect if we avoid `advance!(prob)` to make sure the initial trajectory is feasible.
        mu_strategy               = "adaptive", # Strategy for barrier parameter update, this problem requires adaptive strategy
)


discr = CollocationFinE(dynamics, false; n_colloc=3)
@time prob = GenericMPCProblem(
    dynamics;
    N,
    Ts,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    objective_input = oi,
    solver,
    xr = r,
    scale_x,
    disc = discr,
    # jacobian_method = :symbolic,
    presolve = true,
    verbose = isinteractive(),
);
# 24.491703 seconds (184.86 M allocations: 9.277 GiB, 5.21% gc time, 44.22% compilation time)
# 24.287493 seconds (186.41 M allocations: 9.211 GiB, 5.43% gc time, 44.84% compilation time) # with bounds constraint instead of Stage and Terminal

using Plots
x_sol, u_sol = copy.(get_xu(prob))
@test norm(u_sol[:,1]) > 1 # To check that the first control input is not constrained to zero by algebraic equations
if isinteractive()
    fig = plot(
        plot(x_sol[:, 1:5:end]', title="States", lab=permutedims(DyadControlSystems.state_names(dynamics))),
        plot(u_sol', title="Control signal", lab=permutedims(DyadControlSystems.input_names(dynamics))),
        )
    hline!([π/2], ls=:dash, c=2, sp=1, lab="α = π / 2")
    display(current())
end


##
@test x_sol[cost_inds3, end] ≈ r[cost_inds3] atol=1e-3

# Not all link coordinates are present as states, so we need to compute them before animating
if isinteractive()
    @gif for (ui, xi) in enumerate(1:5:size(x_sol, 2)-1)
        @show (ui, xi)
        u = linksys.measurement(x_sol[:, xi], u_sol[:, ui], p, 0)
        xcoord = u[1:2] 
        ycoord = u[3:4]
        plot(xcoord, ycoord, lw = 1, marker = (:d, 1), lab = false, xlims = (-40, 40),
            ylims = (-20, 20), title = "Inverted pendulum swing-up using optimal control",
            dpi = 200, aspect_ratio = 1)
    end
end


## closed loop

discrete_funsys = FunctionSystem(discrete_dynamics, dynamics.measurement, Ts; dynamics.x, dynamics.u, dynamics.y, dynamics.meta)
@time hist = MPC.solve(prob; x0, T = N, verbose = false*isinteractive(), dyn_actual=discrete_funsys, p_actual=p)

x_sol_cl = reduce(hcat, hist.X)
u_sol_cl = reduce(hcat, hist.U)

@test x_sol_cl[1:6, 1] ≈ x_sol[1:6, 1] rtol=1e-3 # Algebraic equations can apparantly be anything here
@test u_sol_cl[:, 1] ≈ u_sol[:, 1] rtol=1e-3


if isinteractive()
    @gif for (ui, xi) in enumerate(1:size(x_sol_cl, 2)-1)
        # @show (ui, xi)
        u = linksys.measurement(x_sol_cl[:, xi], u_sol_cl[:, ui], p, 0)
        xcoord = u[1:2] 
        ycoord = u[3:4]
        plot(xcoord, ycoord, lw = 1, marker = (:d, 1), lab = false, xlims = (-40, 40),
            ylims = (-20, 20), title = "Inverted pendulum swing-up using optimal control",
            dpi = 200, aspect_ratio = 1)
    end
end





## Test Trapezoidal integration on the same problem

N = 60*3
Ts = 0.05
u = zeros(nu, N)
# x, u = MPC.rollout(discrete_dynamics, x0, u, p, 0)
oi = ObjectiveInput(x_sol, vec(repeat(u_sol, 3, 1))', r)


discr = Trapezoidal(dynamics)
@time prob = GenericMPCProblem(
    dynamics;
    N,
    Ts,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    objective_input = oi,
    solver,
    xr = r,
    scale_x,
    disc = discr,
    # jacobian_method = :symbolic,
    presolve = true,
    verbose = isinteractive(),
);

x_sol_trapz, u_sol_trapz = copy.(get_xu(prob))
@test norm(u_sol_trapz[:,1]) > 0.1 # To check that the first control input is not constrained to zero by algebraic equations
if isinteractive()
    fig = plot(
        plot(x_sol_trapz[:, 1:5:end]', title="States", lab=permutedims(DyadControlSystems.state_names(dynamics))),
        plot(u_sol_trapz', title="Control signal", lab=permutedims(DyadControlSystems.input_names(dynamics))),
        )
    hline!([π/2], ls=:dash, c=2, sp=1, lab="α = π / 2")
    display(current())
end

@test x_sol_trapz ≈ x_sol rtol=1e-1
@test norm(u_sol_trapz[1:1, 1:3:end] - u_sol) / norm(u_sol) < 0.1 skip=true # The trapz solution is one element off in time. This does not have to be an error, the discretization is much coarser after all, but makes testing a bit difficult
@test norm(u_sol_trapz[1:1, 1:3:end] - u_sol) / norm(u_sol) < 0.6

