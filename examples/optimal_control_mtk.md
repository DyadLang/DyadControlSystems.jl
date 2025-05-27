# [Solving optimal-control problems with MTK models](@id optimal_control_mtk_example)

## Pendulum swing-up
In this example, we will solve an open-loop optimal-control problem (sometimes called trajectory optimization). The problem we will consider is to swing up a pendulum attached to a cart. This tutorial is very similar to [Solving optimal-control problems](@ref optimal_control_example), but here we make use of ModelingToolkit (MTK) and ModelingToolkitStandardLibrary to build the model of the system to control. MTK will for this model give us a DAE system, necessitating a different discretization scheme when we solve the optimal-control problem.

We start by defining the dynamics:

```@example OPTCONTROLMTK
using LinearAlgebra
using ModelingToolkit
using ModelingToolkitStandardLibrary
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkitStandardLibrary.Mechanical.MultiBody2D
using ModelingToolkitStandardLibrary.Mechanical.TranslationalPosition
using OrdinaryDiffEq
using DyadControlSystems
using StaticArrays
using Test
connect = ModelingToolkit.connect

@parameters t
D = Differential(t)

@named link1 = Link(; m = 0.2, l = 10, I = 1, g = -9.807)
@named cart  = TranslationalPosition.Mass(; m = 1, s = 0)
@named fixed = TranslationalPosition.Fixed()
@named force = TranslationalPosition.Force()

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
# def[force.flange.v] = 0
model
```

### Build FunctionSystem and solve MPC problem
Once we have our model defined, we wrap it in a [`FunctionSystem`](@ref) so that we can specify the number of inputs and outputs etc. for the MPC framework. We also create an initial state vector `x0` from which we will start our simulation (the downwards equilibrium of the pendulum). 
```@example OPTCONTROLMTK
using DyadControlSystems.MPC
control_input   = [force.f.u]
display_outputs = [cart.s, cart.v, link1.A, link1.dA]
priorities = Dict(link1.x1 => 10, link1.y1 => 10, cart.f => 10)
dynamics = FunctionSystem(model, control_input, display_outputs)
(; nx, nu) = dynamics
Ts = 0.15        # sample time
N  = 60          # Optimization horizon (number of time steps of length Ts)
x0 = ModelingToolkit.varmap_to_vars(def, dynamics.x)  # Initial state

function indexmap(symbols_to_get, symbols_to_get_from)
    inds = map(symbols_to_get) do sym
        findfirst(isequal(sym), symbols_to_get_from)
    end
    any(isnothing, inds) && error("Couldn't find $(symbols_to_get[findall(isnothing, inds)]) in indexmap")
    inds
end

terminal_ref_map = [ # This specifies the desired terminal point (the upwards equilibrium of the pendulum)
    cart.s   => 0
    cart.v => 0
    link1.A  => pi/2
    link1.dA => 0
    link1.x1 => NaN
    link1.y1 => NaN
    link1.ddA => NaN
    link1.fx1 => NaN
]

const r = SVector(ModelingToolkit.varmap_to_vars(terminal_ref_map, dynamics.x)...) # The reference point as a numerical vector

cost_map = [ # This specifies the cost associated with deviations from the reference point. A cost function is not required since we have a terminal constraint, but it allows us to favor certain types of solutions, e.g., lighter use of control input.
    cart.s   => 1.0
    link1.A  => 1.0
    link1.dA => 0.0
    force.flange.v => 1
]
const cost_inds = SVector(indexmap(first.(cost_map), dynamics.x)...)
q = last.(cost_map) # The costs as a numerical vector

# Create objective 
const Q = Diagonal(SVector(q...)) # state cost matrix

p = ModelingToolkit.varmap_to_vars(def, dynamics.p) # The parameters as a numerical vector

# Create a discretized version of the dynamics for simulation and plotting purposes. Since the system is a DAE, we discretize using the integrator `Rodas4` that supports mass matrices.
discrete_dynamics    = MPC.MPCIntegrator(dynamics.dynamics, ODEProblem, Rodas4(); Ts, nx, nu, dt=Ts, adaptive=false, p)

running_cost = StageCost() do si, p, t # Our cost function in the optimal-control problem
    e = si.x[cost_inds] - r[cost_inds]
    dot(e, Q, e) + 0.01*abs2(si.u[])
end

objective = Objective(running_cost)

# Create objective input
u = zeros(nu, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p, 0) # Simulate the system to obtain a dynamically feasible initial trajectory.
oi = ObjectiveInput(x, u, r)

# Create constraints
bounds_constraint = BoundsConstraint(
    umin  = [-20.0],
    umax  = [20.0],
    xmin  = fill(-Inf, nx),
    xmax  = fill(Inf, nx),
    xNmin = Vector(r), # The terminal constraint that forces the pendulum to end in the upright position.
    xNmax = Vector(r),
)

observer = StateFeedback(discrete_dynamics, x0, dynamics.nu, dynamics.ny) # StateFeedback is equivalent to perfect state knowledge

# Specify the solver
solver = IpoptSolver(;
        verbose                    = false,
        tol                        = 1e-4,
        acceptable_tol             = 1e-3,
        max_iter                   = 1000,
        max_cpu_time               = 100.0,
        max_wall_time              = 100.0,
        constr_viol_tol            = 1e-4,
        acceptable_constr_viol_tol = 1e-3,
        acceptable_iter            = 100,
        exact_hessian              = true,
        mu_strategy                = "adaptive", # Strategy for barrier parameter update, this problem improves with adaptive strategy
)

# The following function is a helper
import DyadControlSystems.Symbolics.SymbolicUtils.Code
@inline function Code.create_array(A::Type{<:Base.ReshapedArray{T,N,P,MI}}, S, nd::Val, d::Val, elems...) where {T,N,P,MI}
    Code.create_array(P, S, nd, d, elems...)
end
nothing # hide
```


Since the pendulum system we have created is a DAE system, as evidenced by
```@example OPTCONTROLMTK
equations(dynamics.meta.simplified_system)
```
we choose collocation on finite elements ([`CollocationFinE`](@ref)) as the discretization method.

For numerical performance, we define a scaling of the state variables. The scaling should be chosen to indicate the approximate range of each state for a typical trajectory. This will help the solver converge faster and also results in relative tolerances being used to check convergence for dynamics constraints, which is useful if different state components have very different magnitudes.

When we create the [`GenericMPCProblem`](@ref), we specify `presolve = true`, this will cause the optimal-control problem to be solved already in the constructor.
```@example OPTCONTROLMTK
scale_map = [
    cart.s    => 3.0
    link1.A   => 2.0
    link1.dA  => 2.0
    link1.y1  => 3.0
    link1.x1  => 3.0
    link1.ddA => 1.0
    link1.fx1 => 1.0
    force.flange.v => 4.0
]
scale_x = ModelingToolkit.varmap_to_vars(scale_map, dynamics.x)
disc = CollocationFinE(dynamics, false; n_colloc=3)
prob = GenericMPCProblem(
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
    disc,
    # jacobian_method = :symbolic,
    presolve = true,
    verbose = false,
)
nothing # hide
```

Since we specified `presolve = true`, the solution is now available using the function `get_xu`, we use this to plot the trajectories of the solution:
```@example OPTCONTROLMTK
using Plots
gr(fmt=:png) # hide
x_sol, u_sol = copy.(get_xu(prob))
@test x_sol[cost_inds, end] ≈ r[cost_inds] atol=1e-3 # Check convergence
fig = plot(
    plot(x_sol[:, 1:3:end]', title="States", lab=permutedims(DyadControlSystems.state_names(dynamics))),
    plot(u_sol', title="Control signal", lab=permutedims(DyadControlSystems.input_names(dynamics))),
    )
hline!([π/2], ls=:dash, c=2, sp=1, lab="α = π / 2")
```

We can also animate the swing-up. Not all link coordinates are present as states, so we need to compute them before animating:
```@example OPTCONTROLMTK
link_outputs = [link1.x1, link1.x2, link1.y1, link1.y2]
linksys = FunctionSystem(model, control_input, link_outputs)
@gif for (ui, xi) in enumerate(1:5:size(x_sol, 2)-1)
    u = linksys.measurement(x_sol[:, xi], u_sol[:, ui], p, 0)
    xcoord = u[1:2] 
    ycoord = u[3:4]
    plot(xcoord, ycoord, lw = 1, marker = (:d, 1), lab = false, xlims = (-40, 40),
        ylims = (-20, 20), title = "Inverted pendulum swing-up using optimal control",
        dpi = 200, aspect_ratio = 1)
end
```
