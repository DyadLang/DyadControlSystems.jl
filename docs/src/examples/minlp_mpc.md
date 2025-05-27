# MPC with binary or integer variables (MINLP)

This example demonstrates how to solve MPC problems where some variables are constrained to be binary or integer valued (a binary variable is a special case of an integer variable). A common situation where such constraints arise are when an actuator can be either on or off, but nothing inbetween. Examples of situations in which integer-valued variables are useful
- Compressors without inverter control can only be on or off, but not in between.
- Thermostats often operate in an on-off fashion.
- Actuators controlled through relays or switches.
- Actuators with a significant cost associated with switching on or off, such as power plants.
- Discrete choices, such as selecting between different modes of operation.

Optimization with a mix of integer and real variables is often referred to as "mixed-integer nonlinear programming" (MINLP), so we are in this tutorial thus solving what is referred to as an MINLP MPC problem.

In this example, we will consider a simple double integrator, but constrain the input ``u(t)`` to be integer valued, taking only values $u(t) \in \left\{-1, 0, 1\right\}$. The system dynamics is given below

```math
\begin{aligned}
p^+ &= p + T_s v \\
v^+ &= v + T_s u
\end{aligned}
```

We start by defining the dynamics, cost and objective inputs. For a more detailed walkthrough of these steps, see [MPC for actuator allocation with redundant actuators](@ref)
```@example MINLP_MPC
using DyadControlSystems, DyadControlSystems.MPC, LinearAlgebra, StaticArrays, Plots
gr(fmt=:png) # hide
dynamics = function (x, u, p, t)
    Ts = 0.1
    A = SA[1 Ts; 0 1]
    B = SA[0; Ts;;]
    return A * x + B * u
end

nu = 1              # number of controls
nx = 2              # number of states
Ts = 0.1            # sample time
N  = 10             # MPC optimization horizon
x0 = [10.0, 0.0]    # Initial state
r  = zeros(nx)

measurement = (x,u,p,t) -> x                # The entire state is available for measurement
discrete_dynamics = FunctionSystem(dynamics, measurement, Ts; x=[:p, :v], u=:u, y=:y^nx)


Q1 = Diagonal(@SVector ones(nx))    # state cost matrix
p = (; Q1, q2 = 0.1)

running_cost = StageCost() do si, p, t
    Q1, q2 = p
    e = (si.x)
    u = (si.u)
    dot(e, Q1, e) +
    q2*u[1]^2
end

terminal_cost = TerminalCost() do ti, p, t
    e = ti.x
    dot(e, 10p.Q1, e)
end

objective = Objective(running_cost, terminal_cost)

x = zeros(nx, N+1)
u = zeros(nu, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p, 0)
oi = ObjectiveInput(x, u, r)
nothing # hide
```

We then **define the constraints**, here, we constrain the control input to be between ``-1 \leq u \leq 1``. We will constrain ``u`` to be integer valued when we create the MPC problem. 
```@example MINLP_MPC
# Create constraints
bounds_constraint = BoundsConstraint(umin = [-1], umax = [1], xmin = [-Inf, -Inf], xmax = [Inf, Inf])
observer = StateFeedback(discrete_dynamics, x0)
nothing # hide
```

When we **define the solver**, we need to specify
1. An NLP solver that is used to solve the inner optimization problem. In this case, we use Ipopt.
2. A mixed-integer nonlinear programming (MINLP) solver, in this case, we use Juniper. Juniper expects the inner solver to be defined as a function taking zero arguments, we thus create a closure around the Ipopt solver. 

```@example MINLP_MPC
using Juniper, OptimizationMOI
const MOI = OptimizationMOI.MOI

inner_solver = ()->MPC.IpoptSolver(;
    verbose = false,
    tol = 1e-4,
    acceptable_tol = 1e-3,
    max_iter = 200,
    max_cpu_time = 10.0,
    max_wall_time = 10.0,
    constr_viol_tol = 1e-4,
    acceptable_constr_viol_tol = 1e-3,
    acceptable_iter = 2,
)

solver = OptimizationMOI.MOI.OptimizerWithAttributes(
    Juniper.Optimizer,
    "nl_solver" => inner_solver,
    "allow_almost_solved"=>true,
    "atol"=>1e-3,
    "mip_gap" => 1e-3,
    "log_levels"=>[]
)
nothing # hide
```

To **indicate that the control input is integer valued**, we need to  provide the keyword argument `int_u`. This is a vector of booleans, where the `i`th entry indicates whether the `i`th control input is integer valued. In this case, we only have one control input, so we pass in `[true]`. If we would have had two control inputs, the first one being integer and the second real, we would have passed `int_u = [true false]`. There is also a corresponding argument `int_x` which we do not use in this example. 
```@example MINLP_MPC
prob = GenericMPCProblem(
    discrete_dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    objective_input = oi,
    solver,
    xr             = r,
    presolve       = true,
    verbose        = true,
    int_u          = [true], # Indicate that u is integer valued
);

# Run MPC controller
history = MPC.solve(prob; x0, T = 100)

plot(history, seriestype=:steppre)
```

When plotting the solution, we see that the control signal indeed only take integer values between ``-1`` and ``1`` (up to the numerical tolerance `atol`). For fun, we show also the corresponding solution when we allow the control input to take any real value. 
```@example MINLP_MPC
prob = GenericMPCProblem(
    discrete_dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    objective_input = oi,
    solver,
    xr             = r,
    presolve       = true,
    verbose        = true,
    int_u          = [false], # Allow u to take any real value
);

# Run MPC controller
history = MPC.solve(prob; x0, T = 100)

plot!(history, seriestype=:steppre, linestyle = :dash, plotr=false)
```
The state trajectories are almost identical (the system has strong low-pass character), but the control signal is much smoother when we allow the control input to take any real value.

## Speed up computation
For some problems, computations can be sped up significantly by relaxing the integer constraints applied to control inputs towards the end of the optimization horizon. This possibility applies if
1. The control input corresponds to a real value and not, e.g., a mode of operation.
2. The system dynamics is of low-pass character.

As an example, take the temperature control of a room, with a heating element that can be either on or off as input. The binary constraint on the control input is in this case a technical limitation of the actuator, but an actuator supplying 50% of the heating power could just as well have been used. The plant is also of lowpass character, temperature variations are smoothed out due to the thermal inertia of the building. If we are solving this MPC problem with an MPC horizon of ``N = 10``, we could maybe relax the binary constraint on the control input so that it's only enforced for the first ``N_{int} = 5`` time steps, and let the control input in the optimization be any real value for the last 5 time steps.

We do this be specifying the *integer horizon* using the keyword argument `Nint`, below, we show the solution to the double-integrator from above when ``N_{int} = 2``.

```@example MINLP_MPC
prob = GenericMPCProblem(
    discrete_dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    objective_input = oi,
    solver,
    xr             = r,
    presolve       = true,
    verbose        = true,
    int_u          = [true], # Indicate that u is integer valued
    Nint           = 2,      # Only force the first 2 control inputs to be integer
);

# Run MPC controller
history = MPC.solve(prob; x0, T = 100)

plot!(history, seriestype=:steppre, linestyle = :dot, plotr=false)
```
The solution in this case is almost identical to the one obtained when all ``N=10`` control inputs were constrained to be integer valued, but the solution was obtained over 10 times faster!

The intuition behind why this works is that the control decisions "far into the future" do not have a very large impact on what the first control input in the optimized trajectory should be, but the evolution of the state far into the future may still be important for constraint satisfaction, thus necessitating a long MPC horizon. With a lowpass system, replacing the integer values in the end of the control-signal trajectory with their "average values" is thus a good approximation.

## Closing remarks
We have seen how we can force some variables in an MPC problem to be integer valued, in this case, the control input was constrained to belong to the set ``u(t) \in \{-1, 0, 1\}``. Solving MINLP MPC problems can pose a significant computational challenge, but is nevertheless remarkably useful, in particular for high-level controllers and for processes with long time constants. 