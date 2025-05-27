# MPC for actuator allocation with redundant actuators

This example will demonstrate how we can use an MPC controller to choose which actuator to use in the presence of actuator redundancy, i.e., when we have more than one actuator that does the same thing. We will consider a very simple system of a double integrator in discrete time. The system is given by:
```math
\begin{aligned}
p^+ &= p + T_s v \\
v^+ &= v + T_s (u_1 + u_2)
\end{aligned}
```
where $x = [p, v]$ is the state and $u_1, u_2$ are the two redundant control inputs. In this case, both actuators have the same effect and control authority, but we assume that we want to use only one of them if possible, in which case we let the output of the other actuator be zero. One possible way to accomplish this is to make sure that the gradient of the cost w.r.t. actuating the second actuator, $u_2$, is larger than the gradient for the first actuator, $u_1$, over the entire feasible input set. This way, the cost will always be smaller if we choose to use $u_1$ rather than $u_2$, but if $u_1$ saturates, the controller may still choose to use $u_2$ to accomplish the task. Systems with redundant actuators are common in many practical applications, in some situations redundancy is included for fail safe operation, but often there is some form of a fixed start-up cost associated with using an actuator. A common example is found in power production, where the cost of starting a new generator may be large.

We start by **defining the dynamics**:
```@example REDUNDANT_CONTROL
using DyadControlSystems, Plots
using DyadControlSystems.MPC
using StaticArrays
using LinearAlgebra
gr(fmt=:png) # hide

function dynamics(x, u, p, t)
    Ts = 0.1
    A = SA[1 Ts; 0 1]
    B = SA[0 0; Ts Ts]
    return A * x + B * u
end

nu = 2              # number of controls
nx = 2              # number of states
Ts = 0.1            # sample time
N  = 10             # MPC optimization horizon
x0 = [10.0, 0.0]    # Initial state
r  = zeros(nx)

measurement = (x,u,p,t) -> x                # The entire state is available for measurement
discrete_dynamics = FunctionSystem(dynamics, measurement, Ts; x=[:p, :v], u=:u^nu, y=:y^nx)
nothing # hide
```
The next step is to define our **cost function**. As alluded to above, we are aiming for a cost associated with $u_2$ that has a larger gradient over the feasible control set compared to the gradient for $u_1$, one way to accomplish this is to penalize the absolute value of the control inputs, and use a large coefficient for $u_2$, for instance:
```@example REDUNDANT_CONTROL
plot( x->abs(x), -1, 1, lab="|u_1|")
plot!(x->2abs(x), -1, 1, lab="2|u_2|")
```
There are a few problems associated with this. One problem is that the absolute value is a *non-smooth* function, and optimizing this typically requires special-purpose solvers. One solution is to use a *smooth approximation* of the absolute value, for instance the *Huber* function:
```math
\begin{aligned}
\text{Huber}(x) = \begin{cases}
\frac{1}{2a}x^2 & \text{if } |x| \leq a \\
|x| - \frac{1}{2}a & \text{otherwise}
\end{cases}
\end{aligned}
```
where $a$ is a parameter that controls the smoothness of the function. We can plot the Huber function for different values of $a$:
```@example REDUNDANT_CONTROL
huber(x, a=0.1) = ifelse(abs(x) < a, x^2 / 2a, (abs(x) - a / 2))
plot(x->huber(x, 0.2), -1, 1, lab="a=0.2", title="Huber function")
plot!(x->huber(x, 0.05), -1, 1, lab="a=0.05")
```
By choosing a small $a$, we get a good approximation of the absolute value function, but the gradient is much smoother, and we can use a standard solver to optimize the cost function. (In practice we, we use a sum of the huber function and a quadratic penalty on ``u_2`` to make the solver extra happy and improve the speed of convergence.)

While the Huber function effectively mitigates the problem of using a non-smooth cost function, we may still want to use a quadratic penalty for the first actuator. We may do so without problems, as long as we satisfy the gradient condition mentioned above. For instance,
```math
0.1u_1^2 + \text{Huber}(u_2)
```
```@example REDUNDANT_CONTROL
plot(x->0.1abs2(x),      -1, 1, lab="\$ 0.1u_1^2 \$")
plot!(x->huber(x, 0.01), -1, 1, lab="\$ |u_2| \$")
```
The gradient of the squared penalty ``q_1u_1^2`` is ``2q_1u_1``, while the gradient of the Huber function ``q_2 \, \text{Huber}(u_2)`` is ``q_2 \, \text{sign}(u_2)`` outside of the smooth region. If we let the feasible region be given by ``|u_i| \leq 1``, we may choose our cost multipliers ``q_1, q_2`` such that ``2q_1 < q_2`` to satisfy the gradient condition. 

We pick values for ``q_1`` and ``q_2`` and place them in a named tuple `p` so that we may tune them easily, and choose a quadratic penalty for the state. 
```@example REDUNDANT_CONTROL
Q = Diagonal(@SVector ones(nx))    # state cost matrix
p = (; q1 = 0.1, q2 = 1.0)

running_cost = StageCost() do si, p, t
    q1, q2 = p
    e = (si.x)
    u = (si.u)
    dot(e, Q, e) + 
    q1*u[1]^2 + q1*u[2]^2 + 
    q2*huber(u[2], 0.001)
end

terminal_cost = TerminalCost() do ti, p, t
    e = ti.x
    dot(e, 10Q, e)
end

objective = Objective(running_cost, terminal_cost)
nothing # hide
```

We also define an instance of [`ObjectiveInput`](@ref) to pass an initial guess for the solution (in this case, a random initial guess is fine), and create a [`BoundsConstraint`](@ref) to indicate the feasible control set.
```@example REDUNDANT_CONTROL
# Create objective input
x = zeros(nx, N+1)
u = zeros(nu, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p, 0)
oi = ObjectiveInput(x, u, r)

# Create constraints
bounds_constraint = BoundsConstraint(umin = [-1, -1], umax = [1, 1], xmin = [-Inf, -Inf], xmax = [Inf, Inf])
observer = StateFeedback(discrete_dynamics, x0)

solver = MPC.IpoptSolver(;
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

prob = GenericMPCProblem(
    discrete_dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
    verbose = true,
);

# Run MPC controller
history = MPC.solve(prob; x0, T = 100)

plot(history, seriestype=:steppre)
```

After having solved the MPC problem, we see that during the first few seconds, the input ``u_1`` is saturated and the controller makes use of ``u_2`` as well in order to steer the system to the origin quickly. After a while, ``u_2`` is turned off, and later turned on again when it is time to break to come to a rest. At almost no point is ``u_2`` used unless ``u_1`` is saturated. 

Had we chosen ``q_1, q_2`` that do not meet the gradient requirement, we would have seen ``u_2`` being used more liberally:
```@example REDUNDANT_CONTROL
p = (; q1 = 0.1, q2 = 0.1) # Choose a smaller q2
history = MPC.solve(prob; x0, T = 100, p) # Pass in new parameters
plot(history, seriestype=:steppre)
```
this time, the controller used ``u_2`` in more or less the same way as ``u_1``.

## Closing remarks
This tutorial demonstrated a fairly simple case of actuator allocation to resolve redundancy among actuators. In this simple case we could just as well have optimized for the combined total input ``u_{tot} = u_1 + u_2`` and used a trivial post-allocator function that decided how to split ``u_{tot}`` between the actuators. However, in more complex cases the actuator allocation problem may be non-trivial, and we really would like to optimize the ``L_0`` pseudo-norm of the input vector, i.e., minimize the number of non-zero entries. This problem is NP-hard, and we instead often use the convex relaxation that amounts to minimizing the ``L_1`` norm, i.e., the sum of absolute values. This relaxation is known to produce the desired sparsity, often finding good solutions to the original ``L_0``-penalized problem.

The actuator-allocation problem can also be addressed with integer programming, see the tutorial [MPC with binary or integer variables (MINLP)](@ref) for an example of mixed-integer nonlinear programming (MINLP) MPC.