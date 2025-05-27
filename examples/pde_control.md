# PDE control
This tutorial solves Poisson-style equation and adds an LQR controller to control the solution in the center of the domain.

We finish off by simulating the closed-loop system with a sinusoidal disturbance.


## Outline
1. Define the model and find a **steady-state solution** without disturbances.
2. Discretize the model and **linearize** the discretized system to obtain a linear statespace model.
3. Design an **LQR controller** for the linear system.
4. **Simulate** the dynamic PDE with the LQR controller and a disturbance.

## Define PDE system

The PDE system we will control is a Possion-style system with added inputs, one controlled input and one disturbance input. Both of these inputs act through *"influence functions"* which are chosen to be Gaussian blobs with a very limited support.
```@example PDE_CONTROL
using ModelingToolkit, OrdinaryDiffEq, MethodOfLines, ControlSystems, Plots, ControlSystemsMTK, Interpolations, SteadyStateDiffEq
gr(fmt=:png) # hide

@parameters x y t
@variables u(..) a(t)=0
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt  = Differential(t)

Ainfluence(x,y) = exp(-((x-0.5)^2 + (y-0.5)^2)/0.1^2) # Actuator influence function
Dinfluence(x,y) = exp(-((x-0.3)^2 + (y-0.5)^2)/0.1^2) # Disturbance influence function

# Boundary conditions
bcs = [
	u(0,x,y) ~ 0.0,
	u(t,0,y) ~ 0.0, u(t,1,y) ~ 0.0,
	u(t,x,0) ~ 0.0, u(t,x,1) ~ 0.0
]

# Space and time domains
domains = [
	t ∈ IntervalDomain(0.0,1.0),
	x ∈ IntervalDomain(0.0,1.0),
	y ∈ IntervalDomain(0.0,1.0)
]

# Discretization
dx = 0.04
nd = round(Int, 1 / dx) + 1
discretization = MOLFiniteDifference([x=>dx,y=>dx],t)
nothing # hide
```

## Solve steady-state problem to find a linearization point

We discretize the PDE using [MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/), which uses a finite-difference approximation.

We create the matrix ``M`` that contains the steady-state solution, we will later use this as the reference in the control problem. The root solver only solves for the interior of the domain, so we have to manually expand ``M`` to contain zeros on the boundary.

```@example PDE_CONTROL
eqss  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y)) + sin(pi*x)*sin(pi*y)
@named pde_systemss = PDESystem([eqss],bcs,domains,[t,x,y],[u(t,x,y)])
ss_sys, tspan = SciMLBase.symbolic_discretize(pde_systemss, discretization)
ss_odeprob = discretize(pde_systemss,discretization)
probss = SteadyStateProblem(ss_odeprob)
ss_sol = solve(probss,SSRootfind());

M_ss = zeros(nd,nd)
M_ss[2:end-1, 2:end-1] .= reshape(ss_sol.u, nd-2, nd-2)
nothing # hide
```

## 2D PDE with input for linearization

We now add the control input variable ``a`` to the equation, we then discretize the problem and linearize the spatially discretized system. The operating point in which to linearize is taken to be the steady-state solution from above. The function `named_ss` performs both linearization using `ModelingToolkit.linearize` as well as creating a `NamedStateSpace` using the returned Jacobians. 

```@example PDE_CONTROL
# The actuation input `a` acts through the actuator influence function
eqi  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y)) + sin(pi*x)*sin(pi*y) + Ainfluence(x,y)*a

@named pde_systemi = PDESystem([eqi],bcs,domains,[t,x,y],[u(t,x,y)])
odesys, _ = SciMLBase.symbolic_discretize(pde_systemi, discretization)
s = unknowns(odesys)
inputs = [a]
outputs = s[(end÷2-3:end÷2+3) .- 13] # Extract a number of state variables in the middle of the domain that will serve as our "controlled outputs"

op = Dict([a=>0; unknowns(ss_sys) .=> vec(M_ss)])
lsys = named_ss(odesys, inputs, outputs; op)
w = exp10.(-2:0.01:2)
figb = bodeplot(lsys, w, plotphase=false, layout=1)
```

The linear system has a large number of state variables:
```@example PDE_CONTROL
lsys.nx
```

In a practical scenario where we deploy the controller using an observer, or if we implement an MPC controller etc. ee could perform model-order reduction to reduce the computational complexity:


```@example PDE_CONTROL
rsys, Gram, Tr = baltrunc(lsys) # Perform model reduction
```

This resulted in a model of order 3-4 only, but if you plot in in a Bode plot together with the full-order model you'll see that their frequency responses are identical.
## Design an LQR controller

We penalize the output only, i.e., the few grid cells in the center that were selected as output when we performed the linearization. This is realized by choosing the state-penalty matrix ``Q = C^T C``, since
```math
y^T Q_y y = (Cx)^T Q_y (Cx) = x^T C^T Q_y C x
```
(we chose ``Q_y = I``)

```@example PDE_CONTROL
C = lsys.C
Q = C'*C
R = 0.00000001
L = lqr(lsys, Q, R)
```

How does the LQR controller behave?

```@example PDE_CONTROL
inp(x, t) = -L*x
res = lsim(lsys, inp, 0:0.01:0.6, x0=ones(lsys.nx)) # Simulate closed loop
figs = plot(res.t, res.y', plot_title="lsim", layout=1, lab=permutedims(output_names(lsys)))
```

Since the state-feedback matrix corresponds to state-variables across a spatial domain, we ca reshape the matrix ``L``, which is currently ``L \in \mathbb{R}^{1 \times n_x n_y}`` into ``L_{mat} \in \mathbb{R}^{n_x \times n_y}`` and visualize it:

```@example PDE_CONTROL
Lmat = zeros(nd, nd)
Lmat[2:end-1, 2:end-1] .= reshape(L, nd-2, nd-2)
heatmap(Lmat, title="State feedback array \$L(x, y)\$")
```

Unsurprisingly, it looks like to controller only cares about the few state variables in the center which we selected to penalize when we chose the `outputs` for linearization.
## Closed-loop simulation with disturbance

The disturbance is sinusoidal, ``d = 2\sin(20t)`` and acts through the disturbance-influence function, which has support slightly below the center point of the domain where the actuator-influence function has support.


To simulate the closed-loop system, we extend the discretized LQR controller feedback matrix ``L`` to a continuous spatial domain ``L(x, y)`` using an interpolation. The full controller will thus be
```math
a(x, t) = L(x, t) \big(u(x, t) - M(x, t)\big)
```
where ``M`` is the reference solution obtained from the steady-state problem above, also made continuous through interpolation.

We simulate twice, once with the controller active (`controlled = 1`) and once with the controller turned off (`controlled = 0`).

```@example PDE_CONTROL
nodes = (0:dx:1, 0:dx:1) # For interpolation

# Create continuous state-feedback function using a 2D linear interpolator
Lfun = interpolate(nodes, Lmat, Gridded(Linear()))
Lint(x, y) = Lfun(x, y)
@register_symbolic Lint(x, y)

# Create a 2D interpoaltion also for the steady-state solution
Mfun = interpolate(nodes, M_ss, Gridded(Linear()))
Mint(x, y) = Mfun(x, y)
@register_symbolic Mint(x, y)

@parameters controlled=1
eqcl  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y)) + sin(pi*x)*sin(pi*y) - controlled*Ainfluence(x,y)*Lint(x, y)*(u(t,x,y) - Mint(x, y)) + Dinfluence(x,y)*2sin(20t)

@named pde_systemcl = PDESystem([eqcl],bcs,domains,[t,x,y],[u(t,x,y)], [controlled=>1])
probcl = discretize(pde_systemcl,discretization)
sol_control = solve(probcl,Tsit5())

no_controlprob = remake(probcl, p = [0]) # Turn off the controller
sol_nocontrol = solve(no_controlprob,Tsit5())
nothing # hide
```

## Visualization

We now have a look at the solution to the PDE, with and without the controller active. The controller acts in the center of the domain, while the disturbance acts slightly below the center. It's somewhat hard to see the difference in this kind of plot, but the controller reduces the variation in the center of the domain. Below, we show a time-series plot of the center to make it easier to spot the difference.

**With control**, notice how the center stays relatively constant:

```@example PDE_CONTROL
function showsol(sol)
	xg = 0:dx:1
	yg = 0:dx:1
	tvec = 0:0.012:1
	@gif for i in tvec
		M = sol(i, :, :)[]
		heatmap(xg,yg,M, clims = (0, 0.06), levels=100)
	end
end

showsol(sol_control)
```

**Without control**, notice how the center fluctuates due to the disturbance:
```@example PDE_CONTROL
showsol(sol_nocontrol)
```

Below, we show the solutiuon for a number of cells in the center of the domain:
```@example PDE_CONTROL
yinds = (-1:1) .+ nd ÷ 2
xind = nd ÷ 2
ref = M_ss[xind, yinds]
plot(sol_control.t, sol_control[u(t,x,y)][:, xind, yinds], c=1, lab="Controlled")
plot!(sol_nocontrol.t, sol_nocontrol[u(t,x,y)][:, xind, yinds], c=2, lab="Uncontrolled")
hline!(ref', l=(:black, :dash, 0.2), lab="References")
```


## MPC
Below, we show how to solve the same control problem as before, but this time using a linear MPC controller. To make it a bit more interesting, we include rather restrictive bounds on the control input (without these bounds, the MPC controller had been equivalent to the LQR controller).

We start by defining a symbolic equation that has a symbolic variable `input` to represent the control signal computed by the MPC controller.

```@example PDE_CONTROL
using DyadControlSystems: StateFeedback
using DyadControlSystems.MPC
using LinearAlgebra, Statistics

@variables input(t) [input=true]
eqmpc  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y)) + sin(pi*x)*sin(pi*y) + Ainfluence(x,y)*input + Dinfluence(x,y)*2sin(20t)

@named pde_systemmpc = PDESystem([eqmpc],bcs,domains,[t,x,y],[u(t,x,y)])
```
We then discretize this version of the PDE system and call `ModelingToolkit.generate_control_function` to generate a dynamics function that takes not only the state as argument, but also the control signal.


```@example PDE_CONTROL
"""
A helper function that discretizes the PDE system (in space) and generates a control function with signature f_oop = ẋ <- f(x, u, p, t) for the discretized system.
"""
function pde_control_function(pde_systemmpc, discretization, inputs)
    discsys, _ = SciMLBase.symbolic_discretize(pde_systemmpc, discretization)
    (f_oop, f_ip), dvs, p, io_sys = ModelingToolkit.generate_control_function(discsys, inputs)
end

(f_oop, f_ip), dvs, p, io_sys = pde_control_function(pde_systemmpc, discretization, [input]);
nothing # hide
```



We are now ready to setup the MPC controller, we will use the problem type [`LQMPCProblem`](@ref) which takes a [`LinearMPCModel`](@ref). To make the MPC controller performant, we will use the reduced-order model `rsys` that we computed above (select between full and reduced order using the `use_reduced` variable). We will also use the steady-state solution `M_ss` as the reference for the MPC controller. We use a prediction horizon of `N = 5` steps and control input bounds of ``u \in [-0.7, 0.7]``.



```@example PDE_CONTROL
Ts = 0.01 # Sample time of the MPC controller
use_reduced = true # Indicate whether or not to use the reduced-order linear model for MPC

if use_reduced
    # Relax, we already have the correct Tr matrix
else # Use full-order model
    Tr = I(lsys.nx) # The reduction matrix is the identity matrix
end

dsys = c2d(use_reduced ? rsys : lsys, Ts) # Discretize the linear system

(; nx,ny,nu) = dsys
N = 5 # MPC prediction horizon
x0 = zeros(lsys.nx) # Initial condition of full-order linear system
op = OperatingPoint(Tr*ss_sol.u, 0, dsys.C*Tr*ss_sol.u) # Set the operating point to the steady-state solution (possibly reduced to the statespace of the reduced-order model)

# Control limits
umin = -0.7 * ones(nu)
umax = 0.7 * ones(nu)
constraints = MPCConstraints(; umin, umax)

solver = OSQPSolver(
    verbose = false,
    eps_rel = 1e-4,
    max_iter = 100,
    check_termination = 5,
    polish = false,
)

discrete_dynamics = MPC.rk4(f_oop, Ts, supersample=20)
observer = StateFeedback(discrete_dynamics, Tr*x0, nu, ny) # We assume that we have access to the true state, see the MPC documentation for help on how to implement an observer
predmodel = LinearMPCModel(dsys, observer; constraints, op, x0 = Tr*x0)

prob = LQMPCProblem(predmodel; Q1=dsys.C'dsys.C, Q2=[R;;], N, solver, r=op.x)
```

We discretize the PDE system *in time* using the RK4 integrator, and implement a little helper function `mpc_sim` to perform the simulation for us. In the main simulation loop, we call the low-level function [`MPC.optimize!`](@ref), which requires us to handle the operating point manually, we thus adjust the state using the operating point `op` before calling `MPC.optimize!`. The matrix `Tr` is used to map the state from the full-order model to the reduced-order model.
```@example PDE_CONTROL
function mpc_sim(f_oop; T = 100, Ts, p, op)
    discrete_dynamics = MPC.rk4(f_oop, Ts, supersample=20)
    X = [x0]
    U = []
    Ymiddle = [] # Keep track of the same outputs as we plotted above
    D = zeros(nd, nd)
    for i = 1:T
        t = (i-1)*Ts
        x = X[end]
        Δx = Tr*x - op.x # Since we call the low-level function optimize! directly, we manually adjust for the operating point
        co = MPC.optimize!(prob, Δx, p, t; verbose=false)
        Δu = co.u[1]
        u = Δu + op.u
        xp = discrete_dynamics(x, u, p, t)
        D[2:end-1, 2:end-1] .= reshape(x, nd-2, nd-2)
        push!(X, xp)
        push!(U, u)
        push!(Ymiddle, D[xind, yinds])
    end
    range(0, step=Ts, length=T), X, U, Ymiddle
end

tmpc, X, U, Ymiddle = mpc_sim(f_oop; T = 100, Ts, p, op)

figsol = plot(sol_control.t, sol_control[u(t,x,y)][:, xind, yinds], c=1, lab="Controlled")
plot!(sol_nocontrol.t, sol_nocontrol[u(t,x,y)][:, xind, yinds], c=2, lab="Uncontrolled")
plot!(figsol, tmpc, reduce(hcat, Ymiddle)', lab="MPC", c=3)
hline!(ref', l=(:black, :dash, 0.2), lab="References")
figu = plot(tmpc, reduce(hcat, U)', lab="u")

plot(figsol, figu)
```

We see that the MPC controller behaves similar to the LQR controller, but respects the input constraints that we imposed. In this case, the reduced order model has
```@example PDE_CONTROL
rsys.nx
```
state variables, while the full-order model has
```@example PDE_CONTROL
lsys.nx
```
Solving the MPC problem with the full-order model is thus significantly more expensive than solving it with the reduced-order model, but the results are almost identical.

