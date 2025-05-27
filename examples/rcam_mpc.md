# Model-Predictive Control for the Research Civil Aircraft system
This tutorial will demonstrate how to setup *nonlinear MPC* using a [`GenericMPCProblem`](@ref) for the RCAM model, a system with 9 states and 5 control inputs. The system is described in detail in [The nonlinear Research Civil Aircraft Model (RCAM)](@ref rcam_example).


```@setup RCAM_MPC
using DyadControlSystems
using DyadControlSystems.MPC
using ModelingToolkit
using LinearAlgebra
using StaticArrays
using Plots
gr(fmt=:png) # hide

using DyadControlSystems.Symbolics.SymbolicUtils.Code

# Overload this method to make generated code re-tracable with symbols
function Code.function_to_expr(::typeof(SymbolicUtils.ifelse), O, st)
    args = arguments(O)
    :(Base.ifelse($(toexpr(args[1], st)), $(toexpr(args[2], st)), $(toexpr(args[3], st))))
end

# TODO: https://github.com/JuliaSymbolics/SymbolicUtils.jl/pull/475
@inline function Code.create_array(A::Type{<:Base.ReshapedArray{T,N,P,MI}}, S, nd::Val, d::Val, elems...) where {T,N,P,MI}
    Code.create_array(P, S, nd, d, elems...)
end

@inline function Code.create_array(A::Type{<:Base.ReshapedArray{T,N,P,MI}}, S, nd::Val{1}, d::Val{dims}, elems...) where {T,N,P,MI,dims}
    SArray{Tuple{dims...}, T}(elems...)
end



#Before we are happy with the controller, we should perform some form of robustness analysis of the closed-loop system. Analyzing robustness for a nonlinear closed-loop system is non-trivial in general, but a linear analysis can be performed to give an initial indication. Since our MPC controller is using a quadratic cost, it is reasonable to analyze the linearized closed loop with the corresponding LQR controller. To do this, we create an [`LQGProblem`](@ref)

#```@example RCAM_MPC
#R1 = I(G.nx) |> Matrix # Dynamics covariance
#R2 = 0.001I(G.ny) |> Matrix 
#lqg = LQGProblem(ExtendedStateSpace(G.sys, B1=I(9)), Matrix(Q1_scaled), Matrix(Q2_scaled), Matrix(R1), Matrix(R2))
#S = G_CS(lqg)
#@show Ms, wMs = hinfnorm2(S)
#w = exp10.(LinRange(-2, log10(pi/Ts), 200))
#sigmaplot(S, w, hz=true)
#scatter!([max(wMs, minimum(w))/2π], [Ms], lab=Ms, legend=:bottomright)
#Plqg = system_mapping(lqg)
#C = observer_controller(lqg)
#sim_diskmargin(lqg, 0, 1e-3, 1e3) # Simultaneous perturbation at input and output
#```

# jacobian_method = :symbolics, # Infinite compile time of the generated lagrangian hessian
```



A model of this system is available from the function `DyadControlSystems.ControlDemoSystems.rcam()`, the interested reader may learn how to build this model from scratch [here](@ref rcam_example). The named tuple returned from the `rcam` contains the `ModelingToolkit.ODESystem` describing the full system (`rcam.rcam_model`), as well as pre-simplified system `rcam.iosys`. We extract the symbolic variables representing the states and inputs from this simplified system using the functions `states` and `inputs` for later use. 
```@example RCAM_MPC
using DyadControlSystems
using DyadControlSystems.MPC
using ModelingToolkit

rcam = DyadControlSystems.ControlDemoSystems.rcam() # Fetch pre-specified model

x_sym = unknowns(rcam.iosys)
u_sym = ModelingToolkit.inputs(rcam.iosys)
```

## Linearize the nonlinear model
In order to perform simple linear robustness analysis of our controller, we may linearize the system around the trim point `(rcam.x0, rcam.u0)` specified in the `rcam` tuple, learn more about how to obtain this trim point in [Trimming](@ref). The function [`named_ss`](@ref) automatically wraps the linearized system in a [`NamedStateSpace`](@ref) object with the same signal names as the nonlinear system.
```@example RCAM_MPC
G = named_ss(rcam.rcam_model, u_sym, x_sym; op=merge(rcam.x0, rcam.u0))
```


## Create dynamics functions
The MPC framework wants the dynamics of the nonlinear system in the form of a [`FunctionSystem`](@ref). When calling the constructor of [`FunctionSystem`](@ref) with an `ODESystem` as input, code generation is automatically performed using [`build_controlled_dynamics`](@ref), the code below leads to a `FunctionSystem` with `u_sym` as inputs and `x_sym` as outputs being created, i.e., for the system below
```math
\begin{aligned}
\dot x &= f(x, u, p, t) \\
y      &= h(x, u, p, t)
\end{aligned}
```
we assume that ``y = x`` for this example. Learn more about how to estimate ``y`` from incomplete measurements in, e.g., [Control design for a quadruple-tank system with JuliaSim Control](@ref).

When constructing the `dynamics = FunctionSystem(rcam.rcam_model, u_sym, x_sym)` below, we get continuous-time dynamics representing ``\dot x = f(x, u, p, t)``, for simulation purposes, we need to construct a discrete-time version as well. To this purpose, we make use of [`MPC.rk4`](@ref), but other options are available, documented under [`Discretization`](@ref). We discretize with a sample time of ``T_s = 0.02s``, which is about 36 times faster than the fastest time constant of the linearized model.

```@example RCAM_MPC
dampreport(G) # Print a table of time constants for the linearized model

nu = G.nu # number of control inputs
nx = G.nx # number of states
Ts = 0.02 # sample time

dynamics = FunctionSystem(rcam.rcam_model, u_sym, x_sym)
discrete_dynamics = MPC.rk4(dynamics, Ts)   # Discretize the dynamics using RK4
nothing # hide
```

The trim point stored in the `rcam` tuple is in the form of dictionaries, mapping states to numerical values. The MPC framework is working with generated code and numerical arrays, and we thus map the values stored in the dicts to numerical arrays, the function `ModelingToolkit.varmap_to_vars`
helps us with this. 

```@example RCAM_MPC
x0 = SVector(ModelingToolkit.varmap_to_vars(rcam.x0, x_sym)...)
u0 = SVector(ModelingToolkit.varmap_to_vars(rcam.u0, u_sym)...)
p0 = SVector(ModelingToolkit.varmap_to_vars(rcam.rcam_constants, Symbol.(dynamics.p))...)
r  = copy(x0)
```

## Create an objective

We have now reached a point where it's time to starting thinking more about the MPC problem. We will make use of a quadratic cost function on the form
```math
x(N+1)^T Q_N x(N+1) + \sum_{t=1}^N x(t)^T Q_1 x(t) + u(t)^T Q_2 u(t) + \Delta u(t)^T Q_3 \Delta u(t)
```

To make it easier to choose the weights, we construct scaling matrices ``D_y, D_u`` that indicate the typical range of the inputs and outputs. 

```@example RCAM_MPC
Dy = Diagonal([10, 30, 6, 0.6, 0.3, 0.3, 1.2, 0.25, 1.2])
Du = Diagonal([0.1, 0.5, 0.03, 0.3, 0.3])

C2 = Dy \ I(nx) # The output matrix of `G` is identity

const Q1_scaled = C2'Diagonal(@SVector ones(nx))*C2           # state cost matrix
const Q2_scaled = 0.1*inv(Du)'*Diagonal(@SVector ones(nu))/Du # control cost matrix
const Q3_scaled = Q2_scaled
QNsparse, _ = MPC.calc_QN_AB(Q1_scaled, Q2_scaled, Q3_scaled, discrete_dynamics, Vector(r), Vector(u0), p0) # Compute terminal cost
const QN = Matrix(QNsparse)
nothing # hide
```

We now define the cost function components required to realize the cost function above, and package them all into an [`Objective`](@ref).

```@example RCAM_MPC
running_cost = StageCost() do si, p, t
    e = si.x - si.r
    u = si.u
    dot(e, Q1_scaled, e) + dot(u, Q2_scaled, u)
end

difference_cost = DifferenceCost((si,p,t)->SVector{5}(si.u)) do Δu, p, t
    dot(Δu, Q3_scaled, Δu)
end

terminal_cost = TerminalCost() do ti, p, t
    e = ti.x - ti.r
    dot(e, QN, e)
end

objective = Objective(running_cost, terminal_cost, difference_cost)
nothing # hide
```

## Create objective input
Next up, we define an instance of the type [`ObjectiveInput`](@ref). This structure allow the user to pass in an initial guess for the optimal solution from the starting state. To provide such a trajectory, we simulate the system forward in time from the initial condition `x0` using the function [`MPC.rollout`](@ref), here, we make use of the discretized dynamics. We also define the MPC horizon length `N = 10`. The horizon length is a tuning parameter for the MPC controller, and a good initial guess is to match the time constant of the dominant dynamics in the process to be controlled, i.e., if the time constant is ``\tau``, ``N = \tau / T_s`` is a reasonable start. For this system, we choose `N` much lower based on empirical tuning.
```@example RCAM_MPC
N  = 10 # MPC optimization horizon
x = zeros(nx, N+1)
u = zeros(nu, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p0, 0)
oi = ObjectiveInput(x, u, r)
nothing # hide
```

## Create constraints
Similar to how we created objective components above, we now create constraints we want our MPC controller to respect. We will only constrain the control inputs in this example, but we may in general constrain any arbitrary function of the inputs and states. While we're at it, we define typical bounds for the states as well so that we may set the limits of our plots later. 
```@example RCAM_MPC
umin = [
    deg2rad(-25)
    deg2rad(-25) 
    deg2rad(-30)
    deg2rad(0.5)
    deg2rad(0.5)
]

umax = [
    deg2rad(25)
    deg2rad(10)
    deg2rad(30)
    deg2rad(10)
    deg2rad(10)
]

xmin = [75, -30, -2, -1, -0.1, -0.2, -0.3, -0.1, -0.1]
xmax = [95, 20, 10, 1, 0.5, 0.5, 1, 0.5, 0.5]

bounds_constraints = StageConstraint(umin, umax) do si, p, t
    si.u # The constrained output is just u in this case
end
nothing # hide
```

## Create observer solver and problem
The MPC framework requires the specification of an observer that is responsible for feeding (an estimate of) the state of the system into the optimization-part of the MPC controller. In this example, we use the [`StateFeedback`](@ref), which is so named due to it not relying on measurements, rather, it knows the state of the system from the simulation model.

We also define the solver we want to use to solve the optimization problems. We will make use of IPOPT in this example, a good general-purpose solver. 
```@example RCAM_MPC
observer = StateFeedback(discrete_dynamics, Vector(x0))

solver = MPC.IpoptSolver(;
    verbose                     = false,
    tol                         = 1e-4,
    acceptable_tol              = 1e-2,
    max_iter                    = 100,
    constr_viol_tol             = 1e-4,
    acceptable_constr_viol_tol  = 1e-2,
    acceptable_iter             = 5,
    mu_init                     = 1e-12,
)
nothing # hide
```

Next up, we define the transcription scheme we want to use when transcribing the infinite-dimensional continuous-time problem to a finite-dimensional discrete-time problem. In this example, we will use [`Trapezoidal`](@ref) integration, a method suitable for fast, low-order solving. We provide `scale_x=diag(Dy)` when creating the problem so that the solver will scale the dynamics constraints using the typical magnitude of the state components. This may improve the numerical performance in situations where different state components have drastically different magnitudes.

```@example RCAM_MPC
# hide # disc = MPC.CollocationFinE(dynamics; n_colloc=3) # hide
disc = MPC.Trapezoidal(dynamics) 
# hide # disc = MultipleShooting(discrete_dynamics) # Takes infinite time in Calculating sparse Jacobian  # hide
nothing # hide
```
We are now ready to create the [`GenericMPCProblem`](@ref) structure.
```@example RCAM_MPC
prob = GenericMPCProblem(
    dynamics;
    disc,
    Ts,
    N,
    observer,
    objective,
    constraints = [bounds_constraints],
    p = p0,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
    symbolic_lag_h = false,
    scale_x = diag(Dy),
    scale_u = diag(Du),
    verbose = false, # Turn on to get diagnostics printing
);
nothing # hide
```

## Run MPC controller
With the problem in hand, we may simulate the MPC controller using the function `MPC.solve`. We choose a random initial condition `x0sim` to start the simulation from, and provide the discretized dynamics `dyn_actual=discrete_dynamics` as our simulation model (may be different from the prediction model in the MPC controller, which in this case is using the continuous-time dynamics).
```@example RCAM_MPC
x0sim = xmin .+ (xmax .- xmin) .* rand(nx) # A random initial state
history = MPC.solve(prob; x0=x0sim, T = 2000, verbose = false, dyn_actual=discrete_dynamics);
nothing # hide
```
The history object that is returned from `MPC.solve` contains some signals of interest for us to plot:
```@example RCAM_MPC
X,E,R,U,Y = reduce(hcat, history)

timevec = range(0, step=Ts, length=size(X,2))
figx = plot(timevec, X', label=permutedims(x_sym), xlabel="Time [s]", layout=9, ylims=(xmin', xmax'), tickfontsize=6, labelfontsize=6)
hline!(r', primary=false, l=(:dash, :black))
figu = plot(timevec[1:end-1], U', label=permutedims(u_sym), layout=5, tickfontsize=6, labelfontsize=6)
hline!([umin umax]', primary=false, l=(:dash, :black))
plot(figx, figu, size=(1000, 1000), tickfontsize=6, labelfontsize=6)
```
If everything went well, we expect to see an initial transient where the MPC controller is navigating the aircraft from the random initial condition to the vicinity of the set-point. This example did not add any integral action to the controller, so we do expect a minor stationary error in some of the signals. Learn more about adding integral action in one of the tutorials
- [Integral action](@ref)
- [Disturbance modeling and rejection with MPC controllers](@ref)
- [Mixed-sensitivity $\mathcal{H}_2$ design for MPC controllers](@ref)
- [Adaptive MPC](@ref)
- [Robust MPC tuning using the Glover McFarlane method](@ref)
- [Control design for a quadruple-tank system with JuliaSim Control](@ref)



