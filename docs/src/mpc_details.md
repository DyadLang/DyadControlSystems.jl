# MPC Details

## Observers
This package defines the type
- [`StateFeedback`](@ref) This observer does not incorporate measurement feedback. It can be used if you assume availability of full state information. 

In addition to [`StateFeedback`](@ref), you may use any observer defined in [LowLevelParticleFilters](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/), such as
- [`LowLevelParticleFilters.ParticleFilter`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/api/#LowLevelParticleFilters.ParticleFilter-Tuple{Integer,%20Function,%20Function,%20Any,%20Any,%20Any}): This filter is simple to use and assumes that both dynamics noise and measurement noise are additive.
- [`LowLevelParticleFilters.AuxiliaryParticleFilter`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/@ref): This filter is identical to ParticleFilter, but uses a slightly different proposal mechanism for new particles.
- [`LowLevelParticleFilters.AdvancedParticleFilter`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/api/#LowLevelParticleFilters.AdvancedParticleFilter-Tuple{Integer,%20Function,%20Function,%20Any,%20Any,%20Any}): This filter gives you more flexibility, at the expense of having to define a few more functions.
- [`LowLevelParticleFilters.KalmanFilter`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/api/#LowLevelParticleFilters.KalmanFilter). Is what you would expect. Has the same features as the particle filters, but is restricted to linear dynamics and gaussian noise.
- [`LowLevelParticleFilters.UnscentedKalmanFilter`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/api/#LowLevelParticleFilters.UnscentedKalmanFilter). Is also what you would expect. Has almost the same features as the Kalman filters, but handle nonlinear dynamics and measurement model, still requires an additive Gaussian noise model.
- [`LowLevelParticleFilters.ExtendedKalmanFilter`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/api/#LowLevelParticleFilters.ExtendedKalmanFilter). Runs a regular Kalman filter on linearized dynamics. Uses ForwardDiff.jl for linearization.

How these observers are set up and used are shown in the examples above as well as in the examples section of the documentation.

## Discretization
When the dynamics are specified in continuous time, a discretization scheme must be employed in order for the optimizer to obtain a finite dimensional problem. While the quadratic MPC problem types always make use of multiple shooting, The [`GenericMPCProblem`](@ref) supports multiple different discretization methods, detailed in this section.

On a high level, the MPC library supports three general approaches to transcribe infinite-dimensional optimal-control problems to finite-dimensional optimization problems
- Multiple shooting
- Direct collocation on finite elements
- Trapezoidal integration

The implementation of multiple shooting supports dynamics consisting of ODEs only, i.e., algebraic equations (DAEs) are not supported, while the collocation  and trapezoidal methods support DAEs and stiff dynamics. Generally, we have the following properties of the different transcription methods:
- Multiple shooting introduces optimization variables for the state at each sample instant, $n_x \times N$ in total.
- Direct collocation introduces optimization variables for the state at each collocation point, $n_x \times n_c \times N$ in total, where $n_c$ is the number of collocation points (selected upon creating the [`CollocationFinE`](@ref) structure).
- Trapezoidal integration is an implicit method that introduces optimization variables for the state at each sample instant, similar to multiple shooting, $n_x \times N$ in total.

The multiple-shooting transcription will thus introduce fewer variables than collocation, but is only applicable to non-stiff systems of ODEs. Direct collocation (and the simpler trapezoidal integration scheme) is an implicit method that handles stiff dynamics and algebraic equations.

### Multiple shooting

Systems of ordinary differential equations (ODEs) can be discretized using an explicit method, such as Runge-Kutta 4. For non-stiff systems, the fastest option in this case is to make use of the special-purpose function [`MPC.rk4`](@ref). To discretize continuous-time dynamics functions on the form `(x,u,p,t) -> ẋ` using the function [`MPC.rk4`](@ref), we simply wrap the dynamics function by calling `rk4` like so:
```julia
discrete_dynamics = MPC.rk4(continuous_dynamics, sampletime; supersample=1)
```
where the integer `supersample` determines the number of RK4 steps that is taken internally for each change of the control signal (1 is often sufficient and is the default). The returned function `discrete_dynamics` is on the form `(x,u,p,t) -> x⁺`. The discretized dynamics can further be wrapped in a [`FunctionSystem`](@ref) in order to add a measurement function and names of states, inputs and outputs. 

Dynamics with algebraic equations can be discretized using the [`SeeToDee.SimpleColloc`](@ref) method, which uses a direct collocation method on finite elements. Example:
```julia
discrete_dynamics = SeeToDee.SimpleColloc(continuous_dynamics, sampletime; n=5)
```
for a 5:th order Gauss Lobatto collocation method. The returned object `discrete_dynamics` is once again callable on the form `(x,u,p,t) -> x⁺`.

Dynamics that is difficult to integrate due to stiffness etc. may make use of [`MPCIntegrator`](@ref). This type can use any method from the DifferentialEquations.jl ecosystem to perform the integration. This comes at a slight cost in performance, where `MPCIntegrator` with an internal `RK4` integrator is about 2-3x slower than the `MPC.rk4` function. The main difference in performance coming from the choice of integrator arises during the linearization step when SQP iterations are used.

### Direct collocation on finite elements

As an alternate to `MPC.rk4`, collocation of the system dynamics on finite elements provides a method that combines the rapid convergence of the orthogonal collocation method with the convenience associated with finite difference methods of locating grid points or elements where the solution is important or has large gradients. Instead of integrating the continuous-time dynamics, collocation on finite elements utilizes Lagrange polynomials to approximate the solution of the system dynamics over a finite element of time. These elements are collected over the time horizon of the MPC formulation to yield an optimal solution. The integer degree `deg` of the collocating Legendre polynomial determines the accuracy of the state solution obtained, and is related to the number of collocation points as `deg = n_colloc-1` where `n_colloc` is a user choice. The number of collocation points used is thus a tradeoff between increased computational cost and higher-order convergence. The truncation error depends on the choice of collocation points `roots_c`. For a choice of Gauss-Legendre collocation roots, the truncation error is of the order $\mathcal{O}(h^{2k})$ where $k$ is the degree of the polynomial. For Gauss-Radau collocation, the truncation error is of the order $\mathcal{O}(h^{2k-1})$. 
Collocation on finite elements can also be used to solve continuous-time DAE problems. The discretization structure for collocation on finite elements can be constructed as

```julia
disc = CollocationFinE(dynamics, false; n_colloc = 5, roots_c = "Legendre")
```
where, among the arguments to [`CollocationFinE`](@ref), `false` disables the threaded evaluation of dynamics and `n_colloc` refers to the size of the collocation point vector for each finite element. The `roots_c` option is set to choose Gauss-Legendre collocation by default. This can be specified explicitly by setting `roots_c = "Legendre"`. For Radau collocation points, `roots_c = "Radau"`. This discretization structure can be passed in [`GenericMPCProblem`](@ref) by specifying keyword argument `disc`.

### Accuracy of integration vs. performance

When solving MPC problems, it is sometimes beneficial to favor a faster sample rate and a longer prediction horizon over highly accurate integration. The motivations for this are several
- The dynamics model is often inaccurate, and solving an inaccurate model to high accuracy can be a waste of effort.
- The performance is often dictated by the disturbances acting on the system, and having a higher sample rate may allow the controller to detect and reject disturbances faster.
- Feedback from measurements will over time correct for slight errors due to integration.
- Increasing sample rate leads to each subsequent optimization problem being more similar to the previous one, making warm-staring more efficient and a good solution being found in fewer iterations.



## Solving optimal-control problems
At the heart of the MPC controller is a numerical optimal-control problem that is solved repeatedly each sample instant. For [`LQMPCProblem`](@ref) and [`QMPCProblem`](@ref), a single instance of this problem can be solved by calling
```julia
controlleroutput = MPC.optimize!(prob, x0, p, t)            # Alternative 1
controlleroutput = MPC.optimize!(prob, controlleroutput, p, t) # Alternative 2
```
where `x0` is the initial state and `t` is the time at which the problem starts.
The return object is an instance of [`ControllerOutput`](@ref) which contains the optimal control signal `u` and the optimal state trajectory `x`.
The returned value `controlleroutput.u` may for linear problems need adjustment for offsets, the call
```julia
MPC.get_u!(prob, controlleroutput.x, controlleroutput.u)
```
transforms the result of `optimize!` to the appropriate space.

For [`GenericMPCProblem`](@ref), the interface to `MPC.optimize!` is 
```julia
controlleroutput = MPC.optimize!(prob, controllerinput, p, t)
```
where `controllerinput` and `controlleroutput` are of types [`ControllerInput`](@ref) and [`ControllerOutput`](@ref). The constructor to [`GenericMPCProblem`](@ref) also has an option `presolve` that solves the optimal-control problem directly, after which the state and control trajectories are available as
```julia
x, u = get_xu(prob)
```

For an **example** of solving optimal-control problem, see [Optimal-control example](@ref optimal_control_example).

## Stepping the MPC controller
The a single step of the MPC controller can be taken by calling `MPC.step!`
```julia
uopt, x, u0 = MPC.step!(prob, observerinput, p, t)
```
where `observerinput` is of type [`ObserverInput`](@ref) containing the previous control input `u`, the latest measurement `y` which is used to update the observer in `prob`, a new reference `r` as well as any known disturbances `w`. Internally, `step!` performs the following actions:
1. Measurement update of the observer, forms ``\hat x_{k | k}``.
2. Solve the optimization problem with the state of the observer as the initial condition.
3. Advance the state of the observer using its prediction model, forms ``\hat x_{k+1 | k}``.
4. Advance the problem caches, including the reference trajectory if `xr` is a full trajectory.

The return values of `step!` are
- `uopt`: the optimal trajectory (usually, only the first value is used in an MPC setting). This value is given in the correct space for interfacing with the true plant.
- `x`: The optimal state trajectory as seen by the optimizer, note that this trajectory will only correspond to the actual state trajectory for linear problems around the origin.
- `u0` The control signal used to update the observer in the prediction step. Similar to `x`, this value may contain offsets and is usually of less external use than `uopt` which is transformed to the correct units of the actual plant input.

## Interface to ModelingToolkit
Simulating MPC controllers with ModelingToolkit models is an upcoming feature. To use an MTK model as the prediction model in an MPC problem or to solve optimal-control problems for MTK models, see the tutorial [Solving optimal-control problems with MTK models](@ref optimal_control_mtk_example).

## Index
```@autodocs
Modules = [DyadControlSystems.MPC]
Private = false
```

```@docs
DyadControlSystems.MPC.rollout
DyadControlSystems.MPC.rms
DyadControlSystems.MPC.modelfit
``` 

## MPC signals
```
  ┌───────────────┬──────────────────────┐
  │               │                      │
  │    ┌─────┐    │     ┌─────┐          │
w─┴───►│     │    └────►│     ├─────►v   │
       │     │ u        │     │          │
r─────►│ MPC ├──┬──────►│  P  ├─────►z   │
       │     │  │       │     │          │
 ┌────►│     │  │ d────►│     ├──┬──►    │
 │     └─────┘  │       └─────┘  │y      │
 │              │                │       │
 │   ┌───────┐  │                │       │
 │   │       │◄─┘                │       │
 │   │       │                   │       │
 └───┤  OBS  │◄──────────────────┘       │
     │       │                           │
     │       │◄──────────────────────────┘
     └───────┘
```
All signals relevant in the design of an MPC controller are specified in the block-diagram above. The user is tasked with designing the MPC controller as well as the observer.

The following signals are shown in the block diagram
-  $w$ is a *known* disturbance, i.e., its value is known to the controller through a measurement or otherwise.
-  $r$ is a reference value for the controlled output $z$.
-  $\hat x$ is an estimate of the state of the plant $P$.
-  $u$ is the control signal.
-  $v$ is a set of *constrained* outputs. This set may include direct feedthrough of inputs from $u$.
-  $z$ is a set of controlled outputs, i.e., outputs that will be penalized in the cost function.
-  $y$ is the measured output, i.e., outputs that are available for feedback to the observer. $z$ and $y$ may overlap.
-  $d$ is an *unknown* disturbance, i.e., a disturbance of which there is no measurement or knowledge.

The controller assumes that there are references $r$ provided for *all* controlled outputs $z$.
If $z$ is not provided, the controller assumes that *all states* are to be considered controlled variables and expects $Q_1$ to be a square matrix of size $n_x$, otherwise $Q_1$ is a square matrix of size $n_z$. $z$ may be provided as either a list of indices into the state vector, or as a matrix that multiplies the state vector.