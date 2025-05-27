
# Extremum-seeking control
Extremum-seeking control (ESC) is an online optimization strategy whereby a controller seeks the extremum of a measureable cost function. Consider a system on the form
```math
\begin{aligned}
\dot x &= f(x) + g(x)u \\
y &= h(x)
\end{aligned}
```
where ``y = h(x)`` is the output of an objective function to be minimized. In a practical application, $h$ may be unknown, as long as $y$ is measurable. The extremum-seeking controller manipulates the input $u$ in order to find $x^*$ such that $h(x^*)$ is optimized. 

## Example: Online optimization of a cost function
In this example we will optimize the function $h$ in an online fashion. 
```@example esc
using Plots
gr(fmt=:png) # hide
h(u) = -3u - 2u^2 + u^3 + u^4
plot(h, -2, 2)
```
We do this by constructing a perturbing extremum-seeking controller [`ESC`](@ref) and indicate that the cost function input of the controller, $y$, is related to the controller output, $u$, as $y = h(u)$:
```@example esc
using DyadControlSystems, ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t
@named escont = ESC(k=0.8, a=0.1, w=3, wh=1)

connections = [
    escont.y ~ h(escont.u)
]

@named closed_loop = ODESystem(connections, t, systems=[escont])
sys = structural_simplify(closed_loop)

x0 = []
prob = ODEProblem(sys, x0, (0, 300.0))
sol = solve(prob, Rodas5())
plot(sol, idxs=[escont.u, escont.uh, escont.y])
```
As we can see above, the [`ESC`](@ref) explores the input space by means of a sinusoidal dither signal, and finds the global optimum of $h(1) = -3$. Had we started in $u=-1.5$ instead of $u=0$, we would have gotten stuck in the local minimum $h(-1)=1$ for much longer:
```@example esc
x0 = [
    escont.uh => -1.5
]
prob = ODEProblem(sys, x0, (0, 300.0))
sol = solve(prob, Rodas5())
plot(sol, idxs=[escont.u, escont.uh, escont.y])
```
Increasing the dither amplitude solves this problem:
```@example esc
x0 = [
    escont.uh => -1.5
    escont.a => 0.3
    escont.b => 0.3
]
prob = ODEProblem(sys, x0, (0, 300.0))
sol = solve(prob, Rodas5())
plot(sol, idxs=[escont.u, escont.uh, escont.y])
```
In general, increasing the dither amplitude increases the region of convergence and the convergence speed.


## Example: Dynamical extremum seeking
In this example, we will use an extremum-seeking controller to control a nonlinear dynamical system with a cost function output. The system to be controlled is given by
```math
\begin{aligned}
\dot{x}_1(t) &= x_1(t)^2 + x_2(t) + u(t) \\
\dot{x}_1(t) &= x_1(t)^2 - x_2(t)\\
y(t) &= -1 + x_1(t)^2 - x_1(t) 
\end{aligned}
```
and we will use the controller [`PIESC`](@ref). This controller offers potentially faster convergence compared to the [`ESC`](@ref) controller, at the expense of slightly harder tuning.
```@example esc
using DyadControlSystems: t, D

@named esc = PIESC(k=10, tau=0.1, a=10, w=100, wh=1000)

function Systemmodel(; name)
    @variables x1(t)=0 x2(t)=0 u(t) y(t)
    eqs = [
        D(x1) ~ x1^2 + x2 + u
        D(x2) ~ x1^2 - x2
        y ~ -1 + x1^2 - x1
    ]
    ODESystem(eqs, t; name)
end

@named model = Systemmodel()

connections = [
    model.y ~ esc.y
    esc.u ~ model.u
]

@named closed_loop = ODESystem(connections, t, systems=[model, esc])
sys = structural_simplify(closed_loop)

x0 = [
    # model.x1 => 0.5
    # model.x2 => 0.25
    # esc.uh => -0.5
    esc.v => 0
]

prob = ODEProblem(sys, x0, (0, 6.0))
sol = solve(prob, Rodas5())
plot(sol, idxs=[model.x1, model.x2, model.y, esc.u], layout=4)
```
The controller finds the optimum $h(x^*) = -1.25$ very quickly.

In this example, we let the ESC search for the control signal that optimizes the cost function directly. ESC is very general, and can also be used to find, e.g., the parameter of a parameterized controller that optimizes system performance. Furthermore, several different parameters can be optimized simultaneously by employing several ESCs operating at different dither frequencies. See
> Ariyur, Krstić. "Real Time Optimization by Extremum Seeking Control."
for more details.


## Example: Extremum seeking for model-reference adaptive control
In this example, we will use ESC to adaptively find a feedback gain that minimizes tracking error. We will implement the following control architecture
```
         ┌────────┐       ┌────────┐
         │        │       │  Ref   │  xr
         │  Ref   ├──┬───►│ Model  ├────┐
         │        │  │    │        │    │  ┌────┐    ┌──────┐     ┌────────┐
         └────────┘  │    └────────┘    └─►│+   │    │      │     │        │ y
                     │                     │ err├───►│ abs2 ├────►│  cost  ├──┐
           ┌───┐     │                  ┌─►│-   │    │      │     │        │  │
           │  +│◄────┘                  │  └────┘    └──────┘     └────────┘  │
┌──────────┤   │                        │                                     │
│          │  -│◄───────────────────────┤x                                    │
│          └───┘                        │                                     │
│                                       │                                     │
│  ┌────────┐      ┌───┐  ┌────────┐    │                                     │
│  │        ├─────►│   │  │        │    │                                     │
└─►│   C    │      │ x ├─►│  Model ├────┘                                     │
   │        │   ┌─►│   │  │        │                                          │
   └────────┘   │  └───┘  └────────┘            ┌─────┐                       │
                │          ┌─────┐       K      │     │                       │
                └──────────┤ sat │◄─────────────┤ ESC │◄──────────────────────┘
                           └─────┘              │     │
                                                └─────┘
```
where ``C`` denotes a standard P controller, and the multiplication block after ``C`` is used to implement the adaptive gain. The reference will be a square wave, this is fed into a reference model that implements the desired closed-loop behavior. The error between the output of the reference model ``x_r`` and the output of the system model ``x`` is squared and integrated and fed into the extremum-seeking controller. The [`ESC`](@ref) outputs the adaptive controller gain ``K`` that is used to multiply the output of the P-controller. 

The system model in this example will be
$\dfrac{1}{s + 0.5}$
while the reference model is given by 
$\dfrac{0.5}{s + 1}$
which means that perfect model matching will be obtained for negative feedback with $K = 0.5$ (this is what the ESC is supposed to find out).

```@example esc
using DyadControlSystems
using DyadControlSystems: t
using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkitStandardLibrary.Blocks: StateSpace, LimPI, Add, Square, Integrator, StaticNonLinearity, Product, Limiter
connect = ModelingToolkit.connect

@named ref      = Square(amplitude=1, frequency=0.02)
@named model    = StateSpace([-0.5;;], [1;;], [1;;])
@named refmodel = StateSpace([-1;;], [0.5;;], [1;;])
@named pcont    = LimPI(k=1, T=Inf, u_max=10, Ta=Inf)
@named err      = Add(k1=1, k2=-1)
@named errpi    = Add(k1=1, k2=-1)
@named escont   = ESC(k=1, a=1, b=0.02, w=1, wh=10)
@named cost     = Integrator(k=10)
@named square   = StaticNonLinearity(abs2)
@named gain     = Product()
@named sat      = Limiter(y_min=0.1, y_max = 50) # Used to saturate the adaptive gain between 0.1 and 50


connections = [
    connect(ref.output, refmodel.input)
    connect(refmodel.output, err.input1)
    connect(model.output, err.input2)
    connect(err.output, square.input)
    connect(square.output, cost.input)
    connect(cost.output, escont.input)
    connect(escont.output, sat.input)
    connect(sat.output, gain.input1)
    connect(pcont.ctr_output, gain.input2)
    connect(gain.output, model.input)
    connect(ref.output, errpi.input1)
    connect(model.output, errpi.input2)
    connect(errpi.output, pcont.err_input)
]


@named closed_loop = ODESystem(connections, t; systems=[ref,model,refmodel,pcont,err,errpi,escont,cost,square,gain,sat])

sys = structural_simplify(closed_loop)

x0 = [ # Start with gain 0.4
    escont.uh => 0.4
]

prob = ODEProblem(sys, x0, (0, 100.0))
sol = solve(prob, Rodas5())
plot(
    plot(sol, idxs=[refmodel.output.u, model.output.u, err.output.u]),
    plot(sol, idxs=[sat.y, escont.u]),
    legend = :bottomright,
)
```
The left plot shows the outputs of the reference model and the system model as well as the error between them. The right plot shows the output of the ESC, i.e., the estimated optimal gain ``K``. Notice how the estimate of ``K`` varies due to the continuously applied dither signal used for exploration/excitation.
## Index

```@index
Pages = ["extremum_seeking.md"]
```
```@autodocs
Modules = [DyadControlSystems]
Pages = ["extremum_seeking.jl"]
Private = false
```