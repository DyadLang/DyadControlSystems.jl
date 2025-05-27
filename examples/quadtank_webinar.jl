### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ e7bd4f47-4d1a-40fb-a5d7-59087c7d51f2
begin
	using Pkg, Revise, Test
	Pkg.activate(joinpath(@__DIR__(), ".."))
end

# ╔═╡ e45bb2ef-1e0d-489e-b757-ddee1ac5f2eb
begin
	using DyadControlSystems
	using DyadControlSystems.MPC
	using StaticArrays
	using Plots, Plots.Measures, LinearAlgebra
	using PlutoUI
	gr()
	md"**Load packages**"
end

# ╔═╡ ef24570f-accc-46db-a576-d3f64bda07bd
md"""
## About the presenter
- My name is **Fredrik Bagge Carlson**, I live in Lund in southern Sweden.
- I got my MSc and PhD from Dept. Automatic Control at **Lund University**.
- My background is within **robotics, control and system identification**.
- I enjoy writing **software toolboxes** 😄
- I work with simulation and control at **Julia Computing**.
- `@baggepinnen` on Github and julia discourse
- Reach out on `#control` on slack

![Fredrik](https://portal.research.lu.se/files-asset/114575837/bagge.jpg?w=160&f=webp)
"""

# ╔═╡ 6b92d6dd-3236-4e6b-8016-4e251ef76282
md"""
# Control design for a quadruple-tank system with JuliaSim Control

In this example, we will implement several different controllers of increasing complexity for a nonlinear MIMO process, starting with a PID controller and culminating with a nonlinear MPC controller with state and input constraints.

The process we will consider is a quadruple tank, where two upper tanks feed into two lower tanks, depicted in the schematics below. The quad-tank process is a well-studied example in many multivariable control courses, this particular instance of the process is borrowed from the Lund University [introductory course on automatic control](https://control.lth.se/education/engineering-program/frtf05-automatic-control-basic-course-for-fipi/).

 $(PlutoUI.LocalResource("../docs/src/examples/quadtank.png"))

The process has a *cross coupling* between the tanks, governed by a parameters $\gamma_i$: The flows from the pumps are
divided according to the two parameters $γ_1 , γ_2 ∈ [0, 1]$. The flow to tank 1
is $γ_1 k_1u_1$ and the flow to tank 4 is $(1 - γ_1 )k_1u_1$. Tanks 2 and 3 behave symmetrically.

The dynamics are given by
```math
\begin{aligned}
\dot{h}_1 &= \dfrac{-a_1}{A_1   \sqrt{2g h_1}} + \dfrac{a_3}{A_1 \sqrt{2g h_3}} +     \dfrac{γ_1 k_1}{A_1   u_1} \\
\dot{h}_2 &= \dfrac{-a_2}{A_2   \sqrt{2g h_2}} + \dfrac{a_4}{A_2 \sqrt{2g h_4}} +     \dfrac{γ_2 k_2}{A_2   u_2} \\
\dot{h}_3 &= \dfrac{-a_3}{A_3 \sqrt{2g h_3}}                         + \dfrac{(1-γ_2) k_2}{A_3   u_2} \\
\dot{h}_4 &= \dfrac{-a_4}{A_4 \sqrt{2g h_4}}                          + \dfrac{(1-γ_1) k_1}{A_4   u_1}
\end{aligned}
```
where $h_i$ are the tank levels and $a_i, A_i$ are the cross-sectional areas of outlets and tanks respectively. For this system, if $0 \leq \gamma_1 + \gamma_2 < 1$ the system is *non-minimum phase*, i.e., it has a zero in the right half plane. 

In the examples below, we assume that we can only measure the levels of the two lower tanks, and need to use a state observer to estimate the levels of the upper tanks.

The interested reader can find more details on the quadruple-tank process from the manual provided [here (see "lab 2")](https://canvas.education.lu.se/courses/16044/pages/course-materials?module_item_id=486293), from where the example is taken.

We will consider several variants of the MPC controller:
1. Linear control using a **PID controller**.
2. **Linear MPC** with a linear observer and prediction model.
3. MPC with a linear prediction model but a **nonlinear observer**.
4. Linear MPC with **integral action** and output references.
5. **Nonlinear MPC** using a nonlinear observer and prediction model. 

In all cases we will consider *constraints* on the control authority of the pumps as well as the levels of the tanks.

We start by defining the dynamics.
"""

# ╔═╡ 668ac1b0-7181-47a6-8e6f-14921e5ca475
begin
    ## Nonlinear quadtank
    const kc = 0.5
    function quadtank(h, u, p=nothing, t=nothing)
        k1, k2, g = 1.6, 1.6, 9.81
        A1 = A3 = A2 = A4 = 4.9
        a1, a3, a2, a4 = 0.03, 0.03, 0.03, 0.03
        γ1, γ2 = 0.2, 0.2
    
        ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0
        
		xd = SA[
            -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
            -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
            -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
            -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
        ]
    end
    
    nu = 2 # number of control inputs
    nx = 4 # number of states
    ny = 2 # number of measured outputs
    Ts = 2 # sample time
end;

# ╔═╡ 74eef68c-e476-4464-85e3-31461d1138f6
md"**Discretize the dynamics:**"

# ╔═╡ 23e0714e-d2ca-4782-8c73-b65a8c05406b
begin
	discrete_dynamics0 = rk4(quadtank, Ts, supersample=2) # Discretize the nonlinear continuous-time dynamics
	state_names = :h^4
	measurement = (x,u,p,t) -> kc*x[1:2]
	discrete_dynamics = FunctionSystem(discrete_dynamics0, measurement, Ts, x=state_names, u=:u^2, y=:h^2)
end

# ╔═╡ 827b3891-a9fe-43cf-8568-7a2626fed018
md"""
Next, we define the **constraints and an operating point**. The maximum allowed control signal will be determied by an interactive slider that is placed further down in the notebook. We start by defining a *desired* state at the operating point, `xr0`
"""

# ╔═╡ c829d4da-b523-4d08-b290-923df51a36a3
xr0 = [10, 10, 6, 6]; # desired reference state

# ╔═╡ 7938d902-5813-4a3a-b080-423291067baa
xr, ur = begin # control input at the operating opint
	using Optim
	optres = @views Optim.optimize(xu->sum(abs, quadtank(xu[1:4],xu[5:6],0,0)) + 0.0001sum(abs2, xu[1:4]-xr0), [xr0;.25;.25], BFGS(), Optim.Options(iterations=100, x_tol=1e-7))
	@info optres
	optres.minimizer[1:4], optres.minimizer[5:6]
end

# ╔═╡ d90ad60e-b8b3-4b25-8699-ae6039568794
md"We then solve for the state and control input that is close to the desired state and yields a stationary condition (zero derivatives)"

# ╔═╡ f1368578-b689-44ad-876f-e7a558534169
md"""
## PID control
Our first attempt at controlling the level in the quad-tank system is going use a PID controller. We will tune the controller using the **automatic tuning** capabilities of DyadControlSystems. To make use of the autotuner, we need a **linearized model** of the plant, for which we make use of the function `linearize`.
"""

# ╔═╡ bee822ef-0f76-4a82-968e-da648bc99bf9
begin
	Ac, Bc = DyadControlSystems.linearize(quadtank, xr, ur, 0, 0)
	Cc, Dc = DyadControlSystems.linearize(measurement, xr, ur, 0, 0)
	Gc = ss(Ac,Bc,Cc,Dc)

	disc = (x) -> c2d(ss(x), Ts) # Discretize the linear model using zero-order-hold
    G = disc(Gc)
end;

# ╔═╡ d7b3e38c-e358-4c54-8bc0-03b5de6682b7
Gc

# ╔═╡ b9140c07-d58a-451e-9deb-c4a3413163e7
md"""
Since this is a MIMO system and PID controllers are typically SISO, we look for an **input-output pairing** that is approximately **decoupled**. To this end, we investigate the **relative-gain array** [(RGA)](https://en.wikipedia.org/wiki/Relative_Gain_Array). If we aim for a crossover bandwidth of around 0.01rad/s, we find the optimal input-output pairings for decoupled PID control using the relative gain array:
"""

# ╔═╡ fe30d245-f20d-4e1c-82c2-b4cecf0fdce4
relative_gain_array(Gc, 0.01) .|> abs

# ╔═╡ a80fba55-a834-4426-990e-ad5b26c0203b
md"""
Unfortunately, this matrix is rather far from diagonalizable using permutations, indicating that there is a somewhat **strong cross-coupling** in the system and standard PID control is likely to be difficult (the parameter ``\gamma`` directly influeces the cross-coupling). Investigating the RGA as a function of frequency, we further notice that for high frequencies, the conclusion about the input-output pairing changes!
"""

# ╔═╡ 2a198917-0084-48fb-99d0-e036d785f0e0
md"""
In our case, we will stick with the pairing we deceded upon for low frequencies, since the input and state constraints will limit how fast we can control the system. We proceed to tune a controller for the $u_1 \rightarrow y_2$ mapping and do this by defining an **`AutoTuningProblem`**. This approach lets us put constraints on the largest magnitude of the closed-loop sensitivity functions, ``M_S, M_T, M_{KS}``.
"""

# ╔═╡ aedf84d0-6c8e-45a8-9c0d-f7a14263eff4
begin
	w = exp10.(-3:0.1:log10(pi/Ts)) # A frequency-vector for plotting
	tuningprob = AutoTuningProblem(; P=Gc[2,1], Ts, w, Ms=1.1, Mt=1.1, Mks=1, Tf=1200, metric=:IEIAE)
end;

# ╔═╡ b91388d3-cf19-4485-9dde-d7e301ca80f2
rgaplot(Gc, w, legend=false, layout=4, plot_title="RGA plot", title="", ylabel="", grid=false, link=:both)

# ╔═╡ 0e4d44b3-8745-48cd-9a2c-766a15e1aaa8
md"We solve the problem by calling `solve`"

# ╔═╡ 692a2252-453a-4d92-b669-cbae173dfa06
tuningresult = solve(tuningprob);

# ╔═╡ 0a61360e-b803-4974-bfa8-c683bda6e3bd
md"""
The autotuning returns a result structure, the PID+filter parameters are available as `tuningresult.p` while a controller object is available as `tuningresult.K`. One can further call `OptimizedPID(tuningresult)` to obtain a ModelingToolkit system that represents the tuned controller including anti-windup etc.
"""

# ╔═╡ 5b19ebd0-79c2-48c4-b8b0-d07262b5406c
plot(tuningresult, titlefont=9); xlims!((-5,1), ylims=(-5,0.5), sp=4, legend=:bottomleft)

# ╔═╡ 434f70b1-64c3-4ce4-92b3-01203cbda7eb
md"""
The resulting controller respects the sensitivity constraints, visualized in both Bode plots and the Nyquist plot as dashed lines. 
"""

# ╔═╡ a548b5fe-b724-4ff4-ae1c-7d99a01003a2
md"""
We also construct a **static precompensator** `iG0 =` ``G(0)^{-1}`` that decouples the system at DC. This strategy can sometimes mitigate the problem with cross-coupling between the channels, but may be treacherous if the relative-gain array contains large entries (which is fortunately not the case in our problem).
"""

# ╔═╡ f6fb7dd2-5788-48cb-ad1e-099861b185cf
md"We form the final controller by applying the static precompensator $G_0^{-1}$ to the optimized PID controller:"

# ╔═╡ ee1666a9-26c4-4afc-90f7-8b1231183241
md"""
To increase the realism in the simulation of the final system, we add the **saturation nonlinearity** that corresponds to **actuator limitations**, as well as the offsets implied by the linearization point to get the correct units on the signals:
"""

# ╔═╡ 918717ad-a3a2-4e67-bdb6-c68313d84f84
md"The simulation of the closed-loop system controlled by the PID controller is shown below."

# ╔═╡ 4a7fcbe2-76da-4c81-8bfa-1f4e74110860
md"""
This demonstrates that it can be treacherous to rely on optimization without taking all aspects of the problem into account. Furthermore,  In order to successfully control the tank system and respect input and state constraints using a PID controller, we would have to accept a lower performance, e.g., by lowering the constraint ``M_{KS}`` in the autotuning problem.

Fortunately, the MPC framework is very capable of taking input and state consstraints into account, something we will explore soon.

### Robustness analysis

The quadtank being a MIMO system means that the classical gain and phase margins are somewhat hard to apply. A **robustness measure** that is more suitable to the MIMO setting that also tells you something about simultaneous perturbations to both gain and phase at the plant input is the [*diskmargin*](https://arxiv.org/abs/2003.04771) and disk-based gain and phase margins, which we may plot as a function of frequency:
"""

# ╔═╡ e6ec51f7-c6aa-4bc4-a754-f5caf6674e7c
md"""
## Linear MPC

With a MPC controller, we can take the constraints into account explicitly in the optimization problem solved at each sample time.

In some situations, we may want to resort to a linear MPC controller. A linear controller is often sufficient when the task is *regulation*, i.e., keeping a controlled variable at a fixed set point.

We proceed in a similar fashion to above, making use of the model linearized around the specified operating point ``(x_r,u_r,y_r)``. We also construct a Kalman filter for state estimation.

For a linear MPC controller to work well, we must provide the operating point around which we have linearized. We also construct a [`LinearMPCModel`](@ref) that keeps track of the model and it's operating point
"""

# ╔═╡ c1e7684c-73b8-4143-8168-dc849d8df407
md"""
We also specify some cost matrices for the MPC problem and the prediction horizon ``N``. We will later see how to choose these matrices in more principled ways
"""

# ╔═╡ 1ab8fb06-41b9-447d-bad1-932d9eea1200
begin
	N = 10 # Prediction horizon
	Q1 = 1.0 * I(nx)
	Q2 = 1.0 * I(nu)
	qs = 100
	qs2 = 100000
end;

# ╔═╡ 047f0514-84b2-4587-960b-4b1d66aa65a9
md"""
Let's simulate the linear MPC:
"""

# ╔═╡ b127caf1-61c4-40a1-95e8-f76473d14f7c
md"""
The controller performs reasonably well and respects the input constraints. We notice that the control-signal trajectory looks qualitatively different now compared to when the PID controller was used, in particular during the time when the state constraints are active for the upper tanks.

With a linear observer, we notice **a slight violation of the state constraints** for states ``h_3, h_4``, remember, we do not measure these states directly, rather we rely on the observer to estimate them. Due to the square root in the dynamics that govern the outflow of liquid from the tanks, the observer thinks that the outflow is greater than it actually is at levels well above the linearization point. 

In practice, it's unrealistic to assume that we know the static gain of the system perfectly, in fact, the static gain for this system probably varies with the temperature of the equipment, the tank contents and during the lifetime of the tanks and pumps. We would thus likely end up with a stationary error using the controller above. This highlights a problem with naive MPC control (and similarly for standard LQG control), we do not have any integral action in the controller! We will soon see how we can add integral action, but first we explore how we can **make use of a nonlinear observer** (EKF) together with a linear prediction model in order to **improve the constraint satisfaction**.
"""

# ╔═╡ 4d48315f-9035-4a07-8874-b4f2b99f1f45
md"""
## Linear MPC with nonlinear observer

The nonlinear observer will make use of an extended Kalman filter (EKF). When we use a nonlinear observer together with a linear prediction model, we must adjust the inputs and outputs of the observer to account for the fact that the prediction model operates in ``\Delta``-coordinates. We o this using an `OperatingPointWrapper`. 
"""


# ╔═╡ 8bb9bbc7-2cc0-44c6-a653-b8fd0ab791f6
md"""
To highlight the problem of **lacking integral action**, we throw an **input disturbance** into the mix. The disturbance simulates a leakage in the pump after 500s.
"""

# ╔═╡ b5645119-d0df-42fa-b89e-b766c351ef28
function disturbance(u,t)
	t > 500/Ts ? [-0.02, 0] : [0.0, 0]
end

# ╔═╡ 13e56248-7c40-411b-91ff-0062457c0891
md"""
With the nonlinear observer, we notice that **the violation of the soft state constraints is eliminated**. This is an indication that the estimation of the states is more accurate compared to when we used a standard Kalman filter.
"""

# ╔═╡ ec6d2b75-da5a-4c8c-878d-9c9d649e0c4a
md"""
## Integral action and robust tuning

Finally, we explore how to add integral action to the controller. We will make use of a loop-shaping strategy, where we "shape" the linearized plant $G$ with a PI controller at the inputs.
"""

# ╔═╡ 60f76e82-5be3-497d-a697-0a6cdf8ca247
md"""
When performing loop shaping, it's helpful to inspect the *singular value plot* of the loop-transfer function. Below, we show the singular values for the open-loop plant, the shaped plant and the resulting robustified plant.
"""

# ╔═╡ 0d2d5864-0a88-4958-9a8b-87bea81e65cb
md"""Singular vector frequency: $(@bind freq Slider(-3:0.1:0, show_value=true))"""

# ╔═╡ 13bc40e7-e77a-4461-9c94-5f48fa2636d2
md"""
The parameters of $W_1$ were chosen so as to get a reasonable loop shape. The construction of the `RobustMPCModel` will give a warning if the resulting controller has poor robustness in the sense of the normalized co-prime margin (see [`ncfmargin`](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/api/#RobustAndOptimalControl.ncfmargin-Tuple{Any,%20Any}) and [`glover_mcfarlane`](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/api/#RobustAndOptimalControl.glover_mcfarlane) for more details, the $\gamma$ value is available as `model.gmf[2]`). 

Since this is a linear controller with a linear observer, we once again adjust the constraints to the operating point. We now create a [`RobustMPCModel`](@ref) where we provide the loop-shaping controller $W_1$. Internally, this makes use of the Glover-McFarlane method to find a robust controller and **inverse optimal control** to find the optimal cost matrices $Q_1, Q_2$. 
"""

# ╔═╡ 6c0e906f-ab6c-459d-be7d-961a6b820b79
md"""
The soft state-constraints are not fully respected; since the observer uses a linearized model, it fails to estimate the true value of the state and the actual value might thus be violating the constraints slightly. Thanks to the integrator in the controller, we now manage to reach the reference without steady-state error!
"""

# ╔═╡ 6b983f1d-bda3-4836-a38b-029fdb6322b1
md"""
### Robustness analysis
Also for the linear MPC regulator can we analyze **robustness properties** (assuming no constraints are active). We plot the multivariable diskmargins like we did for the PID controller above. The linear controller that is equivalent to the MPC controller, abscent constraints, is available in the `model` object
"""

# ╔═╡ 57911363-a262-44b6-b5d0-57d3e5cd41a9
md"""
The margins obtained above depend somewhat on the decoupler ``G(0)^{-1}`` that we chose, but are better than for the PID controller, even though the PID controller was optimized for robustness. We should keep in mind that the margins we consider here pertaines to *simultaneous* perturbations at both input channels, and we have thus in essence added twice the perturbation compared to what we would have done in a SISO setting. We may also ask the robust MPC model for its normalized-coprime-factor (NCF) margin:
"""

# ╔═╡ bc99fef9-448f-4e0e-991d-a152ebfe2c93
md"An NCF margin larger than about 0.25-0.3 indicates a successful and robust design."

# ╔═╡ 5e699127-9f18-4bf1-88ad-f49e19f5e743
md"""
## Nonlinear MPC
The last controller we'll consider is a nonlinear MPC controller that uses the nonlinear model of the plant for prediction, paired with a nonlinear observer. We will make use of the extended Kalman filter we defined above.
The nonlinear controller uses the nonlinear model as well as a nonlinear observer.
"""

# ╔═╡ 70447964-c611-41af-a08d-7b675159ed86
md"""
Let's test that the state converged to our reference and that the constraints were met.
"""

# ╔═╡ 9a670e4d-b707-4dde-a5ef-c0b434c1f58f
md"""
## Summary
We have seen how to using JuliaSim control to
- tune PID controllers automatically
- define and simulate MPC controllers 
- analyze robustness of feedback systems

This notebook will be available as a tutorial in JuliaSim (**help.juliahub.com**)

---
**Notebook settings**
"""

# ╔═╡ 10a17753-fd9b-404b-b8f2-920f4e34d097
html"""<style>
main {
    max-width: 65%;
}
"""

# ╔═╡ 3a259f6a-72ea-4881-9a55-515343ccf919
PlutoUI.TableOfContents()

# ╔═╡ ef25792f-4bb7-4558-b3d0-951f05baa6a5
begin
	gui_umax = @bind umax0 Slider(0.1:0.05:1.5, default=1, show_value=true)
	gui_umin = @bind umin0 Slider(-2:0.1:0, default=0, show_value=true)
	gui_decouple = @bind decouple CheckBox()
end;

# ╔═╡ de3d2f2f-acd0-40f5-94bc-b1ffaf6edbd8
begin
	op = OperatingPoint(xr, ur, Cc*xr)
	
    # Control limits
    umin = fill(umin0, nu)
    umax = fill(umax0, nu)
    
    # State limits
    xmin = zeros(nx)
    xmax = Float64[12, 12, 8, 8]
    constraints_pid = MPCConstraints(; umin, umax, xmin, xmax)
    
    x0 = [2, 1, 8, 3] # Initial tank levels
end;

# ╔═╡ a72c8ce8-ed9b-4a89-9566-ca24816c7ad5
md"""
The result looks alight, but the problems are multiple:
1. The input **saturation** causes the control signal to frequently hit the control limits and we thus effectively **loose feedback** during large transients.
2. The rather large **cross-coupling** between the channels is mostly **ignored** by the PID controller.
3. The non-measured states that were not allowed to go above $((constraints_pid.xmax[3:4]...,)) **overshot this constraint**, shown below.
"""

# ╔═╡ 4e88911e-bf21-4337-84f1-e9458c1a8b44
begin
    constraints_lmpc = MPCConstraints(; umin, umax, xmin, xmax)
	
	R1 = 1e-5*I(nx) # Dynamics covariance
	R2 = I(ny)      # Measurement covariance
	
	kf_lmpc = let (A,B,C,D) = ssdata(G)
        KalmanFilter(A, B, C, D, R1, R2)
	end
    pm = LinearMPCModel(G, kf_lmpc; constraints=constraints_lmpc, op, x0, strictly_proper=false)
end;

# ╔═╡ 306d293c-e95a-4dbd-a58d-cc8755715384
begin
	kf_ekf = let
		A,B,C,D = ssdata(G)
		KalmanFilter(A, B, C, D, R1, R2)
	end
	ekf = ExtendedKalmanFilter(
		kf_ekf,
		discrete_dynamics.dynamics,
		discrete_dynamics.measurement,
	)
end;

# ╔═╡ c02d9920-f739-4781-abe6-683f3142bb1a
xmax

# ╔═╡ 401749ce-7e60-4abd-a86f-0550f1abcbc2
begin
	solver = OSQPSolver(
        eps_rel = 1e-5,
        eps_abs = 1e-4,
        max_iter = 5000,
        check_termination = 5,
        verbose = false,
		sqp_iters=1,
        polish = true, # to get high accuracy
    )
	
    prob_lin = LQMPCProblem(pm; Q1, Q2=0Q2, Q3=Q2, qs, qs2, N, r=xr, solver)
    
    @time hist_lin = MPC.solve(prob_lin; x0, T = 1000÷Ts, verbose = false, noise=0, dyn_actual=discrete_dynamics)
    plot(hist_lin, plot_title="Linear MPC", legend=:bottomright )
end

# ╔═╡ ad162d91-6977-40ea-9d8a-4ab268c88a61
begin
	pm_ekf = LinearMPCModel(G, DyadControlSystems.OperatingPointWrapper(ekf, op); constraints=constraints_lmpc, op, x0, strictly_proper=false)
	prob_ekf = LQMPCProblem(pm_ekf; Q1, Q2=0Q2, Q3=Q2, qs, qs2, N, r=xr, solver)
    hist_ekf = MPC.solve(prob_ekf; x0, T = 1000÷Ts, dyn_actual=discrete_dynamics, disturbance)
    plot(hist_ekf, plot_title="Linear MPC with nonlinear observer", legend=:bottomright)
end

# ╔═╡ ab0b26d1-fa3e-47b8-bc40-33ce0a5a1592
constraints_rob = MPCConstraints(; umin, umax, xmin, xmax);

# ╔═╡ 22c65321-f788-4b37-b84e-5bf235d257fc
begin
    solver_nl = OSQPSolver(sqp_iters = 3)
    constraints = NonlinearMPCConstraints(; xmin, xmax, umin, umax)
    prob_nl = QMPCProblem(discrete_dynamics; observer=ekf, constraints, Q1, Q2, qs=0, qs2=0, N, xr, ur, solver=solver_nl)
    
    hist_nl = MPC.solve(prob_nl; x0, T = 1000÷Ts, verbose = false, noise=0.02)
    plot(hist_nl, plot_title="Nonlinear MPC")
end

# ╔═╡ ffe1cafa-aa6c-4f8f-8f8e-a97044427e9b
md"""
As we can see, the nonlinear MPC controller performs quite well and respects the state constraints that the two upper tanks ($h_3, h_4$) are not allowed to reach above a hight of $(constraints.max[3]).
"""

# ╔═╡ 66d4fa9b-30ba-4e8a-ad65-6a3e2c30e510
begin
    @test hist_nl.X[end] ≈ xr atol=0.2
    
    U = reduce(hcat, hist_nl.U)
    @test all(maximum(U, dims=2) .< umax .+ 1e-3)
    @test all(minimum(U, dims=2) .> umin .- 1e-3)
end

# ╔═╡ a88ce6a0-2ffc-489f-90f1-45e225888ff8
md"""
umax = $(gui_umax)
umin = $(gui_umin)
"""

# ╔═╡ 37810371-5d25-455b-9f39-89297c356560
begin
	if decouple
		iG0 = inv(dcgain(Gc)) # Decouple at DC
		iG0 ./= maximum(abs, iG0)
	else
		iG0 = [0 1; 1 0] # No decoupling
	end
end

# ╔═╡ 1b460efc-d092-464e-8b0c-23feb9ca1f0e
Cpid = tuningresult.K * I(2) * iG0;

# ╔═╡ 380ebfc7-6386-4846-b018-aa6f3b90498b
begin
	using ControlSystemsBase: offset, saturation
	Gcop = offset(op.y) * Gc * offset(-op.u) # Apply operating-point offset to controller
	Cpid_sat = saturation(constraints_pid.umin, constraints_pid.umax) * Cpid # Apply output saturation
end;

# ╔═╡ 57fa944e-550f-41d2-9392-87ea253b8745
begin
	fig1 = plot(lsim(feedback(Gcop*Cpid_sat), Gc.C*xr, 0:Ts:1500, x0=[x0-op.x; zeros(Cpid.nx)]), layout=1, sp=1, title="Outputs", ylabel="")
	hline!(Gc.C*xr, l=:dash, c=1, legend=false)
	plot(
		fig1,
		plot(lsim(feedback(Cpid_sat, Gcop), Gc.C*xr, 0:Ts:1500, x0=[zeros(Cpid.nx); x0-op.x]), layout=1, sp=1, title="Control signals", ylabel=""),
		size=(1000,300), margin=4mm
	)
end

# ╔═╡ e632e7e1-32ec-4cf2-a2e8-5094aec38d8a
begin
	res_pid = lsim(feedback(Gcop*Cpid_sat), Gc.C*xr, 0:Ts:1000, x0=[x0-op.x; zeros(Cpid.nx)])
	sfig = plot(res_pid.t, res_pid.x[1:4,:]' .+ op.x', label=string.(permutedims(:x^4)), title="States and constraints", layout=4)
	plot!(OvershootObjective.(constraints_pid.xmax), sp=(1:4)')
	sfig
end

# ╔═╡ 02949501-fc6e-4f4b-90ac-ea8a7456992f
plot(diskmargin(Gc, Cpid, 0, w).simultaneous_input, lower=false)

# ╔═╡ eea71a0e-8f6d-442e-a93a-5088ed310e0d
dm_pid = diskmargin(Cpid*Gc)

# ╔═╡ 031f5f89-15c8-4ec2-a42f-f8a56377690b
md"""
We see that we have a phase margin of about $(round(Int, dm_pid.phasemargin))° and a gain margin of about $(round(dm_pid.gainmargin[2], digits=2)), certainly not too impressive. Keep in mind, that these margins assumes a linear system without, e.g., the input saturation.

We may also visualize the stable region in the plane of simultaneous gain and pahse variations:
"""

# ╔═╡ 4d2e382c-28a2-4610-a7ae-87fb3357852f
plot(dm_pid)

# ╔═╡ 546dbbff-9f2b-43ac-9c95-3defca09613a
begin
    W1 = tf(0.001*[100, 1],[1,1e-6])|> disc # "Shape" the plant with a PI-controller (the 1e-6 is for numerical robustness)
    W1 = W1 * I(2) * iG0 # Make the PI-controller 2×2 since we have 2 inputs
end;

# ╔═╡ ceaf8a49-5702-45be-9bd9-7166fd8b2eeb
W1

# ╔═╡ c9ffde15-09b9-48c2-ba64-c04878001819
model = RobustMPCModel(G; W1, constraints=constraints_lmpc, x0, op, K=kf_lmpc);

# ╔═╡ a7f407f1-8049-4025-a23c-7b466f5482d7
sigmaplot([
	G,
	G*W1,
	G*model.gmf[1],
	], w, lab=["Plant" "Loop-shaped plant" "Robust loop gain"]); vline!([exp10(freq)], l=(:dash, :black), lab="Singular vector frequency")

# ╔═╡ ee279652-c089-4209-96da-0985810825c6
round.(svd(freqresp(G*model.gmf[1], exp10(freq))).V, digits=1) |> real

# ╔═╡ 945ae076-1ee5-4dd6-b856-9dceb103699a
begin

    prob_roby = LQMPCProblem(
        model;
        qs,
        qs2,
        N,
        r = op.y,
        solver,
    )
    
    hist_roby = MPC.solve(prob_roby; x0, T = 1500÷Ts, verbose = false, noise=0, dyn_actual=discrete_dynamics, Cz_actual = G.C, disturbance)
    
    plot(hist_roby, plot_title="Robust LMPC", legend=:bottomright)
	title!("Control signal", sp=2)
end

# ╔═╡ 2b26b8d3-0dce-4f0b-b8ef-26e8c1189596
begin
	equivalent_controller = model.gmf[1]
	f1 = plot(diskmargin(G, equivalent_controller, 0, w).simultaneous_input, lower=false, label="Simultaneous perturbations")
	plot!(diskmargin(G, equivalent_controller, 0, w).input[1], lower=false, label="Single channel perturbation", legend=false)
	ylims!((1, 12), sp=1, yscale=:identity)
	ylims!((-Inf, 100), sp=2, yscale=:identity)
	plot!(legend = :right, sp=1)
	plot(
		f1,
		plot(diskmargin(equivalent_controller*G, 0), titlefont=10),
		size = (900, 400), margin=5mm
	)
end

# ╔═╡ f7cc660b-2887-4ee7-8ad4-de0eecf66b2f
diskmargin(equivalent_controller*G)

# ╔═╡ cb11d98c-774a-4f0a-9766-98d870661d4c
model.info.margin

# ╔═╡ da5d5078-bef3-4b64-a545-effce532eeb6
md"""Decouple: $(gui_decouple)"""

# ╔═╡ 16cb5e31-add3-42a6-988b-7cc7417f5e54
md"""
umax = $(gui_umax)
umin = $(gui_umin)
"""

# ╔═╡ 2fb6abc8-3e47-4362-8431-5f034e7fec89
md"""Decouple: $(gui_decouple)"""

# ╔═╡ b0bb792c-e3ad-4425-b0ee-7986c9220be8
md"""
umax = $(gui_umax)
umin = $(gui_umin)
"""

# ╔═╡ efd2b142-8d25-4d3a-8bf2-0204bdeea965
md"""Decouple: $(gui_decouple)"""

# ╔═╡ 1f11e260-b94c-4392-b438-f19652b08cc1
md"""Decouple: $(gui_decouple)"""

# ╔═╡ 8fc0eaf6-848d-473e-b358-54a800548a25
md"""
umax = $(gui_umax)
umin = $(gui_umin)
"""

# ╔═╡ Cell order:
# ╟─ef24570f-accc-46db-a576-d3f64bda07bd
# ╟─6b92d6dd-3236-4e6b-8016-4e251ef76282
# ╠═668ac1b0-7181-47a6-8e6f-14921e5ca475
# ╟─74eef68c-e476-4464-85e3-31461d1138f6
# ╠═23e0714e-d2ca-4782-8c73-b65a8c05406b
# ╟─827b3891-a9fe-43cf-8568-7a2626fed018
# ╠═c829d4da-b523-4d08-b290-923df51a36a3
# ╟─d90ad60e-b8b3-4b25-8699-ae6039568794
# ╠═7938d902-5813-4a3a-b080-423291067baa
# ╠═de3d2f2f-acd0-40f5-94bc-b1ffaf6edbd8
# ╟─a88ce6a0-2ffc-489f-90f1-45e225888ff8
# ╟─f1368578-b689-44ad-876f-e7a558534169
# ╠═d7b3e38c-e358-4c54-8bc0-03b5de6682b7
# ╠═bee822ef-0f76-4a82-968e-da648bc99bf9
# ╟─b9140c07-d58a-451e-9deb-c4a3413163e7
# ╠═fe30d245-f20d-4e1c-82c2-b4cecf0fdce4
# ╟─a80fba55-a834-4426-990e-ad5b26c0203b
# ╠═b91388d3-cf19-4485-9dde-d7e301ca80f2
# ╟─2a198917-0084-48fb-99d0-e036d785f0e0
# ╠═aedf84d0-6c8e-45a8-9c0d-f7a14263eff4
# ╟─0e4d44b3-8745-48cd-9a2c-766a15e1aaa8
# ╠═692a2252-453a-4d92-b669-cbae173dfa06
# ╟─0a61360e-b803-4974-bfa8-c683bda6e3bd
# ╠═5b19ebd0-79c2-48c4-b8b0-d07262b5406c
# ╟─434f70b1-64c3-4ce4-92b3-01203cbda7eb
# ╟─a548b5fe-b724-4ff4-ae1c-7d99a01003a2
# ╠═37810371-5d25-455b-9f39-89297c356560
# ╟─da5d5078-bef3-4b64-a545-effce532eeb6
# ╟─f6fb7dd2-5788-48cb-ad1e-099861b185cf
# ╠═1b460efc-d092-464e-8b0c-23feb9ca1f0e
# ╟─ee1666a9-26c4-4afc-90f7-8b1231183241
# ╠═380ebfc7-6386-4846-b018-aa6f3b90498b
# ╟─918717ad-a3a2-4e67-bdb6-c68313d84f84
# ╟─57fa944e-550f-41d2-9392-87ea253b8745
# ╟─16cb5e31-add3-42a6-988b-7cc7417f5e54
# ╟─a72c8ce8-ed9b-4a89-9566-ca24816c7ad5
# ╟─e632e7e1-32ec-4cf2-a2e8-5094aec38d8a
# ╟─4a7fcbe2-76da-4c81-8bfa-1f4e74110860
# ╠═02949501-fc6e-4f4b-90ac-ea8a7456992f
# ╠═eea71a0e-8f6d-442e-a93a-5088ed310e0d
# ╟─031f5f89-15c8-4ec2-a42f-f8a56377690b
# ╟─2fb6abc8-3e47-4362-8431-5f034e7fec89
# ╠═4d2e382c-28a2-4610-a7ae-87fb3357852f
# ╟─e6ec51f7-c6aa-4bc4-a754-f5caf6674e7c
# ╠═4e88911e-bf21-4337-84f1-e9458c1a8b44
# ╟─c1e7684c-73b8-4143-8168-dc849d8df407
# ╠═1ab8fb06-41b9-447d-bad1-932d9eea1200
# ╟─047f0514-84b2-4587-960b-4b1d66aa65a9
# ╠═c02d9920-f739-4781-abe6-683f3142bb1a
# ╠═401749ce-7e60-4abd-a86f-0550f1abcbc2
# ╟─b127caf1-61c4-40a1-95e8-f76473d14f7c
# ╟─4d48315f-9035-4a07-8874-b4f2b99f1f45
# ╠═306d293c-e95a-4dbd-a58d-cc8755715384
# ╟─8bb9bbc7-2cc0-44c6-a653-b8fd0ab791f6
# ╠═b5645119-d0df-42fa-b89e-b766c351ef28
# ╟─ad162d91-6977-40ea-9d8a-4ab268c88a61
# ╟─13e56248-7c40-411b-91ff-0062457c0891
# ╟─b0bb792c-e3ad-4425-b0ee-7986c9220be8
# ╟─ec6d2b75-da5a-4c8c-878d-9c9d649e0c4a
# ╠═ceaf8a49-5702-45be-9bd9-7166fd8b2eeb
# ╠═546dbbff-9f2b-43ac-9c95-3defca09613a
# ╟─60f76e82-5be3-497d-a697-0a6cdf8ca247
# ╠═a7f407f1-8049-4025-a23c-7b466f5482d7
# ╟─0d2d5864-0a88-4958-9a8b-87bea81e65cb
# ╠═ee279652-c089-4209-96da-0985810825c6
# ╟─13bc40e7-e77a-4461-9c94-5f48fa2636d2
# ╟─ab0b26d1-fa3e-47b8-bc40-33ce0a5a1592
# ╠═c9ffde15-09b9-48c2-ba64-c04878001819
# ╟─efd2b142-8d25-4d3a-8bf2-0204bdeea965
# ╟─945ae076-1ee5-4dd6-b856-9dceb103699a
# ╟─6c0e906f-ab6c-459d-be7d-961a6b820b79
# ╟─1f11e260-b94c-4392-b438-f19652b08cc1
# ╟─6b983f1d-bda3-4836-a38b-029fdb6322b1
# ╟─2b26b8d3-0dce-4f0b-b8ef-26e8c1189596
# ╠═f7cc660b-2887-4ee7-8ad4-de0eecf66b2f
# ╟─57911363-a262-44b6-b5d0-57d3e5cd41a9
# ╠═cb11d98c-774a-4f0a-9766-98d870661d4c
# ╟─bc99fef9-448f-4e0e-991d-a152ebfe2c93
# ╟─5e699127-9f18-4bf1-88ad-f49e19f5e743
# ╟─8fc0eaf6-848d-473e-b358-54a800548a25
# ╠═22c65321-f788-4b37-b84e-5bf235d257fc
# ╟─70447964-c611-41af-a08d-7b675159ed86
# ╠═66d4fa9b-30ba-4e8a-ad65-6a3e2c30e510
# ╟─ffe1cafa-aa6c-4f8f-8f8e-a97044427e9b
# ╟─9a670e4d-b707-4dde-a5ef-c0b434c1f58f
# ╠═10a17753-fd9b-404b-b8f2-920f4e34d097
# ╠═3a259f6a-72ea-4881-9a55-515343ccf919
# ╠═ef25792f-4bb7-4558-b3d0-951f05baa6a5
# ╠═e7bd4f47-4d1a-40fb-a5d7-59087c7d51f2
# ╟─e45bb2ef-1e0d-489e-b757-ddee1ac5f2eb
