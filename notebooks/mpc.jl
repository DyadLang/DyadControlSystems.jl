### A Pluto.jl notebook ###
# v0.17.7

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

# ╔═╡ e8d063e4-5745-11ec-364d-4792105f762a
begin
	import Pkg
	Pkg.activate() # This deactivates the Pluto package manager, and activates the global environment. See https://github.com/fonsp/Pluto.jl/wiki/%F0%9F%8E%81-Package-management#pattern-the-global-environment
	
	using Revise
	using DyadControlSystems, PlutoUI, Plots, StaticArrays
	using LinearAlgebra, Random
	using LowLevelParticleFilters
	using DyadControlSystems.MPC
	using ModelingToolkit
	Continuous = ControlSystemsBase.Continuous
end

# ╔═╡ 30d87043-e323-48fe-86a5-2b82e6e31b4c
md"""
# MPC tuner

This app assists in the tuning of an MPC controller using a quadratic cost function.

$u^* = \operatorname{arg min}_u x_N^T Q_N x_N + \sum_{t=1}^N x^T Q_1 x + u^T Q_2 u + \Delta u^T Q_3 \Delta u$

where $N$ is the optimization horizon and the final cost matrix $Q_N$ is automatically calculated.
"""

# ╔═╡ 0ec1bd8d-c5c7-4b69-ac6a-adda668ab403
md"Dynamics function in continuous time `(x,u,p,t) = (states, control inputs, params, time)`. We specify the dynamics function explicitly here. To obtain a control dynamics function from an ODESystem, use `DyadControlSystems.build_controlled_dynamics`."

# ╔═╡ 4eb5a13a-cd4e-4348-92e4-0632a9d079b3
md"### Model specification"

# ╔═╡ 8e8dadae-3d5b-4763-bc1c-173782a26a35
begin
	function dynamics_manual(x, u, p=nothing, t=nothing)
	    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81
	
	    q  = x[SA[1, 2]]
	    qd = x[SA[3, 4]]
	
	    s = sin(q[2])
	    c = cos(q[2])
	
	    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
	    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
	    G = @SVector [0, mp * g * l * s]
	    B = @SVector [1, 0]
	
	    qdd = -H \ (C * qd + G - B * u[1])
	    return [qd; qdd]
	end
	measurement_manual = (x,u,p,t) -> x[1:1]
	manual_sys = DyadControlSystems.FunctionSystem((dynamics_manual, nothing), (measurement_manual, nothing), Continuous(), [:x, :q, :v, :qd], 1:1, [:y], nothing, nothing, nothing)
end

# ╔═╡ c777e06d-0b60-4044-9750-b391176879dc
begin
	function cartpolesys(; name)
	    @variables t q(t)=0 qd(t)=0 x(t)=0 v(t)=0 u(t)=0 [input=true] y(t)=0 [output=true] yq(t)=0 [output=true]
	    D = Differential(t)
	    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81
	
	    s = sin(q)
	    c = cos(q)
	    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
	    C = [0 -mp*qd*l*s; 0 0]
	    G = [0, mp * g * l * s]
	    B = [1, 0]
	
	    rhs = -H \ (C * [v; qd] + G - B * u)
	
	    eqs = [
	        D(x) ~ v
	        D(q) ~ qd
	        D(v) ~ rhs[1]
	        D(qd) ~ rhs[2]
	        y ~ x
			yq ~ q
	    ]
	    ODESystem(eqs, t; name)
	end
	sys = cartpolesys(; name = :cartsys)
	#sys = manual_sys
	codesys = DyadControlSystems.build_controlled_dynamics(sys, sys.u, [sys.y, sys.yq])
end;

# ╔═╡ 57e01a59-9749-463f-9cf3-5caa6fce65f4
md"""
### Problem specification
The following things must be defined below:
- `dynamics, measurement` functions.
- The initial state `x0d`
- The reference state $x_r$
- The dynamics noise input matrix $B_w$, indicating the influence of disturbances upon the state dynamics.
- Upper and lower bounds for control inputs and states, `umin, umax, xmin, xmax`.
- State, control and control derivative cost matrices $Q_1, Q_2, Q_3$ and state and measurement noise covariance matrices $R_1, R_2$.
"""

# ╔═╡ 8c3855d3-0e9a-453d-ae4b-61a06433653b
begin
	dynamics = codesys
	(; nx,nu,ny) = codesys
	nw = 2 # number of disturbance inputs
	x0d = Dict(
		sys.x  => 3,
		sys.q  => pi*0.5,
		sys.v  => 0,
		sys.qd => 0
	)
	
	x0 = ModelingToolkit.varmap_to_vars(x0d, codesys.x)
	xr = zeros(nx) # reference state
	
	Bw = Matrix([0I(2); I(2)]) # Dynamics noise input matrix
	
	# Control limits
	umin = fill(-5, nu)
	umax = fill( 5, nu)
	# State limits (state constraints are soft by default)
	xmin = fill(-50, nx)
	xmax = fill( 50, nx)
	constraints = MPCConstraints(dynamics; umin, umax, xmin, xmax)
end;

# ╔═╡ fd17fd3f-3674-49e1-bfdd-eefae5c6315e
md"""
MPC optimization horizon: N $(@bind N Slider(2:100, default=50, show_value=true))

Control cost multiplier ($\log_{10}$): $(@bind logq2 Slider(-3:0.1:3, default=0, show_value=true))

Control derivative cost multiplier ($\log_{10}$): $(@bind logq3 Slider(-3:0.1:3, default=0, show_value=true))

Measurement covariance multiplier ($\log_{10}$): $(@bind logr2 Slider(-3:0.1:3, default=0, show_value=true))

Sample time: $(@bind Tss TextField((5,1); default="0.02"))s
"""

# ╔═╡ eaf5ca2b-1b17-4311-9f24-ba4ea220c8df
md"""
Observer: $(@bind obs Select([
	"openloop" => "Open loop",
	"kalman" => "Kalman filter",
	"extkalman" => "Extended Kalman filter",
]))
"""

# ╔═╡ 3e17f179-739d-46e4-9677-8fc2bc23ee54
md"""
Measurement noise $(@bind noise Slider(0:0.002:0.1, default=0, show_value=true))
"""

# ╔═╡ b78fbfb4-b71f-4a8f-aada-d686c790c792
obs == "openloop" ? md"(measurement noise does not affect open-loop observer)" : nothing

# ╔═╡ 7cea25cb-11f1-47e3-ae05-61c58ffa4a3e
md"""
Activated outputs $(@bind outs PlutoUI.MultiCheckBox((1:ny) .=> output_names(codesys), default=collect(1:ny)))
Activated inputs  $(@bind ins  PlutoUI.MultiCheckBox((1:nu) .=> input_names(codesys), default=collect(1:nu)))
"""

# ╔═╡ 6cefe746-8dae-4dcc-a7f9-880e7047db41
begin
	extrastring = length(outs) > 1 ? md"Since there are multiple outputs activated, singular-value plots for $S$ and $T$ are shown." : ""
	md"""
	Gang-of-four of closed-loop system assuming plant dynamics linearized around the reference point and a linear LQG controller with the Kalman-filter defined above. $extrastring
	"""
end

# ╔═╡ 4c89c492-0294-4209-99d3-4f8d7fc3b850
md"Step response: output complimentary sensitivity function, the transfer function from measurement noise / reference to plant output."

# ╔═╡ bb1b2b8d-bdbf-46f9-8c49-2af1f7002399
md"Step response: input complimentary sensitivity function, the transfer function from load disturbance to control signal."

# ╔═╡ 08ac947d-3efa-4829-9b61-428f46e266cf
Ts = parse(Float64, Tss); # This is defined in a textbox and can not be placed in the same cell as the binding, the cell defining the textbox is not evaluated when the contents of the widget changes.

# ╔═╡ 1261b50c-754c-435e-a6d5-fcbbd2a48352
begin
	discrete_dynamics = rk4(dynamics, Ts)
	P = let 
		A,B = MPC.linearize(discrete_dynamics, zeros(nx), zeros(nu), 0, 0)
		C,D = MPC.linearize(discrete_dynamics.measurement, zeros(nx), zeros(nu), 0, 0)
	    ss(A, B, C, D, Ts)
	end
end;

# ╔═╡ 6ab8c32f-e37e-4acb-a662-6d2204305b22
begin
	P_subset = P[outs, ins]
end;

# ╔═╡ ce1f2d0a-518e-4fef-a644-ea1ed06ae839
begin
	# Sample input noise covariance
	Ac,_ = MPC.linearize(dynamics, zeros(nx), zeros(nu), 0, 0)
    M = exp([Ac.*Ts  Bw.*Ts;
            zeros(nw, nx+nw)])
    Bwd = M[1:nx, nx+1:nx+nw]
end;

# ╔═╡ 13c34348-c481-4a24-87b2-21eae8646db5
begin
	Q1 = spdiagm(ones(nx)) 					 # state cost matrix
	Q2 = exp10(logq2)*Ts * spdiagm(ones(nu)) # control cost matrix
	Q3 = exp10(logq3)*Ts * spdiagm(ones(nu)) # control derivative cost matrix

	R1 = Symmetric(Bwd*Bwd' + 1e-10I)	 # dynamics noise covariance matrix
	R2 = exp10(logr2)*0.01^2 * I(ny) # measurement covariance matrix
end;

# ╔═╡ 9cfcc726-aec5-4b95-b1b3-e36c7cd15e1d
begin
	kf = KalmanFilter(P.A, P.B, P.C, 0, R1, R2)
	if obs == "openloop"
		observer = DyadControlSystems.StateFeedback(discrete_dynamics, x0)
	elseif obs == "kalman"
		observer = kf
	elseif obs == "extkalman"
		observer = ExtendedKalmanFilter(kf, rm_params(discrete_dynamics, 0), rm_params(measurement, 0))
	end
end;

# ╔═╡ 782604a8-e064-47b6-b8e8-a0b80f5bf454
begin
	## Define problem structure	
	solver = OSQPSolver(
	    eps_rel = 1e-5,
	    max_iter = 500,        # in the QP solver
	    check_termination = 5, # how often the QP solver checks termination criteria
	    sqp_iters = 1,
	    dynamics_interval = 2, # The linearized dynamics is updated with this interval
	)
	prob = LQMPCProblem(discrete_dynamics;
		observer, Q1, Q2, Q3,
	    constraints,
	    N, xr, solver,
	)
	Random.seed!(0)
	history = MPC.solve(prob; x0, T = 500, verbose = false, noise)	
end;

# ╔═╡ 6c578452-7a6e-454d-9499-2b306cf32d5f
plot(history, xlims=(0,10))

# ╔═╡ 01f11106-3c50-4af9-9ffe-35b7694e105e
begin
	fs = log10(0.5/Ts)
	w = 2π .* exp10.(LinRange(fs-4, fs, 500))
	K = kalman(P, Matrix(kf.R1), Matrix(kf.R2))
	L = RobustAndOptimalControl.lqr3(P, Matrix(Q1), Matrix(Q2), Matrix(Q3))
	cont = ControlSystemsBase.observer_controller(P, L, K)
	#RobustAndOptimalControl.gangoffourplot2(P, cont, w, xlabel="")
end;

# ╔═╡ 5a3d96f9-94ba-49f6-8bc9-b43830b7f1b8
let
	S,D,N,T = RobustAndOptimalControl.gangoffour2(P, cont)
	sig = true
	bp = (G,args...; kwargs...) -> sig ? sigmaplot(G,args...; kwargs...) : bodeplot(G[outs, ins],args...; plotphase=false, kwargs...)
	f1 = bp(S, w; title="S = 1/(1+PC)", lab="", layout=1)
	Plots.hline!([1.0 1.25 1.5], l=(:dash, [:green :orange :red]), sp=1, lab=["1.0" "1.25" "1.5"], ylims=(0.01,2.8))
	f2 = bodeplot(D[outs, ins], w; title="D = P/(1+PC)", plotphase=false, xlabel="", layout=1)
	Plots.hline!([1], l=(:black, :dash), primary=false)
	f3 = bodeplot(N[ins, outs], w; title="N = C/(1+PC)", plotphase=false, xlabel="", layout=1)
	f4 = bp(T, w; title="T = PC/(1+PC)", ylims=(0.001,2.8), lab="", layout=1)
	Plots.hline!([1], l=(:black, :dash), primary=false)
	Plots.plot(f1,f2,f3,f4, ticks=:default, ylabel="", legend=:bottomright)
end

# ╔═╡ 53df6877-66b5-431a-8a0a-a34b74396581
begin
	steplength = length(history.U)*history.Ts
	plot(step(output_comp_sensitivity(P, cont)[outs, outs], steplength), xticks=false, yticks=true, legend=true, layout=length(outs))
end

# ╔═╡ 8b7ed4b1-f141-482c-a899-af15d898f846
plot(step(input_comp_sensitivity(P, cont)[ins, ins], steplength), legend=true)

# ╔═╡ 1a2df374-072f-48ee-aaa4-2d377e9adc29
if ControlSystemsBase.issiso(P)
	nyquistplot(P*cont, w, xlims=(-3,1), ylims=(-3, 1), Ms_circles=[1, 1.25, 1.5])
end

# ╔═╡ Cell order:
# ╟─30d87043-e323-48fe-86a5-2b82e6e31b4c
# ╟─e8d063e4-5745-11ec-364d-4792105f762a
# ╟─0ec1bd8d-c5c7-4b69-ac6a-adda668ab403
# ╟─4eb5a13a-cd4e-4348-92e4-0632a9d079b3
# ╟─8e8dadae-3d5b-4763-bc1c-173782a26a35
# ╠═c777e06d-0b60-4044-9750-b391176879dc
# ╟─57e01a59-9749-463f-9cf3-5caa6fce65f4
# ╠═8c3855d3-0e9a-453d-ae4b-61a06433653b
# ╠═1261b50c-754c-435e-a6d5-fcbbd2a48352
# ╟─9cfcc726-aec5-4b95-b1b3-e36c7cd15e1d
# ╠═13c34348-c481-4a24-87b2-21eae8646db5
# ╟─fd17fd3f-3674-49e1-bfdd-eefae5c6315e
# ╟─eaf5ca2b-1b17-4311-9f24-ba4ea220c8df
# ╟─3e17f179-739d-46e4-9677-8fc2bc23ee54
# ╟─b78fbfb4-b71f-4a8f-aada-d686c790c792
# ╟─6c578452-7a6e-454d-9499-2b306cf32d5f
# ╟─6cefe746-8dae-4dcc-a7f9-880e7047db41
# ╟─5a3d96f9-94ba-49f6-8bc9-b43830b7f1b8
# ╟─7cea25cb-11f1-47e3-ae05-61c58ffa4a3e
# ╟─4c89c492-0294-4209-99d3-4f8d7fc3b850
# ╟─53df6877-66b5-431a-8a0a-a34b74396581
# ╟─bb1b2b8d-bdbf-46f9-8c49-2af1f7002399
# ╟─8b7ed4b1-f141-482c-a899-af15d898f846
# ╟─1a2df374-072f-48ee-aaa4-2d377e9adc29
# ╟─782604a8-e064-47b6-b8e8-a0b80f5bf454
# ╟─01f11106-3c50-4af9-9ffe-35b7694e105e
# ╟─6ab8c32f-e37e-4acb-a662-6d2204305b22
# ╟─08ac947d-3efa-4829-9b61-428f46e266cf
# ╟─ce1f2d0a-518e-4fef-a644-ea1ed06ae839
