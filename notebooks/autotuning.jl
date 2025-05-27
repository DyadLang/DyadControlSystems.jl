### A Pluto.jl notebook ###
# v0.19.19

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

# ╔═╡ ec6b7e8a-f808-4a9a-b1c6-7783d753fa3f
const quickactivate = true

# ╔═╡ e169496c-4229-11ec-1c03-f1a6c4cd7167
begin
	quickactivate
	using DyadControlSystems, PlutoUI, Plots
	const Layout=PlutoUI.ExperimentalLayout;
	md"**Packages Loaded!**"
end

# ╔═╡ eca63755-08f6-41fe-9520-94a8fdc71270
md"""
## PID autotuning
Specify an initial guess for the controller parameters, $p_0 = [k_P, k_I, k_D, T_f]$.
"""

# ╔═╡ 0eb502fe-d85f-464a-8b7a-e3581dc28a98
begin
	p00 = Float64[0.1, 0.1, 0, 0.001] # Initial parameter guess, kp, ki, kd, T_filter
	p0 = copy(p00)
end

# ╔═╡ 1442720f-6305-4ff0-a048-10d66d2d5006
begin
	T = 4 # Time constant
	L = 1 # Delay
	K = 1 # Gain
	P = delay(L)*tf(K, [T, 1.0]) # process dynamics
end;

# ╔═╡ 2331628e-01d7-438c-b0ad-b36eb8984c78
begin
	gui_ms =  @bind Ms     Slider(1.01:0.01:2, default=1.2, show_value=true)
	gui_mt =  @bind Mt     Slider(1.01:0.01:2, default=1.2, show_value=true)
	gui_mks = @bind logMks Slider(-1:0.1:3,    default=1,   show_value=true)
	
	s = Layout.grid([
	 	md" $M_S$:" gui_ms  	md" $M_T$:" gui_mt  	md" $\log_{10}(M_{CS})$:" gui_mks # Initial space in md string required
	])

	Ms = max(1.01, Ms)
	Mt = max(1.01, Mt)
	s
end

# ╔═╡ c0296af3-4450-4864-934f-102fd860b24d
md"""
A linear statespace system representing the controller can be constructed with the following code
"""

# ╔═╡ ac9164c7-041d-44cb-82aa-0cc3038726f0
md"and an ODESystem representing the controller can be obtained by calling `OptimizedPID(res)`"

# ╔═╡ d17588ac-9a35-4bff-9b2c-093b737c4a00
md"""
### Instructions
This app tunes the parameters of the controller $C$ such that the response to a step in the disturbance $d$ is minimized, subjects to constraints on the maximum magnitudes on the sensitivity functions
-  $S = \dfrac{1}{1+PC}$ The transfer function from $n$ to $y$, from $n$ to $e$ and from $r$ to $e$.
-  $T = \dfrac{PC}{1+PC}$ The transfer function from $r$ to $y$.
-  $CS = \dfrac{C}{1+PC}$ The transfer function from $n$ to $u$.

#### Initial guess
The initial guess for the controller parameters takes the form
```math
\begin{bmatrix}
k_P, \, k_I, \, k_D, \, T
\end{bmatrix}
```
When the metric is set to `:IE`, the length of the initial guess determines the type of controller that will be tuned. If only two parameters are provided, a PI controller is tuned, if the length is three, a PID controller is tuned. When the metric is set to `:IAE`, a PID controller with filter will always be tuned, but the parameters may be upper bounded by entering a bound in the fields labeled $\operatorname{max} k$.

#### Figures
The upper figures show the sensitivity functions $S, T$ and $CS$ together with the specified constraints on their maximum magnitudes, $M_S, M_T, M_{CS}$.

The closed-loop systems reponse to a step disturbance at the input to $PS$ is shown in the lower left. The lower right plot shows the Nyquist plot of the loop-transfer function $PC$, together with circles representing the constraints $M_S, M_T, M_{CS}$.

#### Sliders
The sliders let you set constraints on the sensitivity functions:
-  $M_S$: Maximum allowed sensitivity function magnitude.
-  $M_T$: Maximum allowed complementary sensitivity function magnitude.  
-  $M_{CS}$: Maximum allowed magnitude of transfer function from process output to control signal, sometimes referred to as noise sensitivity.

#### Grid
When you start using the app, use a small number of frequency bins, say $N_f = 30-50$, this makes the optimization faster and the interactive experience better. Make sure the frequency range covers the peaks of the sensitivity functions. The optimization will force the filter to break down within the frequency grid, so the upper limit of the grid determines an upper bound for the filter bandwidth.

When you are satisfied with the settings of the sliders, increase the number of frequency grid points to get a more accurate result.
#### Metric
The `IE` problem optimizes integrated error $\int e(t) dt$ (not integrated *absolute* error). This problem is easy and fast to solve and corresponds well to IAE if the system is well damped. If this metric is chosen, a PI or PID controller is tuned. The method requires a stabilizing controller as an initial guess. If the plant is stable, the zero controller is okay.

If the response is oscillatory, the `IE` metric is expected to perform poorly. If the `IAE` metric is chosen, a PID controller with a low-pass filter is tuned by minimizing $\int |e(t)| dt$. This problem is nonconvex and can be difficult to solve. This method can be initialized with the `IE` method.

If it appears difficult to find a stabilizing controller with either the `IAE` or `IE followed by IAE` methods, choose the `IE` method and tune the sliders until a stabilizing controller is found, then switch to the `IE followed by IAE` method.

$(LocalResource(joinpath(DyadControlSystems.NOTEBOOKS_PATH, "./block_diagram.png")))
"""

# ╔═╡ dda00ea9-f6b0-49cc-926c-b579a9ba3a85
md"""
#### References:
The IE-based optimization builds on 
> M. Hast, K. J. Astrom, B. Bernhardsson, S. Boyd. PID design by convex-concave optimization. *European Control Conference. IEEE. Zurich, Switzerland. 2013.*

The IAE-based optimization builds on
> K. Soltesz, C. Grimholt, S. Skogestad. Simultaneous design of proportional–integral–derivative controller and measurement filter by optimisation. *Control Theory and Applications. 11(3), pp. 341-348. IET. 2017.*
"""

# ╔═╡ 854f36b9-08a5-4525-a27c-24229c3a2a46
begin
	gui_f0  = @bind f0 TextField(5; default="0.01")
	gui_f1  = @bind f1 TextField(5; default="100")
	gui_nf  = @bind Nf confirm(TextField(5; default="50"))
	gui_tf  = @bind Tf confirm(TextField(5; default="25"))
	gui_ts  = @bind Ts confirm(TextField(5; default="0.1"))
	# gui_alg = @bind alg Select(["LD_CCSAQ" => "CCSAQ (balanced)", "LD_MMA" => "MMA (accurate)", "LD_SLSQP" => "SLSQP (fast)", "GN_ISRES" => "ISRES (global)", "GN_ORIG_DIRECT" => "ORIG_DIRECT (global)"])
	gui_metric = @bind metric Select(["IE" => "IE", "IAE" => "IAE", "IEIAE" => "IE followed by IAE"])

	gui_maxeval 		= @bind maxeval TextField(5; default="100")
	gui_tol 			= @bind tol TextField(5; default="1e-3")
	gui_use_randomstart = @bind randstart CheckBox(; default=false)
	gui_n_randomstart 	= @bind nrandstart TextField(5; default="500")

	gui_kpmax = @bind kpmax confirm(TextField(5; default="Inf"))
	gui_kimax = @bind kimax confirm(TextField(5; default="Inf"))
	gui_kdmax = @bind kdmax confirm(TextField(5; default="Inf"))
	gui_Tmax  = @bind Tmax confirm(TextField(5; default="Inf"))
end;

# ╔═╡ 7df13dc3-66db-4e96-959b-9d23b005de26
Layout.grid(
	[
		md"Frequency grid (start, stop, n):" 	gui_f0 		gui_f1 						gui_nf
		md"Metric:" 							gui_metric 	md"Discretization time:" 	gui_ts
		md"" 									md"" 		md"Simulation time:" 		gui_tf
	],
	column_gap = "-1cm"
)

# ╔═╡ 87412a45-632a-4a7c-b26f-b1aced5abad2
if metric != "IE"
	elements = [
		# md"Solver:"  			gui_alg 			md"" md""
		md"Max evaluations:" 	gui_maxeval 		md"" md""
		md"Use random starts:" 	gui_use_randomstart md"" md""
	]
	if randstart
		elements[end, end-1:end] = [md"Number of random starts:" gui_n_randomstart]
	end
	Layout.grid(elements, column_gap = "-1cm")
end

# ╔═╡ 24130c57-bb39-4953-a4d5-625cde2482a6
begin
	pmaxelements = [
		md"Parameter upper bounds:" md" $\operatorname{max} k_P$" gui_kpmax 	md" $\operatorname{max} k_I$" gui_kimax 	md" $\operatorname{max} k_D$" gui_kdmax 	md" $\operatorname{max} T_f$" gui_Tmax
	]
	Layout.grid(pmaxelements)
end

# ╔═╡ e3548a33-49fd-4ece-8d20-e50d07d67ca9
pmax = parse.(Float64, [kpmax, kimax, kdmax, Tmax]);

# ╔═╡ c100fae7-9c92-4dbd-98f2-d7519d260cff
begin
	ControlSystemsBase.issiso(P) || error("Only SISO systems supported by this app.")
	Mks = exp10(logMks)
	w = 2π .* exp10.(LinRange(log10(parse(Float64, f0)), log10(parse(Float64, f1)), parse(Int, Nf))) # frequency grid
	prob = AutoTuningProblem(; P, Ms, Mt, Mks, w, Ts=parse(Float64, Ts), Tf=parse(Float64, Tf), metric=Symbol(metric), pmax)
end;

# ╔═╡ f73b2f31-6b51-4773-b62c-99fc787f85c6
html"""<style>
main {
    max-width: 100%;
	padding: 0px;
}
header#pluto-nav, footer, pluto-helpbox {
	display: none;
}

</style>
<script>
const useful_cells = [
	"eca63755-08f6-41fe-9520-94a8fdc71270",
	"0eb502fe-d85f-464a-8b7a-e3581dc28a98",
	"7df13dc3-66db-4e96-959b-9d23b005de26",
	"2331628e-01d7-438c-b0ad-b36eb8984c78",
	"24130c57-bb39-4953-a4d5-625cde2482a6",
	"29729d37-b373-4f30-8354-777f39176721",
	"6a6a121e-fc02-49c1-ab5c-35ea77da7242",
	"c0296af3-4450-4864-934f-102fd860b24d",
	"0fb39195-807a-4fd2-ad81-3e51ab1c6aba",
	"ac9164c7-041d-44cb-82aa-0cc3038726f0",
	"d17588ac-9a35-4bff-9b2c-093b737c4a00",
	"dda00ea9-f6b0-49cc-926c-b579a9ba3a85",
];
const allCells = [...document.querySelectorAll("pluto-cell")];
allCells.filter(c => !useful_cells.includes(c.id)).map(c => c.style = "display: none");
</script>
"""

# ╔═╡ 97d55e2c-d4e9-4626-8da3-9b225b3d9196
begin
	wtres = with_terminal(show_value=false) do
		res = solve(prob, p0; alg=DyadControlSystems.IpoptSolver(exact_hessian=false), maxeval=parse(Int, maxeval), tole=parse(Float64, tol), random_start = randstart * parse(Int, nrandstart), pmax=parse.(Float64, [kpmax, kimax, kdmax, Tmax]));
	end
	res = wtres.value
end;

# ╔═╡ 29729d37-b373-4f30-8354-777f39176721
begin
	plot(res, size=(1000, 750), bg_color = res.ret == :CONSTRAINT_VIOLATION ? :pink : :white, alpha=1, legend=true)
	ylims!((1e-2, 3), subplot=1)
	#ylims!((1e-1, 20), subplot=2)
	#ylims!((-0.4, 1), subplot=3)
end

# ╔═╡ 6a6a121e-fc02-49c1-ab5c-35ea77da7242
if metric == "IE"
md"""
The found controller is
-  $k_P$ = $(round(res.p[1], sigdigits=4))
-  $k_I$ = $(round(res.p[2], sigdigits=4))
-  $k_D$ = $(round(res.p[3], sigdigits=4))

On the form $k_P + \dfrac{k_I}{s} + sk_D$
"""
else
md"""
The found controller is
-  $k_P$ = $(round(res.p[1], sigdigits=4))
-  $k_I$ = $(round(res.p[2], sigdigits=4))
-  $k_D$ = $(round(res.p[3], sigdigits=4))
-  $T$ = $(round(res.p[4], sigdigits=4))

On the form $\left(k_P + \dfrac{k_I}{s} + sk_D\right)\dfrac{1}{(sT)^2 + 2ζTs + 1}, \quad ζ = 1/√2$
"""
end

# ╔═╡ 0fb39195-807a-4fd2-ad81-3e51ab1c6aba
begin
	io = IOBuffer()
	show_construction(io, balance_statespace(res.K)[1], name="C")
	Kstring = Text(String(take!(io)))
end

# ╔═╡ 1a1c3742-4338-449f-b026-48143f3eb792
if metric != "IE"
	Layout.grid([md"Optimizer output"; wtres;;])
end

# ╔═╡ Cell order:
# ╠═ec6b7e8a-f808-4a9a-b1c6-7783d753fa3f
# ╠═e169496c-4229-11ec-1c03-f1a6c4cd7167
# ╟─eca63755-08f6-41fe-9520-94a8fdc71270
# ╠═0eb502fe-d85f-464a-8b7a-e3581dc28a98
# ╠═1442720f-6305-4ff0-a048-10d66d2d5006
# ╟─7df13dc3-66db-4e96-959b-9d23b005de26
# ╟─87412a45-632a-4a7c-b26f-b1aced5abad2
# ╟─2331628e-01d7-438c-b0ad-b36eb8984c78
# ╟─24130c57-bb39-4953-a4d5-625cde2482a6
# ╟─29729d37-b373-4f30-8354-777f39176721
# ╟─6a6a121e-fc02-49c1-ab5c-35ea77da7242
# ╟─c0296af3-4450-4864-934f-102fd860b24d
# ╟─0fb39195-807a-4fd2-ad81-3e51ab1c6aba
# ╟─ac9164c7-041d-44cb-82aa-0cc3038726f0
# ╟─1a1c3742-4338-449f-b026-48143f3eb792
# ╟─d17588ac-9a35-4bff-9b2c-093b737c4a00
# ╟─dda00ea9-f6b0-49cc-926c-b579a9ba3a85
# ╟─c100fae7-9c92-4dbd-98f2-d7519d260cff
# ╟─e3548a33-49fd-4ece-8d20-e50d07d67ca9
# ╟─854f36b9-08a5-4525-a27c-24229c3a2a46
# ╟─f73b2f31-6b51-4773-b62c-99fc787f85c6
# ╟─97d55e2c-d4e9-4626-8da3-9b225b3d9196
