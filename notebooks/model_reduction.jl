### A Pluto.jl notebook ###
# v0.19.11

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

# ╔═╡ b68097fd-2236-47b7-971f-e5b260f1c577
const quickactivate = true

# ╔═╡ c088d5e4-6d5a-11ec-1721-cd55d7ca5b8c
begin
	quickactivate
	using DyadControlSystems, ControlSystems, PlutoUI, Plots, StaticArrays, RobustAndOptimalControl
	import DSP
	using LinearAlgebra, Random
	using Plots.PlotMeasures
	const Layout=PlutoUI.ExperimentalLayout
end;

# ╔═╡ 9c87a7e2-fe86-4e90-8f23-47bef42e116d
w = exp10.(LinRange(-2, 2, 200))  # Frequency vector for plot

# ╔═╡ 359410b6-18c1-4b0f-a92e-fb688815b539
P = ssrand(2,2,10, proper=false, stable=true);  # System model to be reduced

# ╔═╡ 8d3e2ac3-c02a-46e6-b7b8-ca0d418fce04
md"""
# Help
The model-reduction app helps you find a reduced-order model of suitable order and fit. The reduced-order model is calculated using balanced truncation or balanced residualization, with optional frequency weighting.

## Options
- **Method**:
    - Balanced truncation is the most common model-reduction method. It transforms the model to, so called, balanced form, where the observability and controllability gramians are equal and diagonal. After the transformation, the modes with the least contribution to the input-output behavior of the system are removed. If the model contains unstable modes, an additive decomposition $P(s) = P_{stable}(s) + P_{unstable}(s)$ is performed and only $P_{stable}$ is reduced. In this case, the reported Hankel singular values belong to $P_{stable}$ only.
    - Coprime-factor reduction performs a coprime factorization of the model into $P(s) = M(s)^{-1}N(s)$ where $M$ and $N$ are stable factors even if $P$ contains unstable modes. After this, the system $NM = \begin{bmatrix}N & M \end{bmatrix}$ is reduced using balanced truncation and the final reduced-order model is formed as $P_r(s) = M_r(s)^{-1}N_r(s)$. For this method, the Hankel signular values of $NM$ are reported and the reported errors are $||NM - N_rM_r||_\infty$. This method is of particular interest in closed-loop situations, where a model-reduction error $||NM - N_rM_r||_\infty$ no greater than the normalized-coprime margin of the plant and the controller, guaratees that the closed loop remains stable when either $P$ or $K$ are reduced. The normalized-coprime margin can be computed with `ncfmargin(P, K)`. 
- **Order**: The order (number of states) of the reduced order model.
- **Match DC-gain**: If selected, the reduced-order model will match the full-order model exactly at frequency 0. This generally gives a better overall model fit.
- **Maximum order for exhaustive calculation**: Indicates the maximum order of the full-order model for which all possible reduced order models are pre-calculated. Above this threshold, a single reduced-order model corresponding to the chosen order is calculated and and error bound is shown instead of actual errors. Setting this number too high may cause excessive computation times.
- **Relative fit**: Optimize the relative model error. This is equivalent with an output-weighted fit where the weight is chosen as the inverse of the full-order model. This option is only available for models with a stable inverse.
- **Frequency focus**: Penalize model errors in a certain frequency range more heavily. The app supports selecting an upper and a lower frequency bound. The model-reduction code supports weighting by arbitrary stable linear filters. The frequency-focus option is only available if "relative fit" is deselected. When frequency focus is selected, sliders appear that allow you to select the ($\log_{10}$) frequency range to focus the fit within. If frequency-focus is selected, the Hankel singular values belong to the frequency-weighted gramians and the reported error is frequency weighted.
- **The frequency vector** `w` used for plotting can be modified in order to extend or focus the illustration of the model fit.

## Visualization
- The top figure illustrates the singular values of the full and reduced-order models, as well as the model error.
- The middle figure shows the Hankel singular values of the full-order model, as well as lines indicating 1% and 0.1% of the maximum singular value. For full-order models below order 40, this figure also indicates the $H_\infty$ error resulting from choosing a particular order for the reduced-order model.
- The bottom figure displays the cumulative sum of the Hankel singular values up to and including a particular model order.

## Advice for successful model-order reduction.
- If the reduced-order model is to be used for closed-loop control, it is generally advicable to ensure that the model fit is high in the *frequency region around the desired crossover frequency*.
- Verify that there are no large *unmodeled peaks* in the frequency response that might cause instability or non-robustness of a closed-loop system.
- If the reduced-order model is to be used for filtering-purposes, e.g., as a feedforward filter, a good fit at low frequencies is generally desireable.
"""

# ╔═╡ 4842bf05-c90b-4dda-8a13-44f4bc995b3c
begin
	gui_matchdc = @bind residual CheckBox(default=true)
	gui_relative_fit = @bind relative CheckBox()
	gui_calcall_th = @bind calcall_th confirm(TextField(5, default="40"))
end;

# ╔═╡ c30d434d-a5ad-4eec-99c2-99de81390729
begin
	methods = [
		"baltrunc" => "Balanced truncation",
		"coprime" => "Coprime-factor reduction",
	]
	gui_method = @bind method Select(methods)
end;

# ╔═╡ d7ff06c0-3322-43d7-a524-24e382c80230
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
	"ec7546b5-aa90-456c-aeb8-a95a648103cb",
	"9b885df5-f6ab-430a-b690-8a1f9b74d098",
	"7bb334b7-2e5b-491f-b3e3-0c351c2076da",
	"654c2aab-83e5-486c-ba2f-b1815abebd57",
	"8d3e2ac3-c02a-46e6-b7b8-ca0d418fce04",
];
const allCells = [...document.querySelectorAll("pluto-cell")];
allCells.filter(c => !useful_cells.includes(c.id)).map(c => c.style = "display: none");
</script>
"""

# ╔═╡ 9b885df5-f6ab-430a-b690-8a1f9b74d098
if relative || method == "coprime"
	freqfocus = false
	md""""""
else
	gui_freqfocus = @bind freqfocus CheckBox()
    Layout.hbox([md"Frequency focus:", gui_freqfocus])
end

# ╔═╡ 654c2aab-83e5-486c-ba2f-b1815abebd57
if freqfocus
	gui_logw = @bind logw01 RangeSlider(log10(w[1]):0.1:log10(w[end]), show_value=true)
	Layout.grid(
		 [
			 md" $\log \omega$" gui_logw;
		 ]
	 )
else
	logw01 = 1:2
	md""
end

# ╔═╡ 3f18f4fa-46fb-415e-8d7c-726d8faa90e5
begin
	logw0, logw1 = extrema(logw01)
	if logw1 == logw0
		logw1 += 0.1
	end
end;

# ╔═╡ 55ea800a-a55e-4cc6-897f-b488fab639da
begin
	reducer = DyadControlSystems.ModelReducer(method, P)
	stab = reducer.n_unstable == 0
end;

# ╔═╡ e5a9a26e-10f4-4648-9732-ea47f18697b4
gui_order = @bind n Slider(DyadControlSystems.available_orders(reducer), show_value=true);

# ╔═╡ ec7546b5-aa90-456c-aeb8-a95a648103cb
begin
	elems = 	[
			md"Method"  gui_method;
			md"Order"  gui_order;
		    md"Match DC gain" gui_matchdc;
			md"Maximum order for exhaustive calculation" gui_calcall_th
		]
	if reducer.stable_inverse
		elems = [elems; [md"Relative fit" gui_relative_fit]]
	end
	Layout.grid(elems)
end

# ╔═╡ 9e35abf7-eb57-4aeb-8b84-f6529160a945
begin
	if relative
		W = fudge_inv(P)
		isstable(W) || error("Relative fit is only supported for systems with a stable inverse, the specified system has non-minimum phase zeros and the inverse is unstable. Try using the frequency focus to shape the reduced-order model.")
	else
		w0 = exp10(logw0)
		w1 = exp10(logw1)
		wmax = 1e9
		fc = DSP.analogfilter(DSP.Bandpass(w0, w1), DSP.Butterworth(2))
		tfc = DSP.PolynomialRatio(fc)
		W = tf(DSP.coefb(tfc), DSP.coefa(tfc))
	end
end;

# ╔═╡ 4ad2c774-a322-4576-879e-e45a3bb27eb1
begin
	calcall = P.nx <= parse(Int, calcall_th)
	if calcall
		Prs, hs, errors = DyadControlSystems.model_reduction(reducer, P, DyadControlSystems.available_orders(reducer); residual, frequency_focus = freqfocus || relative, W)
		errors = [fill(Inf,reducer.n_unstable); errors]
	else
		# Theoretical error bound
		Pr0, hs, error0 = DyadControlSystems.model_reduction(reducer, P, n; residual, frequency_focus = freqfocus || relative, W)
		Prs = [Pr0]
		errors = RobustAndOptimalControl.error_bound(hs)
	end
	chs = cumsum(replace(hs, Inf => 0))
	chs ./= chs[end]
	# suggested_n = findfirst(>(0.99), chs)
end;

# ╔═╡ 7d60253e-efcc-48b4-ba58-904cef88b7fc
fig_hankel = begin
	f1 = bar(hs, yscale=:log10, lab="", title="Hankel singular values")
	hline!(sum(hs)*[0.01 0.001], lab=["1.0% of total" "0.1% of total"], l=(:dash,))
	ylims = (max(1e-7*hs[1+reducer.n_unstable], hs[end]), Inf)
	bar!([n], [hs[n:n]]; yscale=:log10, lab="", primary=false, c=:red, ylims)
	error_str = stab ? "H∞ error" : "L∞ error"
	if calcall
		plot!(errors; lab=freqfocus && !relative ? "Frequency-weighted"*error_str : error_str, l=(3,), ylims)
	else
		plot!(errors, lab=error_str*" bound", l=(3,))
	end
	f2 = scatter(chs, lab="Cumulative energy", xlabel="Model Order", legend=:bottomright)
	vline!([n], primary=false, l=(:dash, :red))
	hsvdplot = plot(f1, f2, layout=(2,1), link=:x)
end;

# ╔═╡ 022d3318-8f36-47e4-b0b2-b00a2abda6df
Pr = calcall ? Prs[n-reducer.n_unstable] : Prs[];

# ╔═╡ d80d322e-fd2b-4b3e-bfef-500731b62c5f
fig_sigma = begin
	sigmaP = sigma(P, w)[1]'
	fig = plot(ControlSystemsBase._to1series(w, sigmaP)..., c=1, lab="Full model")
	sigmaplot!(Pr,   w, c=2, lab="Reduced order $n")
	sigmaplot!(P-Pr, w, c=3, lab="Approximation error",
		legend=:bottomright,
		title=ControlSystemsBase.issiso(P) ? "Magnitude" : "Singular values",
		ylims = (1e-3*maximum(sigmaP), Inf)
	)
	if freqfocus && !relative
		vline!([w0 w1], l=(:black, :dash), primary=false)
	end
	svplot = fig
end;

# ╔═╡ 7bb334b7-2e5b-491f-b3e3-0c351c2076da
plot(fig_sigma, fig_hankel, size=(1200, 600), margin=5mm) # The margin is there to not hide labels with GR

# ╔═╡ 9041787a-f86c-4e11-b2d3-8e675bb867e8
function DSP.Filters.normalize_freq(w::Real, fs::Real)
    w <= 0 && error("frequencies must be positive")
    w
end # DSP is buggy for analog filters https://github.com/JuliaDSP/DSP.jl/issues/341

# ╔═╡ Cell order:
# ╟─ec7546b5-aa90-456c-aeb8-a95a648103cb
# ╟─9b885df5-f6ab-430a-b690-8a1f9b74d098
# ╟─654c2aab-83e5-486c-ba2f-b1815abebd57
# ╟─7bb334b7-2e5b-491f-b3e3-0c351c2076da
# ╠═b68097fd-2236-47b7-971f-e5b260f1c577
# ╠═c088d5e4-6d5a-11ec-1721-cd55d7ca5b8c
# ╠═9c87a7e2-fe86-4e90-8f23-47bef42e116d
# ╠═359410b6-18c1-4b0f-a92e-fb688815b539
# ╟─8d3e2ac3-c02a-46e6-b7b8-ca0d418fce04
# ╟─d80d322e-fd2b-4b3e-bfef-500731b62c5f
# ╟─7d60253e-efcc-48b4-ba58-904cef88b7fc
# ╟─e5a9a26e-10f4-4648-9732-ea47f18697b4
# ╟─4842bf05-c90b-4dda-8a13-44f4bc995b3c
# ╟─c30d434d-a5ad-4eec-99c2-99de81390729
# ╟─d7ff06c0-3322-43d7-a524-24e382c80230
# ╟─3f18f4fa-46fb-415e-8d7c-726d8faa90e5
# ╟─55ea800a-a55e-4cc6-897f-b488fab639da
# ╟─4ad2c774-a322-4576-879e-e45a3bb27eb1
# ╟─022d3318-8f36-47e4-b0b2-b00a2abda6df
# ╟─9e35abf7-eb57-4aeb-8b84-f6529160a945
# ╟─9041787a-f86c-4e11-b2d3-8e675bb867e8
