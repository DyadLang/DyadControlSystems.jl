using ControlSystemsBase, RobustAndOptimalControl, Plots, LinearAlgebra


function print_phase_analysis(dm)
	ϕm = dm.ϕm
	cg = cgrad([:red, :yellow, :green], [0.15, 0.28, 0.6])
	bg = reshape(10:0.5:80, 1, :)
	ϕm = clamp(ϕm, bg[1], bg[end])

	rs(x) = (x-bg[1])/bg[end] * length(bg)
	rescaled = rs(ϕm)

    vals = 10:10:80
	xticks = (
		rs.(vals),
		vals
	)

	heatmap(bg; c=cg, yaxis=false, xticks,
	size=(500, 100), colorbar=false, title="Phase margin [°]")
	vline!([rescaled], l=(:black, 3), primary=false)
end

function print_gain_analysis(dm)
	gm = dm.gm[2]
	cg = cgrad([:red, :yellow, :green], [0.15, 0.28, 0.6])
	bg = reshape(0:0.01:1, 1, :)
	gm = clamp(log10(gm), bg[1], bg[end])

	rs(x) = x/bg[end] * length(bg)
	rescaled = rs(gm)

	s = 5
	xticks = (
		1:s:length(bg),
		round.(exp10.(bg[1:s:end]), digits=2)
	)

	vals = [1, 1.5, 2, 3, 5, 10]
	xticks = (
		rs.(log10.(vals)),
		vals
	)

	heatmap(bg; c=cg, yaxis=false, xticks,
	size=(500, 100), colorbar=false, title="Gain margin")
	vline!([rescaled], l=(:black, 3), primary=false)
end

function print_ms_analysis(m, title)
	cg = cgrad([:green, :yellow, :red], [0.15, 0.3, 0.6])
	bg = reshape(0:0.002:log10(3), 1, :)
	gm = clamp(log10(m), bg[1], bg[end])


	rs(x) = x/bg[end] * length(bg)
	rescaled = rs(gm)

	s = 8
	vals = [1, 1.1, 1.2, 1.3, 1.5, 2, 2.5, 3]
    # vals = sort([vals; round(m, digits=2)])
	xticks = (
		rs.(log10.(vals)),
		vals
	)

	heatmap(bg; c=cg, yaxis=false, xticks,
	size=(500, 100), colorbar=false, title)
	vline!([rescaled], l=(:black, 3), primary=false)
end

function print_ncf_analysis(m, title)
	cg = cgrad([:red, :yellow, :green], [0.15, 0.24, 0.5])
	bg = reshape(0:0.01:1, 1, :)
	mc = clamp(m, bg[1], bg[end])


	rs(x) = x/bg[end] * length(bg)
	rescaled = rs(mc)

	s = 8
	# vals = sort([0:0.1:1; round(m, digits=2)])
    vals = 0:0.1:1
	xticks = (
		rs.(vals),
		vals
	)

	heatmap(bg; c=cg, yaxis=false, xticks,
	size=(500, 100), colorbar=false, title)
	vline!([rescaled], l=(:black, 3), primary=false)
end



"""
    analyze_robustness(P, C = nothing; Tf = nothing)

Analyze the closed-loop properties of the feedback interconnection
```
              d             
     ┌─────┐  │  ┌─────┐    
r  e │     │u ▼  │     │ y  
──+─►│  C  ├──+─►│  P  ├─┬─►
  ▲  │     │     │     │ │  
 -│  └─────┘     └─────┘ │  
  │                      │  
  └──────────────────────┘  
```

# Arguments:
- `P`: An LTI plant model.
- `C`: An LTI controller model (optional), if not provided, unit feedback is assumed.
- `Tf`: Duration of the step response.

# Result 
The returned plot contains the following:
- Closed-loop transfer functions ``S = 1/(1+PC)``, ``T = PC/(1+PC)``, ``C / (1+PC)``, ``P/(1+PC)``
- Disk margin, a measure of the robustness w.r.t. combined gain and phase variations
- Gain and phase margins, a measure of the robustness w.r.t. individual gain and phase variations
- Step response of the closed-loop system. The step response is simulated using the four closed-loop transfer functions, and have as inputs the signals ``r`` and ``d`` shown in the diagram above, i.e., disturbances entering at ``y`` and ``u``. The four responses indicate
    - ``u -> u``: The control-signal response to a unit step input disturbance
    - ``u -> y``: The plant output response to a unit step input disturbance
    - ``y -> u``: The control-signal response to a unit reference step
    - ``y -> y``: The plant output response to a unit reference step
"""
function analyze_robustness(P, C=nothing; pos_feedback=false, Tf = nothing)
	# Locally overload step to handle optional Tf
	step(sys, ::Nothing) = ControlSystemsBase.step(sys)
	step(sys, Tf::Real) = ControlSystemsBase.step(sys, Tf)

    if C === nothing
        L = P
        C = ss(I(L.ny))
        plot_time = plot(step(feedback(L), Tf), title="Reference step response")
    else
        C = C
        L = P*C
        (ny, nu) = size(P)
        Zperm = [(1:ny).+nu; 1:nu]
        if C isa NamedStateSpace && P isa NamedStateSpace
            gof0 = feedback(C, P; w2=:, z2=:, pos_feedback) # y,u from r, d
            gof = gof0
        else
            gof0 = feedback(C, P; W2=:, Z2=:, pos_feedback, Zperm) # y,u from r, d
            gof = named_ss(gof0, u = [:r^P.ny; :d^P.nu], y = [:y^P.ny; :u^P.nu])
        end
        plot_time = if pos_feedback
			un = gof.u
			gof = gof*diagm([-1*ones(P.ny); ones(P.nu)]) # To get correct signs in plot
			gof.u .= un # To not change the original signal names
            plot(step(gof, Tf)) 
        else
            plot(step(gof, Tf))
        end
    end
    if pos_feedback
        L = -L
        P = -P
    end
	dm = diskmargin(L)


	eye = ss(I(L.ny))
	plot_indicators = plot(
		print_phase_analysis(dm),
		print_gain_analysis(dm),
        print_ncf_analysis(ncfmargin(P, C)[1], "Normalized Coprime-factor margin"),
		print_ms_analysis(hinfnorm2(feedback(eye, L))[1], "Peak sensitivity"),
		print_ms_analysis(hinfnorm2(feedback(L))[1], "Peak complementary sensitivity"),
		size=(600, 400),
		layout=(5,1),
        topmargin=-2Plots.mm,
	)

    plot_gof = gangoffourplot(P, C, xlabel="", label="")
    plot_dm = plot(dm)
	plot_nyquist = nyquistplot(L, lab="\$L_o(s)\$")
	plot_marg = marginplot(L, lab="\$L_o(s)\$")
    plots = [plot_gof, plot_indicators, plot_dm, plot_time, plot_nyquist, plot_marg]

    plot(plots..., layout=length(plots)) # layout=(2,2), size=(1200, 1000))

end

