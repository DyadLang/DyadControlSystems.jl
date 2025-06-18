using DyadControlSystems
using ModelingToolkit
using ModelingToolkitStandardLibrary.Blocks
using Plots
using Test

t = ModelingToolkit.t_nounits
connect = ModelingToolkit.connect

# Simple SISO LTI system for linearization test
@mtkmodel SecondOrderTest begin
    @parameters begin
        k = 2.0
        T = 1.0
    end
    @components begin
        in = Blocks.Step()
        out = RealOutput()
        fo = SecondOrder(k = k, w = 1, d = 0.1)
    end
    @equations begin
        connect(in.output, :u, fo.input)
        connect(fo.output, :y, out)
    end
end

@named model = SecondOrderTest()

inputs = ["u"]
outputs = ["y"]

# LinearAnalysisSpec
spec = DyadControlSystems.LinearAnalysisSpec(;
    name = :SecondOrderTestAnalysis,
    model = model,
    inputs,
    outputs,
    wl = 0.01,
    wu = 100.0,
    # num_frequencies = 100,
    # duration = Inf,
)

asol = DyadControlSystems.run_analysis(spec)

fig_bode = DyadControlSystems.artifacts(asol, :BodePlot)
@test fig_bode !== nothing

fig_margin = DyadControlSystems.artifacts(asol, :MarginPlot)
@test fig_margin !== nothing

fig_step = DyadControlSystems.artifacts(asol, :StepResponse)
@test fig_step !== nothing

fig_step_info = DyadControlSystems.artifacts(asol, :StepInfoPlot)
@test fig_step_info !== nothing

rl = DyadControlSystems.artifacts(asol, :RootLocusPlot)
@test rl !== nothing

pz = DyadControlSystems.artifacts(asol, :PoleZeroMap)
@test pz !== nothing

nyq = DyadControlSystems.artifacts(asol, :NyquistPlot)
@test nyq !== nothing

rga = DyadControlSystems.artifacts(asol, :RGAPlot)
@test rga !== nothing

plot(fig_bode, fig_margin, fig_step, fig_step_info, rl, pz, nyq, size=(1200, 1200), link=:none)

##

obs = DyadControlSystems.artifacts(asol, :ObservabilityReport)
dr = DyadControlSystems.artifacts(asol, :DampReport)
step_info = DyadControlSystems.artifacts(asol, :StepInfo)
@test step_info isa DyadControlSystems.DataFrame

