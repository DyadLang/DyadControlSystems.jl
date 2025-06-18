using DyadControlSystems
using ModelingToolkit
using ModelingToolkitStandardLibrary.Blocks
using Plots
using Test

t = ModelingToolkit.t_nounits
connect = ModelingToolkit.connect

# Simple SISO LTI system for sensitivity analysis test
@mtkmodel FirstOrderTest begin
    @components begin
        in = Blocks.Step()
        fb = Blocks.Feedback()
        fo = SecondOrder(k = 1, w = 1, d = 0.1)
    end
    @equations begin
        connect(in.output, :u, fb.input1)
        connect(fb.output, :e, fo.input)
        connect(fo.output, :y, fb.input2)
    end
end

@named model = FirstOrderTest()

spec = DyadControlSystems.ClosedLoopSensitivityAnalysisSpec(;
    name = :FirstOrderTestSensitivity,
    model = model,
    analysis_points = ["y"], 
    loop_openings = String[], 
)

asol = DyadControlSystems.run_analysis(spec)

fig_bode = DyadControlSystems.artifacts(asol, :BodePlot)
@test fig_bode !== nothing

@test tf(asol.S.sys) ≈ feedback(1, tf(1, [1,2*0.1, 1]))


## "MIMO"

spec = DyadControlSystems.ClosedLoopSensitivityAnalysisSpec(;
    name = :FirstOrderTestSensitivity,
    model = model,
    analysis_points = ["y", "e"],
    loop_openings = String[], 
)

asol2 = DyadControlSystems.run_analysis(spec)

fig_bode = DyadControlSystems.artifacts(asol2, :BodePlot)
@test tf(asol2.S[1,1]) ≈ tf(asol.S)
@test tf(asol2.S[2,2]) ≈ tf(asol.S)