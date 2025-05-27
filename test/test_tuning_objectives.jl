using DyadControlSystems
##
G = tf(2, [1, 0.3, 2])

w = exp10.(LinRange(-2, 2, 200))
W = tf([2, 0], [1, 1])
S = feedback(1, G)
f1 = bodeplot(S, w, plotphase=false)
plot!(MaximumSensitivityObjective(W), w, title="MaximumSensitivityObjective")

f2 = plot(step(G, 15))
plot!(OvershootObjective(1.2), title="OvershootObjective")
plot(OvershootObjective.(1.2*ones(2)), title="OvershootObjective", layout=2) # plot a vector of objectives

f3 = plot(step(G, 15))
plot!(RiseTimeObjective(0.6, 1), title="RiseTimeObjective")

f4 = plot(step(G, 15))
plot!(SettlingTimeObjective(1, 5, 0.2), title="SettlingTimeObjective")

f5 = plot(step(G, 15))
o = StepTrackingObjective(tf(1, [1,2,1]))
plot!(o, title="StepTrackingObjective")

f6 = plot(step(G*feedback(1, G), 15))
o = StepTrackingObjective(tf([1, 0], [1,2,1]))
plot!(o, title="StepRejectionObjective")

plot(f1,f2,f3,f4,f5,f6)

##

dm = diskmargin(G, 0, w)
f7 = plot(dm)
plot!(GainMarginObjective(2), title="GainMarginObjective")
plot!(PhaseMarginObjective(45), title="PhaseMarginObjective")
