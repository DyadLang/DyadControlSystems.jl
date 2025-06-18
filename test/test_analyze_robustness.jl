using DyadControlSystems
using Plots
##
P = DemoSystems.double_mass_model(); 
C = pid(1,5,1, Tf=0.3, state_space=true)
# dm = diskmargin(P)
# gangoffourplot(P, ss(I(P.ny)), title="", xlabel="")

default(size=(1200, 1000))

analyze_robustness(P)
analyze_robustness(P, C)