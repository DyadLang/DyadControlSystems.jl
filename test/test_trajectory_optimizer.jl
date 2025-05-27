using ControlSystemsBase.Polynomials
using Test
using DyadControlSystems
Ta = 1
x = [0;0;0;0]
xwp = [1 1 -1 0]
vel = 0
px,py,pz,pyaw = optimal_trajectory_gen(Ta,x,xwp,vel)

xp = Polynomial(px[:,1])
yp = Polynomial(py[:,1])
zp = Polynomial(pz[:,1])
yawp = Polynomial(pyaw[:,1])
xd0 = [xp(0);yp(0);zp(0);yawp(0)]
@test isapprox(xd0,x;atol = 1e-3)

xdta = [xp(Ta) yp(Ta) zp(Ta) yawp(Ta)] 
@test isapprox(xdta,xwp;atol = 1e-3)

# Wrapper for polynomial trajectory
gen_optimal_trajectory(t) = time_polynomial_trajectory(t, px, py, pz, pyaw, Ta)

xdes,b1des = gen_optimal_trajectory(0.0)
@test isapprox(xdes[1:4],xd0;atol = 1e-2)
