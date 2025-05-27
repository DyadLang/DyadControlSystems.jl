using DyadControlSystems, ControlSystems, RobustAndOptimalControl, LinearAlgebra
G = ssrand(3,2,4, proper=true)
Q1 = diagm([1.0,2,3,4])
Q2 = diagm([5.0,6])

## Discrete
Gd = c2d(G, 0.01)
L = lqr(Gd, Q1, Q2)

Q,R,S,P,L2 = inverse_lqr(LMI(Q2), Gd, L)
@test L2 ≈ L
@test norm(L2-L) / norm(L) < 0.1
@test norm(S) < 1e-5

Q,R,S,P,L2 = inverse_lqr(Eq29(Q2), Gd, L)
@test L2 ≈ L

Q,R,S,P,L2 = inverse_lqr(Eq32(100), Gd, L)
@test L2 ≈ L


## Continuous
L = lqr(G, Q1, Q2)

Q,R,S,P,L2 = inverse_lqr(LMI(Q2), G, L)
@test L2 ≈ L

Q,R,S,P,L2 = inverse_lqr(Eq29(Q2), G, L)
@test L2 ≈ L


Q,R,S,P,L2 = inverse_lqr(Eq32(100), G, L)
@test L2 ≈ L

## Continuous without providing Q2
L = lqr(G, Q1, Q2)

Q,R,S,P,L2 = inverse_lqr(LMI(), G, L)
@test L2 ≈ L

Q,R,S,P,L2 = inverse_lqr(Eq29(), G, L)
@test L2 ≈ L


## Glover-McFarlane method
using DyadControlSystems, RobustAndOptimalControl
disc = (x) -> c2d(ss(x), 0.01)

G = tf([1, 5], [1, 2, 10]) |> disc
W1 = tf(1,[1, 0]) |> disc

gmf = (K,_,info) = glover_mcfarlane(G; W1)

method = DyadControlSystems.GMF(gmf)
Q,R,S,P,L2 = inverse_lqr(method)
@test L2 ≈ info.F # test that the generated cost function yields the same feedback gain as info.F which was generated my glover_mcfarlane

L3 = lqr(G*W1, Q, R)
@test L3 ≈ -L2

# Test strictly proper case
gmf = (K,_,info) = glover_mcfarlane(G; W1, strictly_proper=true)
method = DyadControlSystems.GMF(gmf)
Qsp,Rsp,Ssp,Psp,L2sp = inverse_lqr(method)

@test norm(L2 - L2sp)/norm(L2) < 0.05 # TODO: needs a better test.

Ko = observer_controller(info)
@test Ko.C ≈ info.F

