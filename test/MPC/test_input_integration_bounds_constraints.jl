using DyadControlSystems
using DyadControlSystems.MPC


function linsys(x, u, p, _)
    Ad = [1.0;;]
    B = [1 1]
    Ad*x + B*u
end

Ts = 1
x0 = [10.0]
xr = [0.0]
running_cost = StageCost() do si, p, t
    e = (si.x-si.r)[1]
    dot(e, e)
end
terminal_cost = TerminalCost() do ti, p, t
    e = (ti.x-ti.r)[1]
    10dot(e, e)
end

bounds_constraint = BoundsConstraint(umin = [-1, -Inf], umax = [1, Inf], xmin = [-0.7], xmax = [Inf], dumin = [-Inf, -0.1], dumax = [Inf, 0.15])

objective = Objective(running_cost, terminal_cost)
p = 1.0
N = 10


## Without input integrators
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x, u=:u^2, y=:y)

observer = StateFeedback(dynamics, x0)

prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    xr,
    presolve = true,
    verbose = isinteractive(),
);

# We do expect difference constraint since we passed dumin
@test any(c->c isa DifferenceConstraint, prob.constraints.constraints)

X0, U0 = get_xu(prob)
@test all(>(-1.001), U0[1,:])
@test all(<(1.001), U0[1,:])

@test all(>(-0.101), diff(U0[2,:]))
@test all(<(0.151), diff(U0[2,:]))


@time hist = MPC.solve(prob; x0, T=20, verbose = false);


# @test abs(hist.X[end][]) < 1e-3
# @test abs(hist.U[1][]) > 3

isinteractive() && plot(hist)

X,E,R,U,Y,UE = reduce(hcat, hist)

@test all(>(-1.001), U[1,:])
@test all(<(1.001), U[1,:])

@test all(>(-0.101), diff(U[2,:]))
@test all(<(0.151), diff(U[2,:]))


## With input integrator on the first input ====================================
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x, u=:u^2, y=:y, input_integrators=1:1)

observer = StateFeedback(dynamics, x0)

prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    xr,
    presolve = true,
    verbose = isinteractive(),
);

# We do expect both stage and difference constraint since we passed 
@test any(c->c isa DifferenceConstraint, prob.constraints.constraints)
@test any(c->c isa StageConstraint, prob.constraints.constraints)

X0, U0 = get_xu(prob)
@test all(>(-1.001), U0[1,:])
@test all(<(1.001), U0[1,:])

@test all(>(-0.101), diff(U0[2,:]))
@test all(<(0.151), diff(U0[2,:]))


@time hist = MPC.solve(prob; x0, T=20, verbose = false);


# @test abs(hist.X[end][]) < 1e-3
# @test abs(hist.U[1][]) > 3

isinteractive() && plot(hist)

X,E,R,U,Y,UE = reduce(hcat, hist)

@test all(>(-1.001), U[1,:])
@test all(<(1.001), U[1,:])

@test all(>(-0.101), diff(U[2,:]))
@test all(<(0.151), diff(U[2,:]))


## With input integrator on the second input ===================================
dynamics = FunctionSystem(linsys, (x,u,p,t)->x, Ts, x=:x, u=:u^2, y=:y, input_integrators=2:2)

observer = StateFeedback(dynamics, x0)

prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    xr,
    presolve = true,
    verbose = isinteractive(),
);

# We do expect difference constraint since we passed dumin
@test !any(c->c isa DifferenceConstraint, prob.constraints.constraints)
@test !any(c->c isa StageConstraint, prob.constraints.constraints)

X0, U0 = get_xu(prob)
@test all(>(-1.001), U0[1,:])
@test all(<(1.001), U0[1,:])

@test all(>(-0.101), diff(U0[2,:]))
@test all(<(0.151), diff(U0[2,:]))


@time hist = MPC.solve(prob; x0, T=20, verbose = false);


# @test abs(hist.X[end][]) < 1e-3
# @test abs(hist.U[1][]) > 3

isinteractive() && plot(hist)

X,E,R,U,Y,UE = reduce(hcat, hist)

@test all(>(-1.001), U[1,:])
@test all(<(1.001), U[1,:])

@test all(>(-0.101), diff(U[2,:]))
@test all(<(0.151), diff(U[2,:]))
