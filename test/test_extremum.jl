using ModelingToolkit, DyadControlSystems, OrdinaryDiffEq, Statistics, Test
@parameters t
D = Differential(t)

# Example 2 from the reference in the docstring
function testmodel(; name=:testmodel)
    @variables x1(t)=0 x2(t)=0 u(t) y(t)
    eqs = [
        D(x1) ~ x1^2 + x2 + u
        D(x2) ~ -x2 + x1^2
        y ~ -1 - x1 + x1^2
    ]
    ODESystem(eqs, t; name)
end

model = testmodel()
@named escontroller = PIESC(k=10, tau=0.1, a=10, w=100, wh=1000)

connections = [
    model.y ~ escontroller.y
    escontroller.u ~ model.u
]

@named closed_loop = ODESystem(connections, t, systems=[model, escontroller])
closed_loop = complete(closed_loop)
sys = structural_simplify(closed_loop)

x0 = [
    # model.x1 => 0.5
    # model.x2 => 0.25
    # esc.uh => -0.5
    closed_loop.escontroller.v => 0
]

prob = ODEProblem(sys, x0, (0, 10.0))

sol = solve(prob, Rodas5())

# plot(sol, idxs=[model.x1, model.x2, model.y, esc.u, esc.uh], layout=5)
# display(current())

@test mean(sol[closed_loop.escontroller.uh][end-200:end]) ≈ -0.5 rtol=0.1
@test mean(sol[model.y][end-200:end]) ≈ -1.25 rtol=0.1
@test mean(sol[model.x1][end-200:end]) ≈ 0.5 rtol=0.1
@test mean(sol[model.x2][end-200:end]) ≈ 0.255 rtol=0.1



## Vanilla
@named escontroller = ESC(k=0.8, a=0.3, b=0.3, w=3, wh=1)

func(x) = -3x - 2x^2 + x^3 + x^4
@variables th(t)
connections = [
    escontroller.y ~ func(escontroller.u)
]

@named closed_loop = ODESystem(connections, t, systems=[escontroller])

sys = structural_simplify(closed_loop)

x0 = []

prob = ODEProblem(sys, x0, (0, 300.0))

sol = solve(prob, Rodas5())

# plot(sol, idxs=[escontroller.u, escontroller.uh, escontroller.y])
# display(current())

@test mean(sol[escontroller.uh][end-200:end]) ≈ 1 rtol=0.1
@test mean(sol[escontroller.y][end-200:end]) < -2.5
.1

## theta0 = -1.5
x0 = [
    escontroller.uh => -1.5
]
prob = ODEProblem(sys, x0, (0, 400.0))

sol = solve(prob, Rodas5())

# plot(sol, idxs=[escontroller.u, escontroller.uh, escontroller.y])
# display(current())

@test mean(sol[escontroller.uh][end-200:end]) ≈ 1 rtol=0.1
@test mean(sol[escontroller.y][end-200:end]) < -2.5

##
