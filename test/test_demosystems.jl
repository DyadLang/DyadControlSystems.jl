using DyadControlSystems, ModelingToolkit
using DyadControlSystems.MPC
import DyadControlSystems.ControlDemoSystems as demo
using StaticArrays

rcam = demo.rcam()
@test length(unknowns(rcam.rcam_model)) >= 9
@test length(unknowns(rcam.iosys)) == 9
@test Set(ModelingToolkit.inputs(rcam.rcam_model)) == Set(rcam.inputs)



sys = demo.dcmotor()



# ==============================================================================
## Furuta
# ==============================================================================
using DyadControlSystems
import DyadControlSystems.ControlDemoSystems as demo
using OrdinaryDiffEq

normalize_angles(x::Number) = mod(x + pi, 2pi) - pi
normalize_angles(x::AbstractVector) = SA[normalize_angles(x[1]), x[2], normalize_angles(x[3]), x[4]]

function controller(x, (p, L))
    θ, θ̇, ϕ, ϕ̇ = x
    ks = 100
    th = 2

    if abs(θ) < 0.2 # Use stabilizing controller
        -L*normalize_angles(x)
    else
        E = demo.furuta_energy(x, p)
        SA[clamp(ks*E*sign(θ̇*cos(θ)), -th, th)]
    end
end

function furuta_cl(x, (p, L), t)
    u = controller(x, (p, L))
    demo.furuta(x, u, p, t)
end


# Lund pendulum parameters
l = 0.413
M = 0.01
Jp = 0.0009
r = 0.235
J = 0.05
m = 0.02
g = 9.82
p = (M, l, r, J, Jp, m, g)

# QubeServo parameters
# m = 0.024
# mr = 0.095
# l = 0.129
# M = 0.005
# r = 0.085
# Jp = mr*r^2/3
# J = m*l^2/3
# g = 9.82
# p = (M, l, r, J, Jp, m, g)

Afu, Bfu = DyadControlSystems.linearize(demo.furuta, zeros(4), [0], p, 0)
Q1 = Diagonal([1, 1, 1e-6, 1])
Q2 = 100
L = lqr(Continuous, Afu, Bfu, Q1, Q2)
L = SMatrix{size(L)...}(L)

x0 = SA[0.99pi, 0.0, 0, 0.0]
# @btime furuta_cl($x0, ($p, $L), 0)

tspan = (0.0, 10.0)
prob = ODEProblem(furuta_cl, x0, tspan, (p, L))
sol = solve(prob, Tsit5())
f1 = plot(sol, layout=4, lab=["\$θ\$" "\$\\dot θ\$" "\$ϕ\$" "\$\\dot ϕ\$"], plot_title="Furuta pendulum swing-up")

time = range(tspan..., length=300)
u = map(time) do t
    controller(sol(t), (p, L))
end
U = reduce(hcat, u)'
f2 = plot(time, U, lab="\$u\$")
plot(f1, f2)


##
@test sol.u[end][1] ≈ 0 atol=1e-2
@test sol.u[end][2] ≈ 0 atol=1e-2
# We don't care about the arm angle
@test sol.u[end][4] ≈ 0 atol=1e-2
