using ModelingToolkit
using DyadControlSystems
using DyadControlSystems.MPC
import ModelingToolkitStandardLibrary
using ModelingToolkitStandardLibrary.Blocks
using Test
t = ModelingToolkitStandardLibrary.Blocks.t
A = [0 1; 0 0]
B = [0, 1]
C = [1 0]
D = 0
@named P = Blocks.StateSpace(A,B,C,D)
@named d = Integrator()

@variables u(t)=0
@named Pe = ODESystem([
    u + d.y ~ P.input.u[1],
], t, systems = [P, d])


# u = Pe.P.u
w = d.u
y = P.output.u
z = P.output.u

cP = complete(Pe)
op = Dict(cP.d.u => 0)
matrices, ssys = ModelingToolkit.linearize(Pe, [w; u], [z; y]; op)
Pelin = ss(matrices...)

G = partition(Pelin, u=2, y=2)
@test sminreal(system_mapping(G)) == ss(A,B,C,D)
@test sminreal(noise_mapping(G)).nx == 3 # the integrator affects the plant input and thus everything
@test poles(G) == [0,0,0] # double integrator + integrator disturbance


Pd = c2d(ss(A,B,C,D), 1)
lfsys = FunctionSystem(Pd)
@test lfsys.Ts == Pd.Ts
x = randn(2)
u = randn(1)
@test lfsys(x,u,0,0) == Pd.A*x + Pd.B*u
@test lfsys.measurement(x,u,0,0) == Pd.C*x + Pd.D*u

## Go from ODESystem to function system

function cartpole(x, u, p=nothing, t=nothing)
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81

    q  = x[[1, 2]]
    qd = x[[3, 4]]

    s = sin(q[2])
    c = cos(q[2])

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = [0 -mp*qd[2]*l*s; 0 0]
    G = [0, mp * g * l * s]
    B = [1, 0]

    qdd = -H \ (C * qd + G - B * u[1])
    return [qd; qdd]
end

function cartpolesys(; name)
    @variables t q(t)=0 qd(t)=0 x(t)=0 v(t)=0 u(t)=0 [input=true] y(t)=0 [output=true]
    D = Differential(t)
    mc, mp, l, g = 1.0, 0.2, 0.5, 9.81

    s = sin(q)
    c = cos(q)
    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = [0 -mp*qd*l*s; 0 0]
    G = [0, mp * g * l * s]
    B = [1, 0]

    rhs = -H \ (C * [v; qd] + G - B * u)

    eqs = [
        D(x) ~ v
        D(q) ~ qd
        D(v) ~ rhs[1]
        D(qd) ~ rhs[2]
        y ~ x
    ]
    ODESystem(eqs, t; name)
end

@named sys = cartpolesys()

@unpack u,y = sys
funsys = DyadControlSystems.build_controlled_dynamics(sys, u, y)
dyn, meas, x_order = funsys.dynamics, funsys.measurement, funsys.x
@test funsys.nx == 4
@test funsys.na == 0

e_manual = eigvals(MPC.linearize(cartpole, zeros(4), zeros(1), 0, 0)[1])
e_codegen = eigvals(MPC.linearize(dyn, zeros(4), zeros(1), 0, 0)[1])
@test sort(e_manual, by=imag) ≈ sort(e_codegen, by=imag)

B_manual = MPC.linearize(cartpole, zeros(4), zeros(1), 0, 0)[2]
B_codegen = MPC.linearize(dyn, zeros(4), zeros(1), 0, 0)[2]

@test all(<(1e-12), minimum(abs, B_codegen .- B_manual', dims=1)) # insensitive to state order


# The performance of the code generated version is a bit worse, likely due to it not using CSE atm., perhaps failing to optimize the repeated use of sin/cos
# @btime $(dyn)($xt, $ut, 0, 0) # 74.962 ns (1 allocation: 96 bytes)
# @btime cartpole($xt, $ut, 0, 0) # 15.882 ns (0 allocations: 0 bytes)


## Test that DAE systems are handled correctly
# @parameters t
# @variables x(t)[1:3]=0 
# @variables u(t)[1:2] #[input=true]
# D = Differential(t)
# y₁, y₂, y₃ = x
# u1, u2 = u
# k₁, k₂, k₃ = 1,1,1
# eqs = [
#     D(y₁) ~ -k₁*y₁ + k₃*y₂*y₃ + u1
#     D(y₂) ~ k₁*y₁ - k₃*y₂*y₃ - k₂*y₂^2 + u2
#     y₁ + y₂ + y₃ ~ 1
# ]

# @named sys = ODESystem(eqs, t)

# inputs = [u[1], u[2]]
# outputs = [y₂]
# linearize(sys, inputs, outputs)


# funsys = FunctionSystem(sys, inputs, outputs)


@parameters t
@variables x(t)[1:3]=[0,0,1]
@variables u1(t)=0 u2(t)=0 #[input=true]
D = Differential(t)
y₁, y₂, y₃ = x
k₁, k₂, k₃ = 1,1,1
eqs = [
    D(y₁) ~ -k₁*y₁ + k₃*y₂*y₃ + u1
    D(y₂) ~ k₁*y₁ - k₃*y₂*y₃ - k₂*y₂^2 + u2
    y₁ + y₂ + y₃ ~ 1
]

@named sys = ODESystem(eqs, t)

inputs = [u1, u2]
outputs = [y₁, y₂, y₃]
linearize(sys, inputs, outputs)

funsys = FunctionSystem(sys, inputs, outputs)

@test funsys.nx == 2
@test funsys.na == 0