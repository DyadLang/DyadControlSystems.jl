using DyadControlSystems
using DyadControlSystems.MPC
using Plots, StaticArrays

d_(t) = 2 + 2sin(3t) + sin(5t)

"mass-spring-damper with disturbance"
function msd_dyn(x, u, p, t)
    m,b,k = 2,5,2
    q, qd = x
    SA[
        qd,
        (-b*qd + -k*q + u[] + d_(t))/m
    ]
end

yr(t) = 1sin(2t)
ydr(t) = 2cos(2t)

function σ(x, p, t)
    q, qd = x
    e = q - yr(t)
    ė = qd - ydr(t)
    ė + e
end

"computes x - xr"
function err(x, p, t)
    q, qd = x
    e = q - yr(t)
    ė = qd - ydr(t)
    [e, ė]
end

Ts = 0.01 # Discrete sample time
smc_standard = SlidingModeController(σ, (s,t) -> -20*sign(s)) |> Stateful
smc_smooth = SlidingModeController(σ, (s,t) -> -20*tanh(10s)) |> Stateful
smc_supertwist = SuperTwistingSMC(50, σ, Ts) |> Stateful


sysc = ss(ControlSystemsBase.linearize(msd_dyn, [0,0], [0], 0, 0)..., [0 1], 0)
sys = c2d(sysc, Ts)

# G = [-0.990049833749168, 1]
desired_poles = [0.991] # Only for the subsystem with size nx-nm

k, ks = 1, 10
smc_linear = DyadControlSystems.LinearSlidingModeController(; k, ks, sys, err, desired_poles) |> Stateful



function simulate_smc(; T, Ts, smc::Stateful, dyn, x0, p = 0)
    X = Float64[]
    U = Float64[]
    R = Float64[]
    x = x0
    smc.x = 0 # Reset the state of the stateful controller

    for i = 0:round(Int, T/Ts)
        t = i*Ts
        q, qd = x
        ui = smc(x, p, t)

        push!(X, q)
        push!(R, yr(t))
        push!(U, ui)
        x = dyn(x, ui, p, t)
    end
    t = range(0, step=Ts, length=length(X))
    X, U, R, t
end

T  = 10   # Simulation duration
ddyn = rk4(msd_dyn, Ts; supersample=2) # Discretized dynamics using RK4
x0   = SA[0.0, 0.0] # Initial state of the system (not including any state in the controller)

#

X, U, R, t = simulate_smc(; T, Ts, smc=smc_standard, dyn=ddyn, x0)
@test √(mean(abs2, (X-R)[end-100:end])) < 0.013
X, U, R, t = simulate_smc(; T, Ts, smc=smc_smooth, dyn=ddyn, x0)
@test √(mean(abs2, (X-R)[end-100:end])) < 0.013
X, U, R, t = simulate_smc(; T, Ts, smc=smc_supertwist, dyn=ddyn, x0)
@test √(mean(abs2, (X-R)[end-100:end])) < 0.001
X, U, R, t = simulate_smc(; T, Ts, smc=smc_linear, dyn=ddyn, x0)
@show √(mean(abs2, (X-R)[end-100:end])) # can reach 0.0021424202206596268
@test √(mean(abs2, (X-R)[end-100:end])) < 0.0022

if isinteractive()
    function simulate_and_plot(smc)
        X, U, R, t = simulate_smc(; T, Ts, smc, dyn=ddyn, x0)
        plot(t, [X U], layout=(1,3), lab=["x" "u"], ylims=[(-1.1, 1.1) (-21, 21)], sp=[1 3])
        plot!(t, R, sp=1, lab="r")
        plot!(t, X-R, lab="e", sp=2, ylims=(-0.3, 0.1))
    end

    fig1 = simulate_and_plot(smc_standard)
    plot!(fig1, plot_title="Standard SMC")

    fig2 = simulate_and_plot(smc_smooth)
    plot!(fig2, plot_title="Smooth SMC")

    fig3 = simulate_and_plot(smc_supertwist)
    plot!(fig3, plot_title="SuperTwisting SMC")

    fig4 = simulate_and_plot(smc_linear)
    plot!(fig4, plot_title="Linear SMC")

    plot(fig1, fig2, fig3, fig4, layout=(4, 1), size=(800, 600))
end

## DMM
P = DemoSystems.double_mass_model(outputs=3)
Ts = 0.002 # Discrete sample time
function dmm_dyn(x, u, p, t)
    P.A*x + vec(P.B*(u + d_(t)))
end

sysd = c2d(P, Ts)


function simulate_smc(; T, Ts, smc::Stateful, dyn, x0, p = 0)
    X = Float64[]
    U = Float64[]
    R = Float64[]
    x = x0
    smc.x = 0 # Reset the state of the stateful controller

    for i = 0:round(Int, T/Ts)
        t = i*Ts
        q, qd = x[3:4]
        ui = smc(x, p, t)

        push!(X, q)
        push!(R, yr(t))
        push!(U, ui)
        x = dyn(x, ui, p, t)
    end
    t = range(0, step=Ts, length=length(X))
    X, U, R, t
end

function simulate_and_plot(smc)
    X, U, R, t = simulate_smc(; T, Ts, smc, dyn=ddyn, x0)
    plot(t, [X U], layout=(1,3), lab=["x" "u"], ylims=[(-1.1, 1.1) (-31, 31)], sp=[1 3])
    plot!(t, R, sp=1, lab="r")
    plot!(t, X-R, lab="e", ylims=(-0.2, 0.1), sp=2)
end


# TODO: change the sliding surface to correspond to a pefectly damped double-mass model, i.e., the reference generator in the LQG case.

function σ(x, p, t)
    q, qd = x[3:4]
    qdd = P.A[4,:]'x
    e = q   - yr(t)
    ė = qd  - ydr(t)
    ë = qdd - yddr(t)
    ë + 2ė + e
end

function err(x, p, t)
    # q, qd = x[3:4]
    # e = q   - yr(t)
    # ė = qd  - ydr(t)
    # [e, ė, e, ė]
    x
end

desired_poles = fill(0.968, 3)
# desired_poles = fill(0.95, 3)


# test only disturbance rejection since we would use computed torque in a realistic scenario
d_(t) = 10(2sin(3t) + sin(5t))
yr(t) = 0*(1sin(2t))
ydr(t) = 0*(2cos(2t))
yddr(t) = 0*(-4sin(2t))


T  = 20   # Simulation duration
ddyn = rk4(dmm_dyn, Ts; supersample=2) # Discretized dynamics using RK4
x0   = SA[0.0, 0.0, 0.0, 0.0] # Initial state of the system (not including any state in the controller)


smc_standard = SlidingModeController(σ, (s,t) -> -30*sign(s)) |> Stateful
smc_smooth = SlidingModeController(σ, (s,t) -> -30*tanh(30s)) |> Stateful
smc_supertwist = SuperTwistingSMC(100, σ, Ts, 1.1) |> Stateful
smc_linear = LinearSlidingModeController(; k=50, ks=2, sys=sysd, err, desired_poles) |> Stateful


# Tweak the inner feedback gain to use LQR instead of pole placement. The result is very sensitive to both the pole placement and the LQR cost. In general, it seems like it's enough to penalize the last state only, to be decided if this holds only for the particular coordinate transform that was chosen for this system or if this holds in general.
Q = diagm([0,0,100])
R = 1I(1)
L = lqr(smc_linear.controller.sysr, Q, R)
smc_linear.controller.Λ[eachindex(L)] .= L[eachindex(L)]



X, U, R, t = simulate_smc(; T, Ts, smc=smc_standard, dyn=ddyn, x0)
@test √(mean(abs2, (X-R)[end-100:end])) < 0.0025
X, U, R, t = simulate_smc(; T, Ts, smc=smc_smooth, dyn=ddyn, x0)
@test √(mean(abs2, (X-R)[end-100:end])) < 0.0025
X, U, R, t = simulate_smc(; T, Ts, smc=smc_supertwist, dyn=ddyn, x0)
@test √(mean(abs2, (X-R)[end-100:end])) < 0.001

X, U, R, t = simulate_smc(; T, Ts, smc=smc_linear, dyn=ddyn, x0)
@show √(mean(abs2, (X-R)[end-100:end]))
@test √(mean(abs2, (X-R)[end-100:end])) < 0.0017

# plot(
#     simulate_and_plot(smc_standard),
#     simulate_and_plot(smc_smooth),
#     simulate_and_plot(smc_supertwist),
#     simulate_and_plot(smc_linear),
#     layout=(4,1),
#     size=(800, 1200)
# )


# ##
# poles(P)

# desired_poles = [-14.15, -14.15, -2, -2]
# Pdes = tf(zpk([], desired_poles, 1))
# denvec(Pdes)[]