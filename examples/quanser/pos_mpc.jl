cd(@__DIR__)
using Pkg
Pkg.activate(".")
using QuanserInterface, HardwareAbstractions
using DyadControlSystems.MPC
using ControlSystemsBase, LowLevelParticleFilters, StaticArrays, Plots
using Distributions: MvNormal
using RobustAndOptimalControl

Ts = 0.005

u0 = zeros(1)
x0 = zeros(4)
y0 = zeros(2)
op = OperatingPoint(x0, u0, y0)

xr = copy(x0)
ur = copy(u0)

# psim = QubeServoPendulumSimulator(; Ts, p = QuanserInterface.pendulum_parameters(true))
# function linearize_pendulum(xr)
#     A, B = ControlSystemsBase.linearize(psim.dynamics, xr, [0], psim.p, 0)
#     C, D = ControlSystemsBase.linearize(psim.measurement, xr, [0], psim.p, 0)
#     ss(A,B,C,D)
# end
# sys = linearize_pendulum(xr)

sys = let
    furutaA = [0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0; 0.0 -118.47675932202503 -2.8981578378981014 -0.008365985972588091; 0.0 -228.85975379059772 -2.698743448368527 -0.016160447845286145]
    furutaB = [0.0; 0.0; 45.260780705578064; 42.14650899959649;;]
    furutaC = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]
    furutaD = [0.0; 0.0;;]
    ss(furutaA, furutaB, furutaC, furutaD)
end

sysd = c2d(sys, Ts)

sysde = add_low_frequency_disturbance(sysd, sysd.B, ϵ = 1e-8)
x0e = [x0; 0]
ope = OperatingPoint(x0e, u0, y0)
xre = [xr; 0]

yr = sys.C * xr

umin = [-10]
umax = [10]

xmax = [Inf, 0.1, Inf, Inf]
xmin = -xmax

constraints = MPCConstraints(; umin, umax, xmin, xmax) 
# constraints = MPCConstraints(; umin, umax) 

solver = OSQPSolver(
    verbose = false,
    eps_rel = 1e-3,
    max_iter = 50000,
    check_termination = 5,
    polish = false,
)

x0sim = [-pi, 0, 0, 0, 0]
##
Q1 = Diagonal([10, 10, 0.001, 0.001]) # Penalties for the 4 controlled variables (not including the disturbance state)
Q2 = 0.1I(1)

R1 = kron(LowLevelParticleFilters.double_integrator_covariance(Ts, 1000), I(2)) + 1e-9I
R1 = cat(R1, 0.01, dims=(1,2)) # Add the disturbance state covariance
R2 = 2pi/2048 * diagm([0.1, 0.1])
kf = KalmanFilter(ssdata(sysde)..., R1, R2, MvNormal(x0e, R1))
N = 50

function rfun(t)
    if t < 7
        y1 = deg2rad(40*sign(sin(2pi/7*t)) + 40)
        [y1, 0, 0, 0]
    else
        y1 = deg2rad(45sin(2pi/5*t) + 20)
        dy1 = deg2rad(45*2pi/5*cos(2pi/5*t))
        [y1, 0, dy1, 0]
    end
    # [y1, 0]
end
tvec = range(0, step=Ts, length=N+1)
r = reduce(hcat, rfun.(tvec))

z = 1:4 # Indicate that we control the first 4 variables but not the disturbance state (state variable 5)
# z could also be specified as a matrix, in this case as I(5)[1:4, :]
predmodel = LinearMPCModel(sysde, kf; constraints, op=ope, x0=x0e, z)




prob = LQMPCProblem(predmodel; Q1, Q2, N, solver, r)
# prob.QN[:, end] .= 0
# prob.QN[end,:] .= 0


normalize_angles(x::Number) = mod(x+pi, 2pi)-pi

function run_control(process, prob; th=5, Tf = 20)
    Ts = process.Ts
    yo = zeros(2)
    dyf = zeros(2)
    initialize(process)
    N = round(Int, Tf/Ts)
    data = Vector{Vector{Float64}}(undef, 0)
    sizehint!(data, N)
    p = nothing
    t_start = time()

    simulation = process isa QubeServoPendulumSimulator
    simulation || QuanserInterface.go_home(process)

    @info "Starting"
    sleep(0.5)

    u = zeros(1)
    try
        GC.enable(false)
        for i = 1:N
            @periodically Ts simulation begin
                t = simulation ? (i-1)*Ts : time() - t_start
                y = QuanserInterface.measure(process)
                dy = @. (y - yo) / Ts
                yo = y
                tvec = range(0, step=Ts, length=prob.N+1) .+ t
                r = reduce(hcat, rfun.(tvec))
                @. dyf = 0.8dyf + 0.2dy
                
                y = normalize_angles.(y)

                observerinput = ObserverInput(; u, y, r)
                t_mpc = @elapsed co = MPC.step!(prob, observerinput, p, t; verbose=false)
                u = co.u[:, 1]
                control(process, u) 
                push!(data, [t; r[1]; y; u; t_mpc])
            end
        end
    catch e
        @error "Terminating" e
        rethrow()
    finally
        control(process, [0.0])
        GC.enable(true)
    end
    reduce(hcat, data)
end


process = QubeServoPendulum(; Ts)

##
sprocess = QubeServoPendulumSimulator(; Ts, p = QuanserInterface.pendulum_parameters(true))
Tf = 11
th = 3

@assert process.Ts == Ts == prob.Ts 
@time D = run_control(sprocess, prob; th=5, Tf)
tvec = D[1, :]
@test std(D[1,:] - D[2,:]) <= 3.7 # Check that the reference is followed
@test count(.! (xmin[2] .<= D[4, :] .<= xmax[2])) < 250 # Check that the softly constrained pendulum angle mostly respects the constraint

plot(tvec, D[2:5, :]', layout=4, lab=["r" "y arm" "y pend" "u"], sp=[1 1 2 3])
hline!([xmax[1:2] -xmax[1:2]]', sp=[1 2], l=(:black, :dash), primary=false)
plot!(ylims=(-0.5, 0.5), sp=2)
plot!(diff(tvec), lab="Δt", sp=4)
plot!(D[6, :], lab="t_mpc", sp=4); hline!([Ts], lab="Ts", framestyle=:zerolines, sp=4)


