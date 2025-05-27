using Distributed
using ProgressLogging
using QuanserInterface
using HardwareAbstractions
using SciMLBase
using ControlSystemsBase
# addprocs(3)
@everywhere using ParallelDataTransfer
@everywhere using Serialization
@everywhere using Dates

@everywhere include("setup_mpc.jl")


## MPC controller




##
MPC.MOI.set(solver, MPC.MOI.RawOptimizerAttribute("max_wall_time"), 15.0)
# MPC.MOI.set(solver, MPC.MOI.RawOptimizerAttribute("print_level"), 1)
# MPC.MOI.set(solver, MPC.MOI.RawOptimizerAttribute("mu_init"), 1e-6)
# MPC.MOI.set(solver, MPC.MOI.RawOptimizerAttribute("mu_strategy"), "adaptive")
# MPC.MOI.set(solver, MPC.MOI.RawOptimizerAttribute("tol"), 100.0)
# MPC.MOI.set(solver, MPC.MOI.RawOptimizerAttribute("compl_inf_tol"), 1e-2)
# @time hist = MPC.solve(prob; x0, T = 100, verbose = true, noise=false, dyn_actual=discrete_dynamics);
# plot(hist)

# ==============================================================================
## Closed-loop simulation
# ==============================================================================
# measure = QuanserInterface.measure

@everywhere normalize_angles_pi(x::Number) = mod(x+pi, pi)-pi
@everywhere normalize_angles(x::Number) = mod(x, 2pi)
@everywhere normalize_angles(x::AbstractVector) = SA[(x[1]), normalize_angles(x[2]), x[3], x[4]]

function mpc_experiment(process, psim, prob, observer; verbose, Tf = 10)
    simulation = processtype(process) isa SimulatedProcess
    initialize(process)
    Ts_fast = process.Ts
    Ts = prob.Ts
    mpc_interval = round(Int, Ts/Ts_fast)
    Nsteps = round(Int, Tf/Ts_fast)
    data = Vector{Vector{Float64}}(undef, 0)
    sizehint!(data, Nsteps)

    # QuanserInterface.go_home(process, r = 0, Ki=0.1)
    simulation || sleep(1) # Wait for the arm to settle
    y_start = QuanserInterface.measure(process)
    x0 = [y_start; 0; 0] # start from hanging down
    MPC.reset_observer!(observer, x0)
    w = [] 



    # if !simulation
    #     @info "Pre solving from current position"
    #     MPC.MOI.set(solver, MPC.MOI.RawOptimizerAttribute("max_wall_time"), 30.0)
    #     MPC.MOI.set(solver, MPC.MOI.RawOptimizerAttribute("print_level"), 5)
    #     r = MPC.get_first(prob.xr)
    #     observerinput = MPC.ObserverInput([0.0],y_start,r,w)
    #     prob.vars.vars .= 0.01 .* randn.()
    #     co = MPC.step!(prob, observerinput, psim.p, 0.0; verbose)
    #     MPC.MOI.set(solver, MPC.MOI.RawOptimizerAttribute("max_wall_time"), 0.95Ts)
    #     MPC.MOI.set(solver, MPC.MOI.RawOptimizerAttribute("print_level"), 4)

    #     x_mpc,u_mpc = get_xu(prob)
    #     plot(
    #         plot(x_mpc', layout=4, title=permutedims(state_names(dynamics))),
    #         plot(u_mpc'),
    #     ) |> display
    # end

    if verbose
        @info "Starting $(simulation ? "simulation" : "experiment") from $(x0), waiting for your input..."
        readline()
    end

    try
        GC.gc()
        GC.enable(false)
        t_start = time()
        u = [0.0]
        mpc_counter = 0
        for i = 1:Nsteps
            @periodically Ts_fast simulation begin 
                t = time() - t_start
                y = measure(process)
                if !(-deg2rad(120) <= y[1] <= deg2rad(120))
                    @error "Out of bounds, terminating"
                    control(process, [0.0])
                    break
                end
                r = MPC.get_first(prob.xr)
                # observerinput = MPC.ObserverInput(u,y,r,w)
                llpf.correct!(observer, u, y, psim.p, t; R2 = observer.R2)
                xh = copy(observer.x)
                if abs(normalize_angles(y[2]) - pi) < 0.4
                    # Use stabilizing controller
                    xe = (r - xh)
                    mul!(u, QuanserInterface.Lup, xe)
                else
                    # Use reference tracking controller
                    r_arm = 0.0 # TODO: use MPC reference traj
                    e = r_arm - y[1]
                    u[1] = 2*e
                end
                @. u = clamp(u, -8, 8)
                control(process, u)

                if mpc_counter == mpc_interval
                    # co = MPC.step!(prob, observerinput, psim.p, t; verbose)
                    # @views u .= co.u[:, 1] # Access first control signal
                    mpc_counter = 0
                else
                    mpc_counter += 1
                end

                verbose && @info "t = $t, u = $(u[])"
                llpf.predict!(observer, u, psim.p, t)
                log = [t; y; xh; u]
                push!(data, log)
            end
        end
    catch e
        @error "Shit hit the fan" e
        rethrow()
    finally
        control(process, [0.0])
        GC.enable(true)
        GC.gc()
    end

    D = reduce(hcat, data)
end


# D = mpc_experiment(process, psim, prob, kf; verbose=true, Tf = 16)
# plotD(D)

##
# using DataInterpolations
# function open_loop_swingup(process, prob; verbose=true)
#     Ts_fast = process.Ts
#     Ts = prob.Ts
#     mpc_interval = round(Int, Ts/Ts_fast)
#     data = Vector{Vector{Float64}}(undef, 0)
#     xtraj, utraj = get_xu(prob)
    
#     trange = range(0, step=Ts, length=size(utraj, 2))
#     sizehint!(data, length(trange))
#     # xtvec = vec(trange' .+ Ts*prob.constraints.constraints[2].taupoints)

#     QuanserInterface.go_home(process, r = 0, K = 0.1, Ki=0.5)
#     ufun = LinearInterpolation(vec(utraj), trange)
#     try
#         GC.gc()
#         GC.enable(false)
#         t_start = time()
#         u = [0.0]
#         mpc_counter = 0
#         for i = 1:length(trange)
#             @periodically Ts_fast begin 
#                 t = time() - t_start
#                 y = QuanserInterface.measure(process)
#                 if !(-deg2rad(120) <= y[1] <= deg2rad(120))
#                     @error "Out of bounds, terminating"
#                     control(process, [0.0])
#                     break
#                 end
#                 u = [ufun(t)]
#                 control(process, u)
#                 verbose && @info "t = $t, u = $(u[])"
#                 log = [t; y; zeros(4); u]
#                 push!(data, log)
#             end
#         end
#     catch e
#         @error "Shit hit the fan" e
#         rethrow()
#     finally
#         control(process, [0.0])
#         GC.enable(true)
#         GC.gc()
#     end

#     D = reduce(hcat, data)
# end

# open_loop_swingup(process, prob)

# ==============================================================================
## Collect training data for explicit MPC
# ==============================================================================

function serializecopy(thing)
    name = gensym()
    path = "/tmp/name"
    serialize(path, thing)
    thingcopy = deserialize(path)
    rm(path)
    thingcopy
end

function collect_optimal_inputs(prob, n; xmin, xmax, method = :random)
    (; nx, nu) = prob

    if method === :grid
        nd = ceil(Int, n^(1/nx))
        ranges = [range(xmin[i], xmax[i], length=nd) for i = 1:nx]
        iterator = Iterators.product(ranges...)
    elseif method === :random

        ranges = [range(xmin[i], xmax[i], length=10_000) for i = 1:nx]
        iterator = Iterators.map(1:n) do i
            [rand(range) for range in ranges]
        end

    end
    
    # sendto(workers(); prob)
    # @passobj 1 workers() prob

    cd(@__DIR__)
    path = mkdir("data/data_$(Dates.now())")
    t = 0.0
    w = []

    write(joinpath(path, "Aprob.jl"), repr(prob))
    
    @progress for x in iterator
        x0 = [x...]
        controllerinput = MPC.ControllerInput(x0, prob.xr, w, zeros(nu)) # We should ideally pass in previous u here but we don't have it, so this will give an approximation to the correct u for the first data point, we could discard this point
        # MOI.set(prob.solver, MOI.RawOptimizerAttribute("warm_start_init_point"), "yes")
        controlleroutput = MPC.optimize!(prob, controllerinput, prob.p, t);
        if Int(controlleroutput.sol.retcode) != 1 # Try again if we fail
            prob.vars.vars .= 1.0 .* randn.();
            # xguess = reduce(hcat, [range(x0[i], r[i], length=2prob.N+1) for i = 1:nx])
            # prob.vars.vars[1:length(xguess)] .= xguess[:]
            # MOI.set(prob.solver, MOI.RawOptimizerAttribute("warm_start_init_point"), "no")
            controlleroutput = MPC.optimize!(prob, controllerinput, prob.p, t);
        end
        xopt, uopt = get_xu(prob)
        xopt = xopt[:, 1:2:end-1] # Only true when n_colloc = 2
        plot(
            plot(xopt', layout=4, title=permutedims(state_names(prob.dynamics))),
            plot(uopt', title="u"),
        ) |> display
        if Int(controlleroutput.sol.retcode) == 1
            serialize("$(path)/pend_$(Dates.now())", (xopt, uopt, controlleroutput.sol.retcode))
        else
            serialize("$(path)/failed_pend_$(Dates.now())_$(controlleroutput.sol.retcode)", (xopt, uopt, controlleroutput.sol.retcode))
        end
        if Int(controlleroutput.sol.retcode) != 1 # Try again if we fail
            prob.vars.vars .= 1.0 .* randn.();
        end
        sleep(0.001)
    end
end

# serialize("x0_opt",(xopt,uopt))
# (xopt,uopt) = deserialize("x0_opt")

using NearestNeighbors
struct KNNReg{T, UT, W}
    k::Int
    tree::T
    U::UT
    widths::W
end

knn_normalize(x, widths) = x ./ widths

function KNNReg(X, U, widths, k=3, kwargs...)
    all(isfinite, X) || @error "X contains non-finite values"
    all(isfinite, U) || @error "U contains non-finite values"
    all(isfinite, widths) || @error "widths contains non-finite values"
    kdtree = KDTree(knn_normalize(X, widths); kwargs...)
    KNNReg(k, kdtree, U, widths)
end

function predict(reg::KNNReg, x0, k=reg.k)
    x = knn_normalize(x0, reg.widths)
    idxs, dists = knn(reg.tree, x, k)
    @show dists
    @. dists = 1 ./ max(dists, 1e-6)
    dists ./= sum(dists)
    @views reg.U[:, idxs] * dists
end
# Test Knn regression
# N = 40
# nx = 1
# fun = sin
# lower = 0.0
# upper = 3*2*pi
# widths = upper .- lower
# X = rand(nx, N) .* widths .+ lower
# U = fun.(X)
# Xp = rand(nx, 10) .* widths .+ lower
# ##
# reg = KNNReg(X, U, widths, 2)
# Up = [predict(reg, Xp[:, i]) for i = 1:size(Xp, 2)]

# Up = reduce(hcat, Up)

# scatter(X', U', m=:o)
# scatter!(Xp', Up', m=:x)


##

# We use different bounds for the data collection since we want to make sure the problem is feasible when we solve it, but we don't want to solve it starting from everywhere
xmin = [-deg2rad(80), 0, -8, -20]
xmax = [deg2rad(80), 2pi, 8, 20]

# xmin = [-deg2rad(79), -deg2rad(175), -2, -4]
# xmax = [deg2rad(79), deg2rad(175), 2, 4]

collect_optimal_inputs(prob, 5000; xmin, xmax)
error()
## Load data

function load_data(path; failed = false)
    cd(@__DIR__)
    if path isa AbstractVector
        files = reduce(vcat, readdir.(path, join=true))
    else
        files = readdir(path, join=true)
    end
    if !failed
        filter!(x -> !occursin("failed", x) && !occursin("Aprob", x), files)
    end
    X = Matrix{Float64}[]
    U = Matrix{Float64}[]
    @progress for file in files
        (xopt, uopt, retcode) = deserialize("$file")
        xopt[2,:] .= normalize_angles.(xopt[2,:])
        push!(X, xopt)
        push!(U, uopt)
    end
    reduce(hcat, X), reduce(hcat, U)
end

# path = "data/data_2023-07-08T08:59_grid/"
path = [
    "data/data_2023-07-08T08:59_grid/"
    # "data/data_2023-07-08T14:46_rand_origin/"
    # "data/data_2023-07-08T14:32_rand_large/"
    "data/data_2023-07-08T17:36_N_100/"
    "data/data_2023-07-08T19:20_N100_slightly_wider/"
    "data/data_2023-07-09T07:48_N100_wide/"
    "data/data_2023-07-09T09:01:09.916/" # N100 Super wide 
]

X, U = load_data(path; failed=false)
importance = [1, 100.0, 10.0, 100.0]
widths = (xmax .- xmin) ./ importance
reg = KNNReg(X, U, widths, 5)

histogram2d(X[1,:], X[2,:], nbins = 51)
hline!([pi -pi]); vline!(deg2rad.([-10 10]))
histogram2d(X[3,:], X[4,:], nbins = 51)
# using StatsPlots
# corrplot(X[:, 1:10:end]')

using QuanserInterface: energy
function knn_control(process, reg, kf; Tf = 10, verbose=true, stab=true)
    # Ts = process.Ts
    Ts = 0.01
    N = round(Int, Tf/Ts)
    data = Vector{Vector{Float64}}(undef, 0)
    sizehint!(data, N)

    simulation = processtype(process) isa SimulatedProcess

    if simulation
        u0 = 0.0
    else
        u0 = 0.5QuanserInterface.go_home(process, r = 0, K = 0.05, Ki=0.2, Kf=0.02)
        @show u0
    end
    y = QuanserInterface.measure(process)
    if verbose && !simulation
        @info "Starting $(simulation ? "simulation" : "experiment") from xh = $(kf.x), y: $y, waiting for your input..."
        readline()
    end
    yo = zeros(2)
    dyf = zeros(2)

    # Friction compensation, Coulomb friciton is about 0.1 input units
    smoothsign(x) = tanh(5*x) 

    try
        GC.gc()
        GC.enable(false)
        t_start = time()
        u = [0.0]
        oob = 0
        for i = 1:N
            @periodically Ts simulation begin 
                t = simulation ? (i-1)*Ts : time() - t_start
                y = QuanserInterface.measure(process)
                llpf.correct!(kf, u, y, R2=0.01*kf.R2)
                xh = copy(kf.x)
                dy = (y - yo) ./ Ts
                @. dyf = 0.5dyf + 0.5dy
                xh = [y; dyf]
                xhn = [xh[1], normalize_angles(xh[2]), xh[3], xh[4]]
                if !(-deg2rad(110) <= y[1] <= deg2rad(110))
                    u = [-0.5*y[1]]
                    @warn "Correcting"
                    control(process, u .+ u0)
                    oob += 20
                    if oob > 600
                        @error "Out of bounds"
                        break
                    end
                else
                    oob = max(0, oob-1)
                    if stab && abs(normalize_angles(y[2]) - pi) < 0.3
                        L = [-7.410199310542298 -36.40730995983665 -2.0632501290782095 -3.149033572767301]
                        @info "stabilizing"
                        u = clamp.(L*(r - xhn), -10, 10)
                    else
                        # xhn = (process.x) # Try with correct state
                        θ = y[2] - pi
                        θ̇ = xh[4]
                        E = energy(θ, θ̇)
                        u = [clamp(80*(E - energy(0,0))*sign(θ̇*cos(θ)) - 0.2*y[1], -10, 10)]
                        # u = 0*predict(reg, xhn)
                    end
                    control(process, u .+ 0*u0 .+ 0.0*sign(xh[3]))
                end
                verbose && @info "t = $t, u = $(u[]), xh = $xh"
                log = [t; y; xh; u]
                push!(data, log)
                llpf.predict!(kf, u)
                yo = y
            end
        end
    catch e
        @error "Shit hit the fan" e
        # rethrow()
    finally
        control(process, [0.0])
        GC.enable(true)
        GC.gc()
    end

    D = reduce(hcat, data)
end
##
process = QuanserInterface.QubeServoPendulum(; Ts)
# home!(process, 0)
##
# sprocess = QuanserInterface.QubeServoPendulumSimulator(; Ts)
function runplot()
    home_pend!(process)
    llpf.reset!(kf)
    global D
    D = knn_control(process, reg, kf; Tf = 25, stab = true)
    plotD(D)
end
runplot()

##jghf



discrete_dynamics(X[:, 1], U[:, 1], psim.p, 0)

for i = rand(1:size(U, 2), 100)
    uh = predict(reg, X[:, 1])
    @test uh ≈ U[:, 1] atol=1e-4
end