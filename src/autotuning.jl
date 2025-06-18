using ForwardDiff
using FiniteDiff
using Random
# TODO: translate this to using MTK. Gang of four constraints remain on linear system

using ControlSystemsBase: LsimWorkspace

const to_autotuning = TimerOutput()
ControlSystemsBase.minreal(G::DelayLtiSystem) = G

@timeit to_autotuning "trapz" Base.@propagate_inbounds function trapz(t,x::AbstractVector)
    @boundscheck length(t) == length(x)
    s = zero(eltype(x))
    @inbounds for i = 1:length(t)-1
        s += (t[i+1] - t[i]) * (x[i+1] + x[i])/2
    end
    s
end

@timeit to_autotuning "trapz" Base.@propagate_inbounds function trapz(t,x::AbstractMatrix)
    @boundscheck length(t) == size(x, 2)
    s = zeros(size(x, 1))
    @inbounds for j = 1:length(t)-1
        for i in axes(x, 1)
            s[i] += (t[j+1] - t[j]) * (x[i, j+1] + x[i, j])/2
        end
    end
    s
end

"""
    AutoTuningProblem{S, W}

Optimizes a controller on the form
```
K(s) = C(s) * F(s) = (kp + ki/s + kd*s) * 1/((s*T)^2+2*ζ*T*s+1), ζ = 1/√2
```
```math
K(s) = C(s) F(s) = (k_p + k_i/s + k_d s)  \\dfrac{1}{(sT)^2 + 2ζTs + 1}, ζ = 1/√2
```
Can be plotted after optimization by `using Plots; plot(prob, C)` where `C` is obtained from the optimization.

See also [`OptimizedPID`](@ref)

# Keyword arguments:
- `P::LTISystem`: Plant model 
- `w::AbstractVector`: Frequency grid vector
- `Ts::Float64`: Discretization time (sample time, arbitrary time unit)
- `Tf::Float64`: Simulation duration in the same time unit as `Ts`.
- `Ms::Float64`: Maximum allowed sensitivity
- `Mt::Float64`: Maximum allowed complimentary sensitivity
- `Mks::Float64`: Maximum allowed noise sensitivity (controller times sensitivity)
- `pmax::Vector{Float64}`: An optional vector of the same length as the number of estimated parameters that contains upper bounds on parameters, the default is `fill(Inf, 4)` (no bounds).
- `metric = :IAE`: The metric to optimize. Choices are `:IAE, :IE, :IEIAE`.
- `disc`: The discretization method to use when optimizinf `metric = :IAE`. Choices are `:zoh, :foh, :tustin` (delay systems only support :zoh).
"""
struct AutoTuningProblem{S, W}
    P::S
    w::W
    Ts::Float64
    Tf::Float64
    Ms::Float64
    Mt::Float64
    Mks::Float64
    metric::Symbol
    disc::Symbol
    pmax::Vector{Float64}
    s_cache::Vector{ComplexF64}
    sge_cache::Matrix{Float64}
    ae_cache::Vector{Float64}
end

"""
    AutoTuningResult

A structure containing the results of performing PID autotuning.

This structure can be plotted `plot(res; stepsize = 1)`, where `stepsize` controls the size of the reference step to show (turn off by setting to 0). See also [`OptimizedPID`](@ref).

# Fields:
- `prob::AutoTuningProblem`
- `p::Vector`: The optimal parameter vector `[kp, ki, kd, Tf]` for a PID controller on parallel form ``K(s) = C(s)F(s) = (kp + ki/s + kd*s) 1/((sT)^2 + 2 ζ Ts + 1), ζ = 1/√2``.
- `K::ControlSystemsBase.StateSpace`: The optimal controller.
"""
struct AutoTuningResult
    prob::AutoTuningProblem
    "Optimized parameters"
    p::Vector
    "The optimized controller in the form of a StateSpace"
    K::ControlSystemsBase.StateSpace
    "Return status from the optimizer"
    ret
    "Optimal cost"
    cost
    "Time taken to solve the problem"
    timeres
    extra::Dict{Symbol, Any}
end

struct BadGuessException <: Exception
end

function AutoTuningProblem(; P, w, Ts, Tf, Ms=Inf, Mt=Inf, Mks=Inf, metric = :IAE, pmax=fill(Inf, 4), reduce=true, disc=:zoh)
    ControlSystemsBase.issiso(P) || throw(ArgumentError("Autotuning only works for SISO systems"))
    if P isa TransferFunction
        P = ss(P)
    end
    if reduce && P isa AbstractStateSpace && P.nx > 10
        @warn "The model is of high order ($(P.nx)-dimensional state). The model will be automatically reduced to no more than 10 state variables using residualized balanced truncation (`baltrunc(minreal(P), n=10, residual=true)`). To turn this off, pass `reduce=false` to `AutoTuningProblem`"
        P = minreal(P, 1e-6)
        ωc = w[end÷2]
        P = time_scale(P, 1/ωc)
        if P.nx > 10
            P.A .+= 1e-10I(P.nx) # This is safe since the time scale is adjusted
            P, _ = isstable(P) ? Core.Compiler.inferencebarrier(baltrunc2)(P, n=10, residual=true) : Core.Compiler.inferencebarrier(baltrunc_unstab)(P, n=10, residual=true)
        end
        P = time_scale(P, ωc)
    end
    if P isa AbstractStateSpace && P.nx <= 12
        P = StaticStateSpace(P)
    end
    s_cache = if ControlSystemsBase.iscontinuous(P)
        s_vec = im*w
    else
        s_vec = @. cis(w*P.Ts)
    end
    nt = length(range(0, step=Ts, stop=Tf))
    ae_cache = zeros(nt)
    sge_cache = zeros(4, nt)
    AutoTuningProblem(P, w, Float64(Ts), Float64(Tf), Float64(Ms), Float64(Mt), Float64(Mks), metric, disc, pmax, s_cache, sge_cache, ae_cache)
end

"""
    res = solve(prob::AutoTuningProblem, p0; kwargs...)
    res = solve(prob::AutoTuningProblem; kwargs...)

Computes PID parameters, that minimize load step IAE, and filter for
noise attenuation.

```math
K(s) = C(s) F(s) = (k_p + k_i/s + k_d s)  \\dfrac{1}{(sT)^2 + 2 ζ Ts + 1}
```

- `p0` Parameter vector guess: `[kp; ki; kd; T]`. If `p0` is not provied, attempts will be made to find one automatically.

See [`AutoTuningProblem`](@ref) for arguments.
`res` is of type [`AutoTuningResult`](@ref) and can be plotted `plot(res)`. It contains the optimized parameters as well as an [`ODESystem`](@ref) representing an optimized controller.

Based on K. Soltesz, C. Grimholt, S. Skogestad. Simultaneous design of proportional–integral–derivative controller and measurement filter by optimisation. Control Theory and Applications. 11(3), pp. 341-348. IET. 2017.

Solver options for metric `:IAE` include
- `maxeval = 500`
- `maxtime = 20`
- `xtol_rel = 1e-3`
- `alg` An optimizer supported by Optimization.jl. The default is IPOPT.
- `random_start = 0` A positive integer indicates a number of random starts to try in order to find a good soluiton. If `random_start = 0`, only the provided initial guess is used.

Solver options for metric `:IE` include
- `maxiter = 15`: Maximum number of convex-concave optimization iterations.
- `tol = 1e-3`: Tolerance for the convex-concave procedure.
- `solver = Hypatia.Optimizer`: Any MOI compatible solver.
- `verbose = false`:

# Extended help
The autotuner optimizes the reponse to load disturbances appearing on the process input. This typically leads to good regulation performance, but may result in a controller that produces large overshoots for step changes in the reference. If step changes in the reference are expected, we advice using one of the following strategies
- Prefilter the reference using aditional lowpass filters.
- If the system is a force/torque controlled servo system, generate a feasible reference trajectory with a continuous acceleration profile and make use of computed force/torque feedforward.
- Let the proportional and derivative terms of the PID controller act on the measurement only. This can be achieved by setting the `wp` and `wd` keyword arguments to `ModelingToolkitStandardLibrary.Blocks.LimPID` (and similar to [`OptimizedPID`](@ref)).
"""
function CommonSolve.solve(prob::AutoTuningProblem, p0; kwargs...)
    if prob.metric ∈ (:IAE, :iae)
        Core.Compiler.inferencebarrier(pidfIAE)(prob, p0; kwargs...)
    elseif prob.metric ∈ (:IE, :ie)
        Core.Compiler.inferencebarrier(pidIE)(prob, p0; kwargs...)
    elseif prob.metric ∈ (:IEIAE, :ieiae)
        prob0 = @set prob.Mks = prob.Mks
        res0 = Core.Compiler.inferencebarrier(pidIE)(prob0, p0[1:3]; kwargs...)
        res = Core.Compiler.inferencebarrier(pidfIAE)(prob, [res0.p; 0.001]; kwargs...)
        push!(res.extra, :res0 => res0)
        res
    else
        error("metric should be any of :IAE, :IE, :IEIAE")
    end
end

function CommonSolve.solve(prob::AutoTuningProblem, p0::Vector{<:AbstractVector}; mapfun = map, kwargs...)
    # NOTE: neither NLopt nor Hypatia appears thread safe :/ 
    res = mapfun(p0) do p0i
        solve(prob, p0i; kwargs...)
    end
    if prob.metric ∈ (:IE, :ie)
        return argmax(res->res.cost, res)
    else
        return argmin(res) do res
            sat = satisfied(prob, prob.P, res.p, prob.w, prob.Mks)
            res.cost + 1e6*(!sat)
        end
    end
end

function CommonSolve.solve(prob::AutoTuningProblem; verbose = false, kwargs...)
    if prob.metric ∈ (:IE, :ie)
        p0 = [
            init_strategy_loopshapingPI(prob, false);
            init_strategy_loopshapingPI(prob, true);
            init_strategy_loopshapingPID(prob);
        ]
    else
        p0 = [
            init_strategy_loopshapingPI(prob, true);
            init_strategy_loopshapingPID(prob);
        ]
    end
    if isempty(p0)
        # all above failed, try random 
        verbose && @info "Initial heuristics failed"
        p0 = Core.Compiler.inferencebarrier(find_stabilizing_controller)(prob.P, prob.w; prob.Ms, prob.Mt, prob.Mks, th=-0.1)
    end
    solve(prob, p0; verbose, kwargs...)
end


"""
    dstep(G, Ts, t, args...; kwargs...)

Discretizes `G` using ZoH with sample time `Ts` and then simulates a step response.
"""
@timeit to_autotuning "dstep" function dstep(sys::AbstractStateSpace, Ts, t, cache_index=0; disc, kwargs...)
    ws = get_cache_step(sys, t, cache_index)
    u = let u_element = [one(eltype(t))] # to avoid allocating this multiple times
        (x,t)->u_element
    end
    if ControlSystemsBase.isdiscrete(sys)
        @assert Ts == sys.Ts
        lsim!(ws, sys, u, t; kwargs...) # set to :tustin if using ForwardDiff
    else
        lsim!(ws, c2d(sys, Ts, disc), u, t; kwargs...) # set to :tustin if using ForwardDiff
    end
end

function dstep(sys::DelayLtiSystem, Ts, args...; disc = :zoh, kwargs...)
    dstep(c2d(sys, Ts, :zoh), Ts, args...; disc, kwargs...) # set to :tustin if using ForwardDiff
end

# dstep(x, Ts, args...; kwargs...) = ControlSystemsBase.step(x, args...; kwargs...)


const _cache_step = Dict{Tuple{Int, Int, Int, DataType, Int, Int}, Any}()

function get_cache_step(sys, t, cache_index)
    T = ControlSystemsBase.numeric_type(sys)
    key = sys.ny, sys.nu, sys.nx, T, length(t), cache_index
    get!(_cache_step, key, LsimWorkspace(sys, length(t)))::LsimWorkspace{T}
end


_square(x) = ss(x * x) # workaround for static matrices
_square(x::ControlSystemsBase.DelayLtiSystem) = (x * x) 

# This version appears better for delay systems for some reason. Might be due to too limited benchmarking systems, or too coarce frequency grid making the problem very non-smooth.
@timeit to_autotuning "autotuning_cost_grad" function autotuning_cost_grad(P, p, prob::AutoTuningProblem{<:DelayLtiSystem}, tfs::TFS) where TFS
    all(isfinite, p) || throw(BadGuessException())
    S, PS, ∇K_p = tfs.S, tfs.PS, tfs.∇K_p
    Ts, Tf = prob.Ts, prob.Tf
    t = range(0, stop=Tf, step=Ts)
    @timeit to_autotuning "dstep 1" e, _ = dstep(P * S(p), Ts, t; prob.disc)
    ve = vec(e)
    @. prob.ae_cache = abs(ve)
    J = trapz(t, vec(prob.ae_cache)) # TODO: this only works for siso
    # ae_cache = abs.(ve)
    # J = trapz(t, ae_cache) # TODO: this only works for siso
    # gK = ∇K_p(p)
    # @timeit to_autotuning "dstep 2" ∇e, _ = dstep(minreal(minreal(-gK * _square(P)) * _square(S(p))), Ts, t) # step(P*dS/dp)
    # # if P isa DelayLtiSystem && ndims(∇e) == 3 # This system behaves slightly differently
    # #     ∇e = dropdims(∇e, dims=1)'
    # # end
    # @. prob.sge_cache = sign(ve') * ∇e
    # Jp = trapz(t, prob.sge_cache)
    J > 1e10 && throw(BadGuessException())
    J
end

# This implementation takes a few more iterations on the benchmark problem, but allocates less
@timeit to_autotuning "autotuning_cost_grad" function autotuning_cost_grad(P, p, prob, tfs::TFS) where TFS
    all(isfinite, p) || throw(BadGuessException())
    S, PS, ∇K_p = tfs.S, tfs.PS, tfs.∇K_p
    Ts, Tf = prob.Ts, prob.Tf
    t = range(0, stop=Tf, step=Ts)
    PSp = PS(p)
    # PSp = minreal(P*S(p))
    @timeit to_autotuning "dstep 1" e, _ = dstep(PSp, Ts, t, 1; prob.disc)
    ve = vec(e)
    @. prob.ae_cache = abs(ve)
    J = trapz(t, prob.ae_cache) # TODO: this only works for siso
    # ae_cache = abs.(ve)
    # J = trapz(t, ae_cache) # TODO: this only works for siso

    # gK = ∇K_p(p)
    # gK.C .*= -1
    # gK.D .*= -1
    # if P isa DelayLtiSystem
    #     PSpd = c2d(PSp, Ts, :zoh)
    #     # simsys = balance_statespace(c2d(gK, Ts, :tustin)*(PSpd*PSpd))[1]
    #     simsys = c2d(gK, Ts, :tustin)*(PSpd*PSpd)
    #     @timeit to_autotuning "dstep 2" ∇e, _ = dstep(simsys, Ts, t, 2; prob.disc) # step(P*dS/dp)
    # else
    #     @timeit to_autotuning "dstep 2" ∇e, _ = dstep(gK*(PSp*PSp), Ts, t, 2; prob.disc) # step(P*dS/dp)
    # end
    # # if P isa DelayLtiSystem && ndims(∇e) == 3 # This system behaves slightly differently
    # #     ∇e = dropdims(∇e, dims=1)'
    # # end
    # @. prob.sge_cache = sign(ve') * ∇e
    # Jp = trapz(t, prob.sge_cache)
    J > 1e10 && throw(BadGuessException())
    J
end

struct TFS{TC,TF,TK,TS,TPS,T∇C_pC,T∇F_pF,T∇K_p}
    C::TC
    F::TF
    K::TK
    S::TS
    PS::TPS
    ∇C_pC::T∇C_pC
    ∇F_pF::T∇F_pF
    ∇K_p::T∇K_p
end

"""
    autotuning_constraints(p::AbstractVector{T}, fr, c0, prob, tfs)

Compute the constraints for the autotuning problem

# Arguments:
- `p`: parameter vector
- `fr`: frequency evaluation function
- `c0`: regularization parameter
- `prob`: `AutoTuningProblem` instance
- `tfs`: `TFS` instace
"""
#@timeit to_autotuning "autotuning_constraints"
function autotuning_constraints(p::AbstractVector{T}, fr, c0, prob, tfs::TFS{TC,TF,TK,TS,T∇C_pC,T∇F_pF,T∇K_p}, pmax)::typeof(p) where {T,TC,TF,TK,TS,T∇C_pC,T∇F_pF,T∇K_p}
    @unpack Ms, Mt, Mks = prob
    S, C, F = tfs.S, tfs.C, tfs.F
    cS = T[]
    cT = T[]
    cKS = T[]
    cp = T[]

    # Sensitivity
    @timeit to_autotuning "fr(S)" Sw=fr(S(p)) |> vec
    if Ms < Inf
        Sm = abs.(Sw)
        cS = Sm .- Ms
    end

    # Complementary sensitivity
    if Mt < Inf
        cT = @. abs(1 - Sw) - Mt 
    end

    # Noise sensitivity
    if Mks < Inf
        @timeit to_autotuning "fr(CF)" Kw=fr(C(p) * F(p)) |> vec
        # Km = abs.(Kw)
        # KSm = Km .* Sm .* c0 # Undo regularization
        # cKS = KSm .- Mks
        cKS = Sm # This can be reused here
        @. cKS = abs(Kw) * Sm * c0 - Mks # optimized version of the above
    end

    # derivative vs. filter corner frequencies
    kp, ki, kd, Tf = p
    s0 = kp + sqrt(max(kp^2 - 4ki*kd, 1e-10one(kd))) # Don't use zero here in case kp is zero, the gradient will be NaN

    cTf = Tf*abs(s0) - 2kd

    all(isfinite, cS) || throw(BadGuessException())
    all(isfinite, cT) || throw(BadGuessException())
    all(isfinite, cKS) || throw(BadGuessException())
    all(isfinite, cp) || throw(BadGuessException())
    all(isfinite, cTf) || throw(BadGuessException())

    # Constraints
    c = [cS; cT; cKS; cTf] # one long vector
end

const C1 = let
    s = tf("s")
    [1; 1 / s; s]
end

using StaticArrays
"""
    TFS(P, pC, pF)

Create a structure holding closures for the autotuning problem.
"""
function TFS(P, pC, pF)
    s = tf("s")
    z = 1 / sqrt(2)    # Filter damping
    C = p -> transpose(pC(p)) * C1
    # F = p -> 1 / ((pF(p) * s)^2 + 2 * z * pF(p) * s + 1)
    F = function (p::AbstractVector{T}) where T
        Tf = p[4]
        Tf = max(Tf, T(1e-20))
        tf([one(Tf)], [Tf^2, 2*z*Tf, one(Tf)])
    end
    K = p -> ControlSystemsBase.series(F(p), C(p))
    if P isa AbstractStateSpace
        S, PS = let P = P
            S = function (p::AbstractVector{T}) where {T}
                kp, ki, kd, Tf = p
                Tf = max(T(1e-12), Tf)
                Tf2 = Tf*Tf
                # all other functions can convert known-size controllers to static systems. Turned out not to be worth it
                A = SA[0 1 0
                       0 0 1
                       0 -(1 / Tf2) -(1.414213562373095 / Tf)]
                B = SMatrix{3,1,Float64,3}(0.0, 0, 1)

                C_ = SA[ki/Tf2 kp/Tf2 kd/Tf2]
                D = SMatrix{1, 1, Float64, 1}(0.0)
                Kp = HeteroStateSpace(A, B, C_, D)
                PK = ControlSystemsBase.series(P, Kp)
                feedback(1, PK)
            end

            PS = function (p::AbstractVector{T}) where {T}
                kp, ki, kd, Tf = p
                Tf = max(T(1e-12), Tf)
                Tf2 = Tf*Tf
                # all other functions can convert known-size controllers to static systems. Turned out not to be worth it
                A = SA[0 1 0
                       0 0 1
                       0 -(1 / Tf2) -(1.414213562373095 / Tf)]
                B = SMatrix{3,1,Float64,3}(0.0, 0, 1)

                C_ = SA[ki/Tf2 kp/Tf2 kd/Tf2]
                D = SMatrix{1, 1, Float64, 1}(0.0)
                Kp = HeteroStateSpace(A, B, C_, D)
                feedback(P, Kp)
            end

            S, PS
        end
    else # E.g., delay systems
        S = p -> feedback(1, ControlSystemsBase.series(P, convert(StateSpace, K(p), balance = false)))
        PS = p -> feedback(P, convert(StateSpace, K(p), balance = false))
    end

    # Sensitivities
    ∇C_pC = p -> C1                         # dK/dpC
    # ∇F_pF = p -> (-2) * F(p)^2 * s * (s * pF(p) + z)  # dF/dpF
    ∇F_pF = function (p::AbstractVector{T}) where T
        kp, ki, kd, Tf = p
        Tf = max(Tf, T(1e-20))
        Tf2 = Tf*Tf
        Tf3 = Tf2*Tf
        Tf4 = Tf2*Tf2
        tf([-2Tf, -sqrt(2), 0], [Tf4, 2.82842712474619*Tf3, 4.0*Tf2, 2.82842712474619*Tf, 1]) 
    end
    ∇K_p = p -> minreal(ss([∇C_pC(p) * F(p); C(p) * ∇F_pF(p)])) # Not much is saved by making this static, most allocations come from lsim. Converting to ss before minreal does improve performance in at least one test, but causes an error due to improper tf if there is no filter

    TFS(C,F,K,S,PS,∇C_pC,∇F_pF,∇K_p)
end

# p0 = @vars kp ki kd Tf; p0 = [p0...]
# # p0 = @variables kp, ki, kd, Tf
# tfs.∇K_p(p0) |> ss |> sminreal

# [tfs.∇C_pC(p0) * tfs.F(p0); minreal(tfs.C(p0) * tfs.∇F_pF(p0))] |> ss

# s = tf("s")
# z = 1 / sqrt(2)    # Filter damping
# (-2) * tfs.F(p0)^2 * s * (s * p0[4] + z)

function faster_freqresp(sys, w, s_vec) # No timer output here since it sometimes causes an access to undefined reference
    ny = ControlSystemsBase.noutputs(sys)
    [ControlSystemsBase.evalfr(sys[i,1], s)[] for s in s_vec, i in 1:ny]
end

function faster_freqresp(sys::TransferFunction, w, s_vec) # No timer output here since it sometimes causes an access to undefined reference
    ny = ControlSystemsBase.noutputs(sys)
    FR = get_cache_freqresp(sys, w)
    for (j,s) in enumerate(s_vec)
        for i in 1:ny
            FR[i,1,j] = ControlSystemsBase.evalfr(sys.matrix[i,1], s)
        end
    end
    transpose(dropdims(FR, dims = 2))
end

function faster_freqresp(sys::AbstractStateSpace, w, s_vec)
    FR = get_cache_freqresp(sys, w)
    ControlSystemsBase.freqresp_nohess!(FR, sys, w)
    # ControlSystemsBase.freqresp!(FR, sys, w) # Slower, at least for small systems
    transpose(dropdims(FR, dims = 2))
end

const _cache_freqresp = Dict{Tuple{Int, Int, DataType, Int}, Any}()

function get_cache_freqresp(sys, w)
    T = Complex{ControlSystemsBase.numeric_type(sys)}
    key = sys.ny, sys.nu, T, length(w)
    get!(_cache_freqresp, key, zeros(T, sys.ny, sys.nu, length(w)))::Array{T, 3}
end

# Base.delete_method.(methods(faster_freqresp))

function adjust_initial_guess(p0, P, w)
    length(p0) == 4 || throw(ArgumentError("Initial guess vector must be of length 4 [kp, ki, kd, Tf]"))
    # Extract options
    pC = p0[1:3]
    pF = p0[4]

    # regularize
    if pC[1] != 0
        c0 = 1#pC[1]
        pC ./= c0
        P = P * c0
    else
        c0 = one(eltype(p0))
    end

    # Ensure LP filter breakdown within grid
    Tmin = 1 / w[end]
    pF = max(pF, Tmin)
    p::typeof(p0) = [pC; pF]

    pCf = p -> p[1:3]
    pFf = p -> p[4]
    p[4] = max(p[4], Tmin)
    p, pCf, pFf, c0, P, Tmin
end

function find_initial_guess(p, cost, con, Tmin, N)
    N = max(3, N)
    th = 0.5 # we allow some constraint violation in the start
    ranges = [shuffle!(exp10.(LinRange(-3, 3, N))) for _ in 1:3]
    push!(ranges, shuffle!(exp10.(LinRange(log10(Tmin), 3, N))))
    candidates = reduce(hcat, ranges)
    bestpar = p
    J = cost(p)
    bestval = Inf
    if maximum(con(p)) < th
        bestval = J
    end
    disable_timer!(to_autotuning) # Timer not thread safe
    Threads.@threads for i = 1:N
    # for i = 1:N
        pi = candidates[i,:]
        c = maximum(con(pi))
        c > th && continue
        J = cost(pi)
        if J < bestval # NOTE: this is non-deterministic under threads, but okay since this is a fundamentally stochastic algorithm anyways
            bestval = J
            bestpar = pi
        end
    end
    enable_timer!(to_autotuning)
    @info "Found initial value $bestval at $bestpar"
    bestpar
end

IpoptSolver(;
    verbose                     = false,
    tol                         = 1e-3,
    acceptable_tol              = 1e-2,
    max_cpu_time                = 30.0,
    max_wall_time               = 30.0,
    max_iter                    = 500,
    constr_viol_tol             = 0.001,
    acceptable_constr_viol_tol  = 0.02,
    exact_hessian               = false,
    kwargs...
) = MPC.IpoptSolver(; verbose,tol,acceptable_tol,max_cpu_time,max_wall_time,max_iter,constr_viol_tol,acceptable_constr_viol_tol,exact_hessian,kwargs...)


"""
    res = solve(prob::AutoTuningProblem; kwargs...)
    res = solve(prob::AutoTuningProblem, p0; kwargs...)

Computes PID parameters, that minimize load step IAE, and filter for
noise attenuation.

```math
K(s) = C(s) F(s) = (k_p + k_i/s + k_d s)  \\dfrac{1}{(sT)^2 + 2 ζ Ts + 1}
```

- `p0` Parameter vector: `[kp; ki; kd; T]`. If not provided, an attempt at automatically finding a feasible initial guess is made.

See [`AutoTuningProblem`](@ref) for arguments.
`res` is of type [`AutoTuningResult`](@ref) and can be plotted `plot(res)`. It contains the optimized parameters as well as an [`ODESystem`](@ref) representing an optimized controller.

Based on K. Soltesz, C. Grimholt, S. Skogestad. Simultaneous design of proportional–integral–derivative controller and measurement filter by optimisation. Control Theory and Applications. 11(3), pp. 341-348. IET. 2017.

Solver options for metric `:IAE` include
- `alg = IpoptSolver(; verbose)` A solver compatible with Optimization.jl
- `random_start = 0` A positive integer indicates a number of random starts to try in order to find a good soluiton. If `random_start = 0`, only the provided initial guess is used.

Solver options for metric `:IE` include
- `maxiter = 15`: Maximum number of convex-concave optimization iterations.
- `tol = 1e-3`: Tolerance for the convex-concave procedure.
- `solver = Hypatia.Optimizer`: Any MOI compatible solver.
- `verbose = false`:

# Extended help
The autotuner optimizes the reponse to load disturbances appearing on the process input. This typically leads to good regulation performance, but may result in a controller that produces large overshoots for step changes in the reference. If step changes in the reference are expected, we advice using one of the following strategies
- Prefilter the reference using aditional lowpass filters.
- If the system is a force/torque controlled servo system, generate a feasible reference trajectory with a continuous acceleration profile and make use of computed force/torque feedforward.
- Let the proportional and derivative terms of the PID controller act on the measurement only. This can be achieved by setting the `wp` and `wd` keyword arguments to `ModelingToolkitStandardLibrary.Blocks.LimPID` (and similar to [`OptimizedPID`](@ref)).
"""
function pidfIAE(prob0::AutoTuningProblem, p0; maxeval = 500, maxtime = 40, xtol_rel = 1e-3, random_start = false, verbose = true, alg = IpoptSolver(; verbose),kwargs...)

    prob, p0, scale_info = _scale_numerics(prob0, p0)
    reset_timer!(to_autotuning)
    @unpack P, Ms, Mt, Mks, pmax, w = prob
    p0 = copy(p0)

    p, pC, pF, c0, P, Tmin = adjust_initial_guess(p0, P, w)

    # Helper function
    fr = x -> faster_freqresp(x, w, prob.s_cache) # frequency response over grid

    tfs = TFS(P, pC, pF)

    function costfun(x::AbstractVector, args...)
        @timeit to_autotuning "cost_grad" cons = autotuning_cost_grad(P, x, prob, tfs)
        return cons
    end
    jacfun = let fr = fr, c0 = c0, prob = prob, tfs = tfs
        x -> autotuning_constraints(x, fr, c0, prob, tfs, pmax)
    end
    num_const = count(isfinite, [Ms, Mt, Mks])
    num_pmaxcon = any(isfinite, pmax) ? min(length(p), length(pmax)) : 0
    num_pmaxcon = 0 # count(isfinite, pmax[1:num_pmaxcon]) # zero with Optimization.jl
    num_tf_const = 1

    total_num_const = num_pmaxcon + num_const * length(w) + num_tf_const


    lb = [eps(), 0, 0, Tmin]
    for i = 1:3
        p[i] <= 0 && (lb[i] = -1e6)
    end
    
    ub = [1e6, 1e6, 1e6, 1e6]

    if random_start > 0
        p = find_initial_guess(p, p->autotuning_cost_grad(P, p, prob, tfs), jacfun, Tmin, random_start)::typeof(p)
    end


    for i in eachindex(p)
        isfinite(pmax[i]) || continue
        ub[i] = pmax[i] / (i <= 3 ? c0 : one(c0))
    end

    for i = 1:min(4, length(p))
        p[i] = min(p[i], ub[i])
    end
    

    # It's useful to print those for debugging purposes. Errors within the call to NLopt are not shown.
    # @show p
    # @show costfun(p)
    # @show autotuning_constraints(p, fr, c0, prob, tfs, pmax)
    # @assert length(autotuning_constraints(p, fr, c0, prob, tfs, pmax)) == total_num_const
    # @show constraintfun(zeros(total_num_const), p, jac_cache')
    # @show total_num_const


    # ForwardDiff isn't always better than FiniteDiff here, likely because the numerical smoothing implied by the finite difference is beneficial for the matrix exponential function. FD also takes much longer to compile
    optfun = OptimizationFunction(costfun, Optimization.AutoFiniteDiff();
                    cons=(c, x, p)->c .= autotuning_constraints(x, fr, c0, prob, tfs, pmax))
    optprob = OptimizationProblem(optfun, p; lb, ub, lcons=fill(-Inf, total_num_const), ucons=fill(0.0, total_num_const))
    maxtime = float(maxtime)
    local sol, timeres
    try
        timeres = @elapsed sol = solve(optprob, alg)
    catch e
        e isa BadGuessException || rethrow()
        timeres = 0.0
        sol = (u = p, retcode = :BAD_GUESS, minimum = Inf)
    end
    # numevals = opt.numevals # the number of function evaluations
    p = sol.u
    ret = sol.retcode
    minf = sol.minimum
    verbose && println("got $minf at $p")
    c = autotuning_constraints(p, fr, c0, prob, tfs, pmax)
    cv = maximum(c)
    satisfied = cv < 3e-2
    if !satisfied
        verbose && @warn "Constraints not satisfied after optimization, maximum violation: $(cv)"
        minf = Inf
        ret = :CONSTRAINT_VIOLATION
    end

    p[1:end-1] = p[1:end-1] * c0

    p = _unscale_numerics(p, scale_info)

    K = let tfs2 = TFS(prob0.P, pC, pF)
        ss(tfs2.K(p))
    end
    K, _ = ControlSystemsBase.balance_statespace(K)
    K = K |> minreal
    # pid = OptimizedPID(p, name=:pid)
    extra = Dict{Symbol, Any}(:p0 => p0, :c0 => c0, :tfs => tfs, :alg => alg, :opt => sol, :scale_info => scale_info, :P_scale=>P, :prob_scale => prob)
    AutoTuningResult(prob0, p, K, ret, minf, timeres, extra)
end


@recipe function plot(res::AutoTuningResult; stepsize = 1)
    prob = res.prob
    C = res.K
    tv = 0:prob.Ts:prob.Tf
    w, P = prob.w, prob.P
    Ms = prob.Ms
    Mt = prob.Mt
    S, PS, CS, T = ControlSystemsBase.gangoffour(P, C)
    layout --> 4 # @layout [[a b; c d] e]
    plotphase := false
    xguide --> "Frequency [rad/s]"
    legend := :bottomright
    legend_background_color --> Colors.ARGB(1.0,1.0,1.0,0.5)
    subplot := 1
    titlefontsize --> 10
    @series begin
        xscale --> :log10
        yscale --> :log10
        b, _ = bode(S,w,unwrap=false)
        b = vec(b)
        label --> "S"
        w, b
    end
    @series begin
        xscale --> :log10
        yscale --> :log10
        b, _ = bode(T,w,unwrap=false)
        b = vec(b)
        label --> "T"
        w, b
    end
    @series begin
        linestyle --> :dash
        linecolor --> 1
        seriestype --> :path
        primary --> false
        label := "Ms"
        collect(extrema(w)), [Ms, Ms]
    end
    @series begin
        title --> "Sensitivity functions (S / T)"
        ylims --> (1e-2, 3)
        # seriestype --> :hline
        linestyle --> :dash
        linecolor --> 2
        seriestype --> :path
        primary --> false
        label := "Mt"
        collect(extrema(w)), [Mt, Mt]
    end
    subplot := 2
    @series begin
        xscale --> :log10
        yscale --> :log10
        b, _ = bode(CS,w,unwrap=false)
        b = vec(b)
        label --> "CS"
        w, b
    end
    @series begin
        xscale --> :log10
        yscale --> :log10
        b, _ = bode(C,w,unwrap=false)
        b = vec(b)
        label --> "C"
        w, b
    end
    @series begin
        title --> "Noise sensitivity and controller (CS / C)"
        seriestype --> :path
        linestyle --> :dash
        linecolor --> 1
        label --> "Mks"
        primary --> false
        collect(extrema(w)), [prob.Mks, prob.Mks]
    end
    subplot := 3
    # All series are split up into separate @series blocks due to https://github.com/JuliaPlots/Plots.jl/issues/4108
    @series begin
        step1 = dstep(PS, prob.Ts, tv; prob.disc) |> deepcopy # Deepcopy due to dstep using cached workspaces
        linecolor --> 1
        label := "PS (disturbance step)"
        step1
    end
    if stepsize != 0
        @series begin
            step2 = dstep(stepsize*T, prob.Ts, tv; prob.disc) |> deepcopy # Deepcopy due to dstep using cached workspaces
            title --> "Step responses"
            linecolor --> 2 # Use linecolor 2 for reference step to align with T
            xguide := "Time [s]"
            label := "T  (reference step)"
            step2
        end
    end
    cs = -1             # Ms center
    rs = 1 / Ms         # Ms radius
    ct = -Mt^2/(Mt^2-1) # Mt center
    rt = Mt/(Mt^2-1)    # Mt radius
    θ = range(0, stop=2π, length=100)
    Sin, Cos = sin.(θ), cos.(θ)
    re, im = nyquist(P*C,w)
    subplot := 4
    @series begin
        linecolor --> 1
        label := ""
        vec(re), vec(im)
    end
    # If legend entries end up out of order, it might be due to https://github.com/JuliaPlots/Plots.jl/issues/4108
    @series begin
        # primary := true
        linestyle := :dash
        linecolor := 1
        seriestype := :path
        markershape := :none
        label := "S = $Ms"
        cs.+rs.*Cos, rs.*Sin
    end
    @series begin
        # primary := true
        linestyle := :dash
        linecolor := 2
        seriestype := :path
        markershape := :none
        label := "T = $Mt"
        ct.+rt.*Cos, rt.*Sin
    end
    @series begin # Mark the critical point
        primary := false
        markershape := :xcross
        seriescolor := :red
        markersize := 5
        seriesstyle := :scatter
        xguide := "Re"
        yguide := "Im"
        framestyle := :zerolines
        title --> "Nyquist plot"
        xlims --> (-3, 1)
        ylims --> (-3, 1)
        [-1], [0]
    end
    nothing
end

"""
    OptimizedPID(popt; name, kwargs...)
    OptimizedPID(res::AutoTuningResult; name, kwargs...)

Takes optimized parameters `popt` and returns the following system
```
          ┌───┐
     ────►│ F ├───────┐ ┌─────┐
reference └───┘       └►│     │   ctr_output
                        │ PID ├─────►
          ┌───┐       ┌►│     │
       ┌─►│ F ├───────┘ └─────┘
       │  └───┘
       │
       └───── measurement
```

# Arguments:
- `popt`: Obtained by solving an [`AutoTuningProblem`](@ref)
- `kwargs`: Are passed to `ModelingToolkitStandardLibrary.Blocks.LimPID`
"""
function OptimizedPID(popt::Vector; name, Nd = 10000, kwargs...) # Nd is set to a very high value since we have separate noise filters
    # We set Nd very high by default since we have an explicit filter F.
    kp, ki = popt[1:2]
    kd = T = false
    if length(popt) >= 3
        popt[3] < 0 && error("Negative derivative gain is not supported")
        kd = max(popt[3], 0)
        if length(popt) >= 4
            T = popt[4]
        end
    end
    # (kp + ki/s + kd*s) * 1/((s*T)^2+2*ζ*T*s+1)
    k, Ti, Td = ControlSystemsBase.convert_pidparams_to_standard(kp, ki, kd, :parallel)
    if T == 0
        return Blocks.LimPID(; k, Ti, Td, Nd, name, kwargs...)
    end
    @named pid = Blocks.LimPID(; k, Ti, Td, Nd, kwargs...)
    ζ = 1/√2
    @named filter_r = Blocks.SecondOrder(; w = 1/T, d = ζ, k=1)
    @named filter_y = Blocks.SecondOrder(; w = 1/T, d = ζ, k=1)
    @named reference = Blocks.RealInput()
    @named measurement = Blocks.RealInput()
    @named ctr_output = Blocks.RealOutput()
    
    t = ModelingToolkit.get_iv(pid)
    ODESystem(
        [
            reference.u ~ filter_r.input.u
            connect(filter_r.output, pid.reference)
            
            measurement.u ~ filter_y.input.u
            connect(filter_y.output, pid.measurement)
            
            pid.ctr_output.u ~ ctr_output.u
        ],
        t; systems=[filter_r, filter_y, pid, reference, measurement, ctr_output], name)
end

OptimizedPID(res::AutoTuningResult; kwargs...) = OptimizedPID(res.p; kwargs...)

"""
    check_stabilizing(p, P)

Check if the parameters `p` representing a PID controller stabilizes `P`.
"""
function check_stabilizing(p, P)
    try
        C = ControlSystemsBase.pid(p) * tf(1, [1e-6, 1])
        return isstable(feedback(P*C, 1))
    catch
    end
end

## pidIE
"""
    pidIE(prob::AutoTuningProblem, p0; maxiter = 15, tol = 0.001, solver = Hypatia.Optimizer, verbose = false)

Computes PID parameters, that minimize load step integrated error (notice, *not* integrated absolute error). 
```
K(s) = (kp + ki/s + kd*s)
```
```math
K(s) = (k_p + k_i/s + k_d s)
```

`p0`: Parameter vector: `[kp; ki; kd]` The `kd` parameter is optional, if it's omitted, a PI controller will be designed.

See [`AutoTuningProblem`](@ref) for arguments.
`res` is of type [`AutoTuningResult`](@ref) and can be plotted `plot(res)`. It contains the optimized parameters as well as an [`ODESystem`](@ref) representing an optimized controller.

Based on M. Hast, K. J. Astrom, B. Bernhardsson, S. Boyd. PID design by convex-concave optimization. European Control Conference. IEEE. Zurich, Switzerland. 2013.

# Arguments:
- `prob`: A problem description of type [`AutoTuningProblem`](@ref)
- `p0`: Initial guess of parameter vector: `[kp; ki; kd]` The `kd` parameter is optional, if it's omitted, a PI controller will be designed.
- `maxiter`: In the outer loop
- `tol`: In the outer loop
- `solver`: Currently defaults to Hypatia.jl
- `verbose` set to true to print diagnostics.
"""
function pidIE(prob0::AutoTuningProblem, p0;
    maxiter = 15,
    tol = 1e-3,
    solver = Hypatia.Optimizer,
    verbose = false,
    kwargs...
)

    # TODO: this function operates completely in the frequency domain and can thus be made to work with FRD objects. This would be nice since then nonlinear plants can be linearized by means of Fourier methods, for which the coherence function would give a nice uncertainty measure. This method can incorporate constraints to account for plant uncertainty, see the paper by Hast. The constraints amount to forming circles around the Nyquist curve with a frequency-dependent radius. These can also be used for parametric uncertainty by solving one convex circle-cover problem per frequency prior to optimization.

    prob, p, scale_info = _scale_numerics(prob0, p0)
    @unpack P, Ms, Mt, Mks, pmax, w = prob

    extra = Dict{Symbol, Any}(:p0 => p0, :solver => solver, :scale_info => scale_info)

    stab = check_stabilizing(p, P)
    if stab == false # stab can be nothing
        if isstable(P)
            p = find_stabilizing_controller(P, w; Ms, Mt, Mks)
            verbose && @warn "Initial controller p0 not stabilizing, setting initial guess to $p"
        else
            return AutoTuningResult(prob0, p, tf(0), nothing, 0.0, 0.0, extra)
            # error("Initial controller not stabilizing, adjust p0 to initialize optimization with a stabilizing controller.")
        end
    end

    Pf = faster_freqresp(P, w, prob.s_cache)

    # Constraint represented as circles with centers and radii
    cs = -1             # Ms center
    rs = 1 / Ms         # Ms radius
    ct = -Mt^2/(Mt^2-1) # Mt center
    rt = Mt/(Mt^2-1)    # Mt radius

    # controller structure
    K1 = [ones(size(w))  1 ./ complex.(0, w)  complex.(0, w)]
    n = min(3, length(p)) # 2 for PI, 3 for PID
    K1 = K1[:,1:n]
    K = K1*p[1:n]
    prev_obj = -Inf
    local pv, oprob
    pv = Convex.Variable(n)
    feasible_found = satisfied(prob, P, p0, w, prob.Mks)
    best_feasible = p0
    best_objective = p0[2]
    timeres = @elapsed for iter = 1:maxiter
        verbose && @info "pidIE iteration: $iter"
        Lc = Pf .* K # Loop transfer function
        oprob = Convex.maximize(pv[2]) # Maximize integral gain
        Kv = K1*pv
        L = Pf.*Kv
        if Ms < Inf
            Lccs = Lc .- cs
            oprob.constraints += real(conj(Lccs./abs.(Lccs)).*(L-cs)) >= rs # Sensitivity constraint
        end
        if Mt < Inf
            Lcct = Lc .- ct
            oprob.constraints += real(conj(Lcct./abs.(Lcct)).*(L-ct)) >= rt  # Complementary sensitivity constraint
        end
        if Mks < Inf
            oLc = 1 .+ Lc
            oprob.constraints += abs(Kv) - Mks*real(identity(conj(oLc)./(abs.(oLc))).*(1+L)) <= 0 # Noise sensitivity constraint NOTE: the call to identity is to circumvent Convex bug https://github.com/jump-dev/Convex.jl/issues/355
        end
        for i in 1:min(length(pv), length(pmax))
            pmax[i] < Inf || continue
            oprob.constraints += pv[i] <= pmax[i]
        end


        Convex.solve!(oprob, solver; silent=!verbose)
        p1 = evaluate(pv)
        any(isnan, p) && error("Nans detected, try increasing Mₛ and Mₜ")

        Mks_satisfied = Mks == prob.Mks

        if check_stabilizing(p1, P) == false # can be nothing
            Mks *= 1.5
            verbose && @info "Controller not stabilizing, increasing Mks to $Mks"
            if Mks > 1e6
                error("Controller not stabilizing, Mks has been increased to $Mks without any success")
            end
            continue
        elseif !Mks_satisfied # If successful and we've previously increased Mks, try lowering it back down
            Mks = max(prob.Mks, Mks/1.5)
            verbose && @info "Decreasing Mks to $Mks"
        elseif satisfied(prob, P, p1, w, prob.Mks) # All good
            feasible_found = true
            best_feasible = p1
            if p1[2] > best_objective && (p1[2]-best_objective) < tol*p1[2] # Stop if tolerance met and Mks is at specified value
                break
            end
            best_objective = p1[2]
        end
        p = p1 # accept change
        K = evaluate(Kv)
        any(isnan, K) && error("Nans detected, try increasing Mₛ and Mₜ")
    end

    p = best_feasible

    if check_stabilizing(p, P) == false
        verbose && @error("Failed to find a stabilizing controller, Mks = $Mks")
        return AutoTuningResult(prob0, p, tf(0), nothing, 0.0, 0.0, extra)
    end
    p = _unscale_numerics(p, scale_info)
    K = ControlSystemsBase.pid(p) * tf(1, [1/maximum(w), 1])
    K, _ = ControlSystemsBase.balance_statespace(ss(K))
    # pid = OptimizedPID(p, name=:pid)
    AutoTuningResult(prob0, p, K, oprob, feasible_found ? p[2] : 0.0, timeres, extra) # p[2] is the cost function value
end

function satisfied(prob, P, p, w, Mks)
    try
        F = if length(p) >= 4
            Tf = p[4]
            tf([one(Tf)], [Tf^2, 2/sqrt(2)*Tf, one(Tf)])
        else
            tf(1, [1/w[end], 1])
        end
        C = ss(pid(p)*F)
        PC = P*C
        S = feedback(1, PC)
        T = feedback(PC)
        CS = G_CS(P, C)
        nS = hinfnorm2(S)[1]
        nS <= prob.Ms*1.02 || return false
        nT = hinfnorm2(T)[1]
        nT <= prob.Mt*1.02 || return false
        nCS = hinfnorm2(CS)[1]
        nCS <= Mks*1.1 || return false
        return true
    catch
        false
    end
end


function find_stabilizing_controller(P, w; Ms, Mt, Mks, th=-0.1)
    P, _, w, scale_info = _scale_numerics(P, nothing, w)
    p0 = [-6.0, -9, -9]
    function cost(p)
        p = exp10.(p)
        try
            C = pid(p)*tf(1, [1/w[end], 1])
            PC = P*C
            S = feedback(1, PC)
            T = feedback(PC)
            CS = C*S
            polecost = 10max(-th, (maximum(real(poles(T))) - th))
            Scost = sqrt(max(0, hinfnorm2(S)[1] - Ms))
            Tcost = sqrt(max(0, hinfnorm2(T)[1] - Mt))
            b = bodev(CS, w, unwrap=false)[1]
            b .= b .- Mks
            CScost = sqrt(max(0, maximum(b)))
            noinf(x) = isfinite(x) ? x : 100one(x)
            polecost + noinf(Scost) + noinf(Tcost) + noinf(CScost)
        catch
            Inf
        end
    end
    callback = r -> r.value < th
    res = Optim.optimize(
        cost,
        p0,
        Optim.ParticleSwarm(),
        Optim.Options(;
            store_trace       = false,
            show_trace        = false,
            show_every        = 1,
            iterations        = 1000,
            allow_f_increases = false,
            time_limit        = 100,
            x_tol             = 0,
            g_tol             = 1e-8,
            f_calls_limit     = 0,
            g_calls_limit     = 0,
            # callback,
        ),
    )
    p = exp10.(res.minimizer)
    _unscale_numerics(p, scale_info)
end


# Create a PID controller from a parameter vector
ControlSystemsBase.pid(p::AbstractVector) = ControlSystemsBase.pid(p[1], p[2], length(p) >= 3 ? p[3] : zero(eltype(p)), form=:parallel)



"""
Scale the gain and frequency axis of P to be around 1.
"""
function _scale_numerics(P, p, w)
    # return (P, p, w, (; g_scale=1, ωc=1))
    P isa AbstractStateSpace || (return (P, p, w, (; g_scale=1, ωc=1)))
    # return (P, p, w, (; g_scale=1, ωc=1))
    ωc = 1 # w[end÷2] # Center frequency NOTE: this is deactivated since Mks constraints are not always satisfied when we rescale frequency axis. 
    P = time_scale(P, 1/ωc)
    ps = poles(P)
    num_integ = count(abs(p) < 1e-5 for p in ps)
    if num_integ == 0
        g_scale = abs(dcgain(P)[])
    else
        g_scale = abs(freqresp(P, ωc)[])
    end
    P = (1/g_scale) * P
    if p !== nothing
        p = copy(p)
        # Time scaling with ωc corresponds to dividing ki, kd, and Tf with ωc, 
        p[1] *= g_scale
        p[2] *= (g_scale/ωc)
        length(p) >= 3 && (p[3] *= (g_scale*ωc))
        length(p) == 4 && (p[4] *= ωc)

    end
    w = w ./ ωc
    P, p, w, (; g_scale, ωc)
end

"The inverse of _scale_numerics"
function _unscale_numerics(p, scaling_info)
    g_scale, ωc = scaling_info
    p = copy(p)
    p[1] /= g_scale
    p[2] /= g_scale / ωc
    length(p) >= 3 && (p[3] /= ωc*g_scale)
    length(p) >= 4 && (p[4] /= ωc)
    p
end

function _scale_numerics(prob::AutoTuningProblem, p=nothing)
    P, p, w, scale_info = _scale_numerics(prob.P, p, prob.w)
    _, pmax, _ = _scale_numerics(prob.P, prob.pmax, prob.w)

    ωc = scale_info.ωc
    Mks = prob.Mks * scale_info.g_scale

    scaleprob = AutoTuningProblem(
        P,
        w,
        prob.Ts*ωc,
        prob.Tf*ωc,
        prob.Ms,
        prob.Mt,
        Mks,
        prob.metric,
        prob.disc,
        pmax,
        prob.s_cache,
        prob.sge_cache,
        prob.ae_cache,
    )
    scaleprob, p, scale_info
end


"""
This init strategy is good unless there are double integrators, in which case the D part is needed.
"""
function init_strategy_loopshapingPI(prob, D=false)
    Ms = prob.Ms
    phasemargin = rad2deg(2asin(1/(2Ms)))
    gm = Ms/(Ms-1)
    P, _, w, scale_info = DyadControlSystems._scale_numerics(prob.P, nothing, prob.w)
    p0s = map([w[[1, end÷2, end]]; scale_info.ωc]) do wp
        C, kp, ki = loopshapingPI(P, wp; phasemargin, form=:parallel, doplot=false)
        p = D ? [kp, ki, 0, 0] : [kp, ki]
        _unscale_numerics(p, scale_info)
    end
    filter(p0->all(>=(0), p0), p0s)
end

function init_strategy_loopshapingPID(prob; extended = false)
    Mt = prob.Mt
    P, _, w, scale_info = DyadControlSystems._scale_numerics(prob.P, nothing, prob.w)
    # P,w = prob.P, prob.w
    
    if extended
        ws = w[1:(end÷20):end]
        ϕts = [1, 10, 45, 75, 85, 110]
    else
        ϕts = [5, 45, 80]
        ws = w[[end÷4, end÷2, 3end÷4]]
    end

    p0s = map(Iterators.product(ws, ϕts)) do (wp, ϕt)
        try
            C, kp, ki, kd = ControlSystemsBase.loopshapingPID(P, wp; Mt, form=:parallel, ϕt, verbose=false)
            p = [kp, ki, kd, 0]
            _unscale_numerics(p, scale_info)
        catch
            [-1.0]
        end
    end
    # if dcgain(P, 1e-10)[] > 0
        filter(p0->all(>=(0), p0[1:min(3, end)]), p0s)
    # else
    #     filter(p0->all(<=(0), p0[1:3]), p0s)
    # end
end
