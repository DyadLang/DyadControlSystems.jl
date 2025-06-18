#=
- scale and unscale
- Does not work for delay systems due to NamedSS

=#

using RobustAndOptimalControl, ControlSystemsBase
using Optimization, Statistics, LinearAlgebra, StaticArrays, RecipesBase
using Ipopt, OptimizationMOI;
using ForwardDiff
# MOI = OptimizationMOI.MOI;

# """
# Scale the gain and frequency axis of P to be around 1.
# """
# function _scale_numerics(P0, p, w, Mks, Tf, Ts)
#     # return (P, p, w, (; g_scale=1, ωc=1))
#     P = deepcopy(P0)
#     P isa AbstractStateSpace || (return (P, p, w, Mks, Tf, Ts, (; g_scale=1, ωc=1)))
#     # return (P, p, w, (; g_scale=1, ωc=1))
#     ωc = 1 # w[end÷2] # Center frequency NOTE: this is deactivated since Mks constraints are not always satisfied when we rescale frequency axis. 
#     P = time_scale(P, 1/ωc)
#     ps = poles(P)
#     num_integ = count(abs(p) < 1e-5 for p in ps)
#     if num_integ == 0
#         g_scale = abs(dcgain(P)[])
#     else
#         g_scale = abs(freqresp(P, ωc)[])
#     end
#     P = (1/g_scale) * P
#     if p !== nothing
#         p = copy(p)
#         # Time scaling with ωc corresponds to dividing ki, kd, and Tf with ωc, 
#         p[1] *= g_scale
#         p[2] *= (g_scale/ωc)
#         length(p) >= 3 && (p[3] *= (g_scale*ωc))
#         length(p) == 4 && (p[4] *= ωc)

#     end
#     w = w ./ ωc
#     P = named_ss(P; P0.x, P0.u, P0.y)

#     Mks, Tf, Ts = Mks*g_scale, Tf*ωc, Ts*ωc

#     P, p, w, Mks, Tf, Ts, (; g_scale, ωc)
# end

# "The inverse of _scale_numerics"
# function _unscale_numerics(p, scaling_info)
#     g_scale, ωc = scaling_info
#     p = copy(p)
#     p[1] /= g_scale
#     p[2] /= g_scale / ωc
#     length(p) >= 3 && (p[3] /= ωc*g_scale)
#     length(p) >= 4 && (p[4] /= ωc)
#     p
# end

function initial_guess(prob)
    (; P, w, lb, ub) = prob
    Pfb = minreal(sminreal(P[prob.measurement, prob.control_input]))
    ints = ControlSystemsBase.integrator_excess(Pfb)


    with_kp = lb[1] < ub[1]
    with_ki = lb[2] < ub[2]
    with_kd = lb[3] < ub[3]
    with_Tf = lb[4] < ub[4]
    with_d = prob.optimize_d

    w_center = w[end÷3] # more conservative than w[end÷2]
    igain_center = 1/abs(freqresp(Pfb, w_center)[])
    if with_kp
        kp = clamp(igain_center, lb[1], ub[1]) # This will aproximately set the gain cross-over frequency to the center frequency
    else
        kp = 0.0
    end

    ki = clamp(0.1*igain_center, lb[2], ub[2])

    kd = if !with_kd
        0.0
    elseif ints == 2
        clamp(igain_center, lb[3], ub[3])
    elseif ints >= 3
        @error("The system has more than 2 integrators, a PID controller will not stabilize the system")
        clamp(igain_center, lb[3], ub[3])
    else
        clamp(0.1*igain_center, lb[3], ub[3])
    end

    Tf = if !with_Tf
        0.0
    else
        # Ensure LP filter breakdown within frequency grid
        # Alternatively, pick this so that one of the constraints is satisfied?
        Tf = clamp(1 / w[end], lb[4], ub[4])
    end

    if with_d
        d = clamp(0.95, lb[5], ub[5]) # 0.9 appears to be generally better than 1/√2, most benchmark problems convered to >0.99, with the smallest one being 0.94
        [kp, ki, kd, Tf, d]
    else
        [kp, ki, kd, Tf]
    end


end

function reducedim(P)
    if P.nx > 10
        @warn "The model is of high order ($(P.nx)-dimensional state). The model will be automatically reduced to no more than 10 state variables using residualized balanced truncation (`baltrunc(minreal(P), n=10, residual=true)`). To turn this off, pass `reduce=false` to `AutoTuningProblem2`"
        P = minreal(P, 1e-6)
        # ωc = w[end÷2]
        # P = time_scale(P, 1/ωc)
        if P.nx > 10
            P.A .+= 1e-10I(P.nx) # This is safe since the time scale is adjusted
            P, _ = isstable(P) ? Core.Compiler.inferencebarrier(baltrunc2)(P, n=10, residual=true) : Core.Compiler.inferencebarrier(baltrunc_unstab)(P, n=10, residual=true)
        end
        # P = time_scale(P, ωc)
    end
    P
end

# function static_pid(p::AbstractArray{T}) where T
#     kp, ki, kd, Tf = p
#     if Tf <= 0
#         @assert kd == 0 "kd must be zero if Tf is zero"
#         if ki == 0
#             return HeteroStateSpace(@SMatrix(zeros(0,0)), @SMatrix(zeros(0,1)), @SMatrix(zeros(1,0)), SA[kp;;])
#         else
#             A = SA[0.0;;]
#             B = SA[1.0;;]
#             C = SA[ki;;] # Ti == 0 would result in division by zero, but typically indicates that the user wants no integral action
#             D = SA[kp]
#         end
#     else
#         Tf2 = Tf*Tf
#         if ki == 0
#             A = SA[0 1; -2/Tf^2 -2/Tf]
#             B = SA[0; 1.0;;]
#             C = 2 / Tf2 * SA[kp kd]
#         else
#             A = SA[ 0 1 0
#                     0 0 1
#                     0 -(2 / Tf2) -(2 / Tf)]
#             B = SMatrix{3,1,Float64,3}(0.0, 0, 1)

#             C = SA[2ki/Tf2 2kp/Tf2 2kd/Tf2]
#         end
#     end
#     D = SMatrix{1, 1, Float64, 1}(0.0)
#     HeteroStateSpace(A, B, C, D)
# end

function static_pid(p::AbstractArray{T}; filter_order=2) where T
    Kp, Ki, Kd, Tf = p
    if Tf > 0
        d42 = length(p) < 5 ? 2.0 : 4p[5]^2
        if Ki == 0
            if filter_order == 1
                A = SA[-1 / Tf;;]
                B = SA[-Kd/Tf^2]
                C = SA[1.0;;]
                D = SA[Kd/Tf + Kp;;]
            else # 2
                A = SA[0 1; -d42/Tf^2 -d42/Tf]
                B = SA[0; 1]
                C = d42 / Tf^2 * SA[Kp Kd]
                D = SA[0.0;;]
            end
        else
            if filter_order == 1
                A = SA[0 0; 0 -1/Tf]
                B = SA[Ki; -Kd/Tf^2]
                C = SA[1.0 1]
                D = SA[Kd/Tf + Kp;;]
            else # 2
                A = SA[0 1 0; 0 0 1; 0 -d42/Tf^2 -d42/Tf]
                B = SA[0.0; 0; 1]
                C = d42 / Tf^2 * SA[Ki Kp Kd]
                D = SA[0.0;;]
            end
        end
    else
        if Ki != 0
            A = SA[0.0;;]
            B = SA[1.0;;]
            C = SA[Ki;;]
            D = SA[Kp;;]
        else
            # return HeteroStateSpace(@SMatrix(zeros(0,0)), @SMatrix(zeros(0,1)), @SMatrix(zeros(1,0)), SA[Kp;;])
            return ss(SA[Kp])
        end
    end
    # HeteroStateSpace(A, B, C, D)
    ss(A,B,C,D)
end


function trapz2(res)
    t = res.t
    x = res.y
    length(t) == size(x, 2)
    s = zero(eltype(x))
    if size(x, 1) == 1
        @inbounds for i = 1:length(t)-1
            s += (t[i+1] - t[i]) * (x[i+1] + x[i])/2
        end
    else
        for k = 1:size(x, 3)
            for i = 1:length(t)-1
                for j = 1:size(x, 1)
                    @inbounds s += (t[i+1] - t[i]) * (x[j,i+1,k] + x[j,i,k])/2
                end
            end
        end
    end
    s
end

"""
    AutoTuningProblem2(P::NamedStateSpace; kwargs...)

A problem representing the automatic tuning of a PID controller with filter on the form
```
K(s) = C(s) F(s) = (k_p + k_i/s + k_d s)  \\dfrac{1}{(sT)^2/(4d^2) + Ts + 1}
```
where ``d = 1`` by default.

# Keyword Arguments
- `w::Vector{Float64}`: Frequency vector for the optimization
- `Ts::Float64`: Sampling time
- `Tf::Float64`: Duration of simulation (final time)
- `measurement::Union{Symbol, Vector{Symbol}}`: The measured output of `P` that is used for feedback
- `control_input::Union{Symbol, Vector{Symbol}}`: The control input of `P`
- `step_input::Union{Symbol, Vector{Symbol}}`: The input to the system when optimizing the step response. Defaults to `control_input`
- `step_output::Union{Symbol, Vector{Symbol}}`: The output to the system when optimizing the step response. Defaults to `measurement`
- `response_type::Function`: A function on the form `(::StateSpace, time_vector) -> ::SimResult`. Defaults to `step`
- `Ms::Union{Float64, Vector{Float64}}`: Maximum allowed peak in the sensitivity funciton. Defaults to `Inf` for no constraint.
- `Mt::Union{Float64, Vector{Float64}}`: Maximum allowed peak in the complementary sensitivity function. Defaults to `Inf` for no constraint.
- `Mks::Union{Float64, Vector{Float64}}`: Maximum allowed peak in the noise sensitivity function. Defaults to `Inf` for no constraint.
- `metric::Function`: The cost function to minimize. Defaults to `abs2`
- `ref::Union{Float64, Array{Float64}}`: The reference signal for the response optimization. Defaults to `0.0`. If a multivariate response is optimized, this should be a matrix of size `(ny, nT, nu)` where `ny` is the number of outputs, `nT` is the number of time steps, and `nu` is the number of inputs.
- `disc::Symbol`: The discretization method. Defaults to `:tustin`
- `lb::Vector{Float64}`: Lower bounds for the optimization. A vector of the same length and layout as the parameter vector. Defaults to `[0.0, 0, 0, 1e-16]`.
- `ub::Vector{Float64}`: Upper bounds for the optimization. A vector of the same length and layout as the parameter vector. Defaults to `[Inf, Inf, Inf, Inf]`.
- `timeweight::Bool`: If `true`, time-weighted error is used as the cost function. Defaults to `false`. Set this to true to increase the cost for large errors at the end of the simulation.
- `autodiff`: The automatic differentiation method to use. Defaults to `Optimization.AutoForwardDiff()`
- `reduce::Bool`: If `true`, the state-space model is reduced to at most order 10 using balanced truncation. Defaults to `true`
- `filter_order::Int`: The order of the filter. Options are `{1, 2}`. Defaults to `2`.
- `optimize_d::Bool`: If `true`, the filter damping ratio is optimized. Defaults to `false`. if this is set to true, the parameter vector should have 5 elements, where the last element is the damping ratio. The default lower bound is `1/√2` and the default upper bound is `1`. Upper bounds greater than 1 are allowed to allow for a wider peak in the filter response.

See [autotuning documentation](https://help.juliahub.com/DyadControlSystems/stable/autotuning2/) for more details.

See also [`OptimizedPID2`](@ref)
"""
struct AutoTuningProblem2{S, W, RS, M, AD}
    P::S
    w::W
    measurement::Union{Symbol, Vector{Symbol}}
    control_input::Union{Symbol, Vector{Symbol}}
    step_input::Union{Symbol, Vector{Symbol}}
    step_output::Union{Symbol, Vector{Symbol}}
    response_type::RS
    Ts::Float64
    Tf::Float64
    Ms::Union{Float64, Vector{Float64}}
    Mt::Union{Float64, Vector{Float64}}
    Mks::Union{Float64, Vector{Float64}}
    metric::M
    ref::Union{Float64, Array{Float64}}
    disc::Symbol
    lb::Vector{Float64}
    ub::Vector{Float64}
    timeweight::Bool
    autodiff::AD
    reduce::Bool
    filter_order::Int
    optimize_d::Bool
    # ucons::Vector{Float64}
    # lcons::Vector{Float64}
    # cl_outputs::Vector{Symbol}
end

function AutoTuningProblem2(
    P::NamedStateSpace;
    w = ControlSystemsBase._default_freq_vector(P, Val{:bode}()),
    measurement = length(P.y) == 1 ? P.y[] : error("Process model has multiple outputs, specify which is the measured output that is used for feedback"),
    control_input = length(P.u) == 1 ? P.u[] : error("Process model has multiple inputs, specify which is the control input connected to the controller"),
    step_input = control_input,
    step_output = measurement,
    response_type = step,
    Ts,
    Tf,
    Ms = Inf,
    Mt = Inf,
    Mks = Inf,
    filter_order = 2,
    metric = abs2,
    ref = step_input === :reference_input ? 1.0 : 0.0,
    disc = :tustin,
    ub = fill(Inf, 4),
    lb = [0.0, 0, 0, 1e-16], # Filter Tf may not be too small
    timeweight = false,
    autodiff = Optimization.AutoForwardDiff(),
    reduce = true,
    optimize_d = false,
)
    if ub[3] > 0 && ub[4] <= 0
        error("Filter time constant must be positive in the presence of derivative action, you provided a lower bound of Tf=$(lb[4]) and a derivative upper bound of $(ub[3])")
    end
    if ub[3] == lb[3] == 0 && step_input != :reference_input
        # With no derivative gain and no reference input the filter has no effect and we avoid optimizing it to avoid numerical problems
        lb[4] == 0 || @warn "The derivative gain is zero and a disturbance response is optimized, the filter will not have any effect on the optimization and is not included."
        lb[4] = ub[4] = 0 # No filter
    end
    if length(ub) >= 5
        ub[5] ≥ 0 || error("The filter damping ratio must be greater than 0 (typically <= 1), but got $(ub[5])")
        lb[5] ≥ 0 || error("The filter damping ratio must be greater than 0 (typically <= 1), but got $(lb[5])")
        # The rationale for allowing > 1 is that this allows the filter to have a wider peak (two different real poles). This approximates the first-order filter, which typically gives higher performance but also higher noise gain. Most benchmarks converge to close to the upper limit used, but this tends to make the PID controller less of a PID controller and more of a generic lead-lag controller with two very different filter poles, the default chosen below is this to upper bound to 1
    elseif optimize_d
        ub = [ub; 1]
        lb = [lb; 1/√(2)]
    end

    P = sminreal(P)
    if reduce
        reduce && (P = reducedim(P))
    end
    nu = step_input isa AbstractVector ? length(step_input) : 1
    ny = step_output isa AbstractVector ? length(step_output) : 1
    if nu > 1 || ny > 1
        nT = length(0:Ts:Tf)
        length(ref) == 1 && 
            error("The reference is scalar but more than one step input is provided. If optimizing both reference and disturbance responses, you must provide references for each response.")
        if ref isa AbstractVector && (nu == 1 || ny == 1)
            R = zeros(ny, nT, nu)
            if nu == 1
                R[:, :, 1] .= ref
            else # ny == 1
                R[1, :, :] .= ref'
            end
            ref = R
        elseif ref isa AbstractVector
            error("Expected reference to be a ($ny, $nu) matrix, but got a vector of length $(length(ref))")
        else
            R = zeros(ny, nT, nu)
            for i = 1:ny, j = 1:nu
                R[i, :, j] .= ref[i, j]
            end
            ref = R
        end
    end
    # ref ny nT nu

    AutoTuningProblem2(
        P,
        w,
        measurement,
        control_input,
        step_input,
        step_output,
        response_type,
        float(Ts),
        float(Tf),
        float(Ms),
        float(Mt),
        float(Mks),
        metric,
        float(ref),
        disc,
        float.(lb),
        float.(ub),
        timeweight,
        autodiff,
        reduce,
        filter_order,
        optimize_d,
        # ucons,
        # lcons,
        # cl_outputs,
    )
end

AutoTuningProblem2(P; kwargs...) = AutoTuningProblem2(named_ss(ss(P)); kwargs...)
function AutoTuningProblem2(P::DelayLtiSystem; kwargs...)
    @info "Approximating delay with a Pade approximation of order 3"
    AutoTuningProblem2(pade(P, 3); kwargs...)
end

function Optimization.solve(prob::AutoTuningProblem2, params = initial_guess(prob), solver = IpoptSolver(exact_hessian=false, mu_strategy="adaptive"); kwargs...)
    (:lb ∈ keys(kwargs) || :ub ∈ keys(kwargs)) && @warn("Overriding the parameter bounds when calling solve is strongly discouraged, it bypasses validation and can lead to incorrect results. Create the AutoTuningProblem2 with the desired bounds instead.")
    C, G, sol, timeres, costscaling = autotune(
        prob.P;
        prob.w,
        prob.measurement,
        prob.control_input,
        prob.step_input,
        prob.step_output,
        prob.response_type,
        prob.Ts,
        prob.Tf,
        prob.Ms,
        prob.Mt,
        prob.Mks,
        prob.filter_order,
        prob.optimize_d,
        prob.metric,
        prob.ref,
        prob.disc,
        prob.lb,
        prob.ub,
        prob.timeweight,
        params,
        prob.autodiff,
        solver,
        kwargs...
    )
    AutoTuningResult2(prob, sol.u, C, sol, sol.minimum/costscaling, timeres, G)
end

"""
    AutoTuningResult2

A structure containing the results of performing PID autotuning by means of [`AutoTuningProblem2`](@ref).

This structure can be plotted `plot(res)`. See also [`OptimizedPID2`](@ref).

# Fields:
- `prob::AutoTuningProblem`
- `p::Vector`: The optimal parameter vector `[kp, ki, kd, Tf]` for a PID controller on parallel form displayed below, or if `optimize_d = true`, `p = [kp, ki, kd, Tf, d]` where ``d = ζ`` is the damping ratio of the filter
- `K::NamedStateSpace`: The optimal controller with named input `:error_input_C` and output `:u_controller_output_C`

```math
K(s) = C(s)F(s) = (k_p + k_i/s + k_d s) 1/(sT)^2/(4ζ^2) + Ts + 1), \\quad ζ = 1
```
"""
struct AutoTuningResult2{KT}
    prob::AutoTuningProblem2
    "Optimized parameters as a vector `[kp, ki, kd, Tf]` for a PID controller on parallel form ``K(s) = C(s)F(s) = (k_p + k_i/s + k_d s) 1/(sT)^2/(4ζ^2) + Ts + 1), ζ = 1``"
    p::Vector
    "The optimized controller in the form of a NamedStateSpace"
    K::KT
    "Optimization.jl solution object"
    sol
    "Optimal cost"
    cost
    "Time taken to solve the problem"
    timeres
    "Constrained closed-loop transfer functions"
    G
end

function autotune(
    P::AbstractStateSpace;
    w,
    measurement,
    control_input,
    step_input = control_input,
    step_output = measurement,
    response_type = step,
    Ts,
    Tf,
    Ms,
    Mt,
    Mks,
    filter_order = 2,
    optimize_d = false,
    metric = abs2,
    ref,
    disc = :tustin,
    lb = zeros(4),
    ub = fill(Inf, 4),
    timeweight = false,
    params,
    autodiff = Optimization.AutoForwardDiff(),
    solver = IpoptSolver(),
    scale = true,
    kwargs...
    # ucons,
    # lcons,
    # cl_outputs,
)
    optimize_d = optimize_d && filter_order == 2
    if length(params) < 3
        error("Expected at least 3 parameters, but got $(length(params))")
    elseif optimize_d && length(params) < 5
        params = [params; 1.0] # If 1.0 default is changed the OptimizedPID2 function must be updated
    elseif !(optimize_d || length(params) == 4)
        error("Expected 4 parameters [kp, ki, kd, Tf], but got $(length(params))")
    end
    # The costscaling factor below is a heuristic numerical scaling. If a large gain is required it typically means that the plant gain is low, and the integral of the cost will thus be a small number.
    scaling_input = if step_input === :reference_input || (step_input isa AbstractArray && :reference_input ∈ step_input)
        control_input
    else
        step_input
    end
    scaling_output = (step_output isa AbstractArray && length(step_output) > 1) ? measurement : step_output

    # @show scaling_input, scaling_output
    costscaling = scale ? 1/exp(mean(log.(abs.(freqrespv(P[scaling_output, scaling_input], w))))) : 1.0

    tvec = 0:Ts:Tf

    workspaceF = nothing
    workspaceD = nothing

    external_inputs = unique([
        control_input
        step_input
        :reference_input
    ])

    external_outputs = unique([
        :error_input_C
        measurement
        :u_controller_output_C
        step_output
    ])
    connections = [
        :error_input_C => :error_input_C
        :u_controller_output_C => control_input
        measurement => :y_measurement_feedback
    ]
    feedback_node = sumblock("error_input_C = reference_input - y_measurement_feedback")


    function systemspid2(params, P)
        # C = named_ss(pid(kp, ki, kd; form = :parallel, Tf, state_space = true), :controller, u = :error_input_C, y = :u_controller_output_C)
        C = named_ss(static_pid(params; filter_order), :controller, u = :error_input_C, y = :u_controller_output_C)
        sysvec = [P, C, feedback_node]
        G = RobustAndOptimalControl.connect(sysvec, connections; external_inputs, external_outputs, verbose=false)
        C, G
    end

    function sim(G)
        Gd = c2d(G[step_output, step_input], Ts, disc)   # Discretize the system
        response_type(Gd.sys, tvec) # Simulate the step response
    end

    function cost(params::AbstractVector{T}, P) where {T}
        try
            _, Gc = systemspid2(params, P)
            res = sim(Gc)
            y = res.y
            y .-= ref
            y .= metric.(y)
            if timeweight            # frequency-weighted noise sensitivity
                y .*= res.t'
            end
            val = trapz2(res)
            return costscaling*val
        catch
            return T(Inf)
        end
    end

    function constraints(out, params::AbstractVector{T}, P) where {T}
        
        Gcon = try
            _, Gcon = systemspid2(params, P)
            Gcon
        catch
            return nothing
        end
        # S = r -> e
        # T = r -> measurement or 1-S
        # CS = r -> u
        Gfreq = Gcon[cl_outputs, :reference_input]
        # if Gfreq.nx <= 10
        # # This may encounter setindex! error in bodemag_nohess! 
        #     Gfreq = StaticStateSpace(Gfreq)
        # end
        if T <: ForwardDiff.Dual
            if workspaceD === nothing
                workspaceD = BodemagWorkspace(Gfreq, w)
            end
            F = ControlSystemsBase.bodemag_nohess!(workspaceD::BodemagWorkspace{T}, Gfreq, w)
        else
            if workspaceF === nothing
                workspaceF = BodemagWorkspace(Gfreq, w)
            end
            F = ControlSystemsBase.bodemag_nohess!(workspaceF::BodemagWorkspace{T}, Gfreq, w)
        end

        out[1:end-1] .= vec(F) # max sensitivity

        # Ensure filter breakdown after derivative gain increase
        kp, ki, kd, Tf = params
        s0 = kp + sqrt(max(kp^2 - 4ki*kd, 1e-10one(kd))) # Don't use zero here in case kp is zero, the gradient will be NaN
        cTf = Tf*abs(s0) - 2kd
        out[end] = cTf

        nothing
    end

    cl_outputs = Symbol[]
    any(isfinite, Ms) && push!(cl_outputs, :error_input_C)
    any(isfinite, Mt) && push!(cl_outputs, measurement)
    any(isfinite, Mks) && push!(cl_outputs, :u_controller_output_C)
    n_con_outputs = length(cl_outputs)

    # Build ucons
    ucons = zeros(n_con_outputs, length(w))
    ind = 1
    if any(isfinite, Ms)
        ucons[ind, :] .= Ms
        ind += 1
    end
    if any(isfinite, Mt)
        ucons[ind, :] .= Mt
        ind += 1
    end
    if any(isfinite, Mks)
        ucons[ind, :] .= Mks
    end
    ucons = [vec(ucons); 0] # One extra zero for the filter breakdown constraint
    lcons = [fill(-Inf, n_con_outputs*length(w)); -Inf]

    fopt = OptimizationFunction(cost, autodiff; cons = constraints)
    prob = OptimizationProblem(
        fopt,
        params,
        P;
        lb,
        ub,
        ucons,
        lcons,
    )
    timeres = @elapsed sol = solve(prob, solver; kwargs...)
    sol.u .= clamp.(sol.u, lb, ub)
    C, G = systemspid2(sol.u, P)
    # isstable(G) || @error("The closed-loop system is not stable")

    # AutoTuningResult2(prob, sol.u, C, sol, sol.minimum, timeres, G)
    C, G, sol, timeres, costscaling
end

"""
Convert from ``1 / ((sT_f)^2/(4d^2) + sT_f + 1)`` to ``ω^2 / (s^2 + 2ζω s + ω^2)``
"""
function Tfd2w(Tf, d)
    0 ≤ d ≤ 2 || error("The damping ratio must be between 0 and 2, but got $d")
    ω = 2d/Tf
    ω
end


"""
    OptimizedPID2(res::AutoTuningResult2; name, kwargs...)

Takes optimized parameters `popt` or an [`AutoTuningResult2`](@ref) and returns the following system (`filter_order = 2`)
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
or if `filter_order = 1`, the built in filter on the derivative term present in `Blocks.LimPID`` is used. The connectors `reference, measurement, ctr_output` are the same in both cases.


# Arguments:
- `popt`: Obtained by solving an [`AutoTuningProblem2`](@ref)
- `kwargs`: Are passed to `ModelingToolkitStandardLibrary.Blocks.LimPID`
"""
function OptimizedPID2(popt::Vector; name, filter_order = 2, Nd = 10000, kwargs...) # Nd is set to a very high value since we have separate noise filters
    # We set Nd very high by default since we have an explicit filter F.
    kp, ki, kd, Tf = popt
    # (kp + ki/s + kd*s)
    k, Ti, Td = ControlSystemsBase.convert_pidparams_to_standard(kp, ki, kd, :parallel)
    if Tf == 0
        return Blocks.LimPID(; k, Ti, Td, Nd, name, kwargs...)
    end
    if filter_order == 1
        Nd = Td/Tf
        # In this case the correct form of the filter is already present in the PID controller
        return Blocks.LimPID(; k, Ti, Td, Nd, name, kwargs...)
    end

    d = length(popt) >= 5 ? popt[5] : 1.0
    w = Tfd2w(Tf, d)

    @named pid = Blocks.LimPID(; k, Ti, Td, Nd, kwargs...)
    @named filter_r = Blocks.SecondOrder(; w, d, k=1)
    @named filter_y = Blocks.SecondOrder(; w, d, k=1)
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

OptimizedPID2(res::AutoTuningResult2; kwargs...) = OptimizedPID2(res.p; res.filter_order, kwargs...)

@recipe function plot(res::AutoTuningResult2; stepsize = 1)
    prob = res.prob
    (; Ts, Tf, disc, Ms, Mt, w, P) = prob
    C = res.K
    tv = 0:Ts:Tf

    Pfb = P[prob.measurement, prob.control_input]

    Pus = res.G[:u_controller_output_C, prob.step_input]

    function sim()
        Gd = c2d(res.G[prob.step_output, prob.step_input], Ts, disc)   # Discretize the system
        prob.response_type(Gd.sys, tv) # Simulate the step response
    end

    function sim_u()
        Gd = c2d(Pus, Ts, disc)   # Discretize the system
        prob.response_type(Gd.sys, tv) # Simulate the step response
    end

    S, PS, CS, T = ControlSystemsBase.gangoffour(Pfb.sys, C.sys)
    # layout --> 5
    layout --> @layout [[a b; c d] e{0.3w}]
    size --> (800,600)
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
        title --> "Sensitivity functions"
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
        label --> "KS"
        w, b
    end
    @series begin
        xscale --> :log10
        yscale --> :log10
        b, _ = bode(C,w,unwrap=false)
        b = vec(b)
        label --> "K"
        w, b
    end
    @series begin
        title --> "Noise sensitivity and controller (KS / K)"
        seriestype --> :path
        linestyle --> :dash
        linecolor --> 1
        label --> "Mks"
        primary --> false
        collect(extrema(w)), [prob.Mks, prob.Mks]
    end
    subplot := 3
    # All series are split up into separate @series blocks due to https://github.com/JuliaPlots/Plots.jl/issues/4108
    step1 = sim() |> deepcopy # Deepcopy due to dstep using cached workspaces
    step2 = sim_u() |> deepcopy

    for i in eachindex(vcat(prob.step_input))
        for j in eachindex(vcat(prob.step_output))
            @series begin
                title --> "Optimized response"
                xguide := "Time [s]"
                # link := :none
                # linecolor --> 1
                label := "$(vcat(prob.step_input)[i]) → $(vcat(prob.step_output)[j])"
                step1.t, step1.y[j,:,i]
            end
        end
    end

    subplot := 4
    for i in eachindex(vcat(prob.step_input))
        @series begin
            link := :none
            xguide := "Time [s]"
            label := "$(vcat(prob.step_input)[i]) → u"
            step2.t, step2.y[i, :]
        end
    end
    cs = -1             # Ms center
    rs = 1 / Ms         # Ms radius
    ct = -Mt^2/(Mt^2-1) # Mt center
    rt = Mt/(Mt^2-1)    # Mt radius
    θ = range(0, stop=2π, length=100)
    Sin, Cos = sin.(θ), cos.(θ)
    re, im = nyquist(Pfb*C,w)
    subplot := 5
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
        # xguide := "Re"
        # yguide := "Im"
        ratio --> 1
        framestyle := :zerolines
        title --> "Nyquist plot"
        xlims --> (-3, 1)
        ylims --> (-5, 2)
        [-1], [0]
    end
    nothing
end


function check_feasibility(prob; p0 = nothing, τ, verbose = true)
    P = prob.P[prob.measurement, prob.control_input]
    ps = poles(P)
    zs = tzeros(P)
    int = ControlSystemsBase.count_integrators(P)
    int >= 3 && error("The plant contains $int integrators and cannot be stabilized with a PID controller (a PID controller can only lift the phase by at most 90 degrees)")
    rhpp = filter(p->real(p) > 0, ps)
    rhpz = filter(z->real(z) > 0, zs)
    p = maximum(abs, rhpp)
    z = maximum(abs, rhpz)
    if length(rhpp) > 1 && length(rhpz) > 1
        verbose && @info "RHP pole and zero requires z > 4p which is $(z > 4p)"
        verbose && @info "Ms and Mt are always larger than |(p+z) / (p-z)| = $(abs((p+z) / (p-z))) due to RHP pole and zero pair"
        prob.Ms > abs((p+z) / (p-z)) || error("Ms cannot be smaller than abs((p+z) / (p-z)) for unstable pole-zero pair p,z")
        prob.Mt > abs((p+z) / (p-z)) || error("Mt cannot be smaller than abs((p+z) / (p-z)) for unstable pole-zero pair p,z")
    end
    if τ > 0 && length(rhpp) > 0
        verbose && @info "RHP pole and delay τ > 0 limits Ms and Mt ≥ exp(p*τ) = $(exp(p*τ))"
        prob.Ms > exp(p*τ) || error("Ms cannot be smaller than exp(p*τ) for plant with unstable pole p and delay τ")
        prob.Mt > exp(p*τ) || error("Mt cannot be smaller than exp(p*τ) for plant with unstable pole p and delay τ")        
    end
    # TODO: triple integrator -> infeasible with PID controller
    # TODO: use upper and lower fundamental limitations to pick automatic frequency grid
end

