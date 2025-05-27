import JuMP
using Hypatia
using ControlSystemsBase

"""
    (; K, P, cost) = common_lqr(systems::AbstractVector{<:AbstractStateSpace}, Q, R)
    (; K, P, cost) = common_lqr(systems::AbstractVector{<:ExtendedStateSpace})

Similar to [`lqr`](@ref), but finds a stabilizing controller for a vector of systems.

Returns the state-feedback gain `K`, the solution to the algebraic Riccati equation `P`, and the cost `cost`.

If the algorithm succeeds, ``xᵀPx`` is a common Lyapunov function for all `systems` in feedback with ``u = -Kx``.

# Keyword Arguments:
- `opt = Hypatia.Optimizer`: A JuMP compatible solver.
- `verbose = true`: Print termination status.
- `silent_solver = true`: Silence the output of the solver.
- `ϵ = 1e-6`: A small numerical value to enforce positive definiteness.

# Example:
See [Uncertainty-aware LQR](@ref)
"""
function common_lqr(systems::AbstractVector{<:AbstractStateSpace}, Q1, Q2; kwargs...)
    # partial references sec 9.1.1 http://maecourses.ucsd.edu/~mdeolive/mae280b/lecture/lecture9.pdf
    # or http://control.asu.edu/Classes/MiniCourse/L01_MINI.pdf slide 117

    sys = systems[1]

    (; nx, nu, ny) = sys
    esystems = map(systems) do sys
        T = ControlSystemsBase.numeric_type(sys)
        sQ1 = Matrix(sqrt(Q1)) # Q1 = C1'C1
        sQ2 = Matrix(sqrt(Q2)) # Q2 = D12'D12

        C1 = [
            sQ1
            zeros(T, nu, nx)
        ]
        D12 = [
            zeros(T, nx, nu)
            sQ2
        ]

        # Q1 = C'C
        # Q2 = D'D
        B1 = one(T)*I(nx) # B1

        ss(sys.A, B1, sys.B, C1, sys.C; D12, D22=sys.D, Ts=sys.timeevol)
    end
    common_lqr(esystems; kwargs...)
end

function common_lqr(
    systems::AbstractVector{<:ExtendedStateSpace};
    opt = Hypatia.Optimizer,
    balance = false,
    verbose = true,
    silent_solver = true,
    ϵ = 1e-6,
)
    sys = systems[1]
    (; nx, nu, ny) = sys
    n = nx+nu # size(C,1)
    model = JuMP.Model(opt)
    JuMP.set_optimizer_attribute(model, JuMP.MOI.Silent(), silent_solver)

    W = I # Covariance of w
    JuMP.@variable(model, P[1:nx,1:nx], PSD)
    JuMP.@variable(model, L[1:nu,1:nx])
    JuMP.@variable(model, Z[1:nx,1:nx], PSD)
    JuMP.@objective(model, Min, tr(Z*W))

    for sys in systems
        ControlSystemsBase.iscontinuous(sys) || error("common_lqr only support continuous-time systems")
        (; A, B1, B2, C1, D12) = sys
        M11 = A*P + P*A' + B2*L + L'B2'
        M21 = C1*P + D12*L
        M12 = M21'
        M22 = -I(n)
        M = [
            M11 M12;
            M21 M22;
        ]

        JuMP.@constraint(model, M <= -ϵ*I, PSDCone())
        JuMP.@constraint(model, [Z B1'; B1 P] >= ϵ*I, PSDCone())
    end


    JuMP.optimize!(model)
    if verbose
        @info JuMP.termination_status(model)
    end

    P = JuMP.value.(P)
    L = JuMP.value.(L)
    K = -L/P
    cost = JuMP.objective_value(model)
    (; K, P, cost)
end

using MonteCarloMeasurements: AbstractParticles, nparticles
using RobustAndOptimalControl: sys_from_particles

function ControlSystemsBase.lqr(sys::StateSpace{<:Any, <:AbstractParticles}, Q, R, args...; kwargs...)
    systems = sys_from_particles(sys)
    common_lqr(systems, Q, R, args...; kwargs...)[1]
end

function ControlSystemsBase.kalman(sys::StateSpace{<:Any, <:AbstractParticles}, Q, R, args...; kwargs...)
    N = nparticles(sys.A)
    systems = map(1:N) do i
        si = sys_from_particles(sys, i)
        A,B,C,D = ssdata(si)
        ss(A', C', B', D', si.timeevol)
    end
    Matrix(common_lqr(systems, Q, R, args...; kwargs...)[1]')
end


"""
    common_lyap(timeevolution, As::Vector{<:Matrix}, Q; opt, verbose = true, silent_solver = true, ϵ = 1.0e-6)
    common_lyap(Discrete,   As::Vector{<:Matrix}, Q; opt, verbose = true, silent_solver = true, ϵ = 1.0e-6)
    common_lyap(systems::Vector{StateSpace}, ...)
    common_lyap(sys::StateSpace{Particles}, ...)

Find ``P`` in the common quadratic Lyapunov function ``x^T P x`` that simultaneously solves the Lyapunov inequality
```math
A_i P + P A_i^T + Q \\preceq 0 \\; \\forall \\, i
```
for continuous-time systems, or
```math
A_i P A_i^T - P + Q \\preceq 0 \\; \\forall \\, i
```
for discrete-time systems, for all ``Aᵢ`` in ``As``. If the algorithm succeeds, ``xᵀPx`` is a Lyapunov function for all systems that can be written as a linear combination ``∑ᵢ αᵢ Aᵢ, \\; αᵢ ≥ 0``. The Lyapunov function also proves stability for the hybrid system implied by the switching between the ``Aᵢ``, even if the switching is infinitely fast. The result is sufficient but not necessary, implying that the set of systems may be robustly stable even though the algorithm fails to find a common Lyapunov function.

# Arguments:
- `timeevolution`: If matrices are passed as arguments, indicate if they represent a continuous time or discrete time system by passing either of `Continuous` or `Discrete`. If a vector of `StateSpace` objects is passed, the time evolution is inferred from the systems.
- `As`: A vector of matrices
- `systems / sys`: Instead of passing matrices, a vector of `StateSpace` objects or a single `StateSpace` object with `MonteCarloMeasurements.Particles` as coefficients can be passed. In this case, the matrices are extracted from the `StateSpace` objects.
- `Q`: A positive definite matrix
- `opt`: An SDP solver
- `verbose`: Print stuff?
- `silent_solver`: Set to true to silence output from the solver.

# Returns:
- `P`: The solution to the Lyapunov equation
- `termination_status`: The termination status of the solver, e.g., `OPTIMAL`. Has the type `MathOptInterface.TerminationStatusCode
- `primal_status`: The primal status of the solver, e.g., `FEASIBLE_POINT`. Has the type `MathOptInterface.ResultStatusCode`
- `dual_status`: The dual status of the solver. Has the type `MathOptInterface.ResultStatusCode`
- `model`: The JuMP model
"""
function common_lyap(
    TE,
    As::AbstractVector{<:AbstractMatrix}, Q;
    opt = Hypatia.Optimizer,
    verbose = true,
    silent_solver = true,
)
    A = As[1]
    nx = LinearAlgebra.checksquare(A)
    model = JuMP.Model(opt)
    JuMP.set_optimizer_attribute(model, JuMP.MOI.Silent(), silent_solver)

    JuMP.@variable(model, P[1:nx,1:nx], PSD)
    JuMP.@objective(model, Min, tr(Q*P))

    for A in As
        if TE isa ControlSystemsBase.ContinuousType
            JuMP.@constraint(model, A*P + P*A' + Q <= 0, PSDCone())
        else
            JuMP.@constraint(model, A*P*A' - P + Q <= 0, PSDCone())
        end
    end

    JuMP.optimize!(model)
    if verbose
        @info "Termination status: \t$(JuMP.termination_status(model))"
        @info "Primal status: \t\t$(JuMP.primal_status(model))"
        @info "Dual status: \t\t$(JuMP.dual_status(model))"
    end

    P = JuMP.value.(P)
    (; P, termination_status = JuMP.termination_status(model), primal_status = JuMP.primal_status(model), dual_status = JuMP.dual_status(model), model)
end

function common_lyap(systems::AbstractVector{<:AbstractStateSpace}, args...; kwargs...)
    As = transpose.(getproperty.(systems, :A))
    common_lyap(ControlSystemsBase.timeevol(systems[1]), As, args...; kwargs...)
end

function common_lyap(sys::StateSpace{TE, <:AbstractParticles}, args...; kwargs...) where TE
    As = map(1:nparticles(sys.A)) do i
        transpose(sys_from_particles(sys, i).A)
    end
    common_lyap(TE, As, args...; kwargs...)
end
