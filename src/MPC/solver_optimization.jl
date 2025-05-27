##
using Optimization, SparseArrays
using OptimizationMOI
using Ipopt
const MOI = OptimizationMOI.MOI
using ForwardDiff
using Printf

using Symbolics
using Symbolics.SymbolicUtils
using Symbolics.SymbolicUtils.Code
using Symbolics: variable, variables, unwrap, term
using SparseArrays
# The two lines below are defined so that unwrap.(sparse_mat) does not throw ypeError: non-boolean (SymbolicUtils.Term{Bool, Nothing}) used in boolean context
types = Union{SymbolicUtils.Term, SymbolicUtils.Add, SymbolicUtils.Div}
SparseArrays.HigherOrderFns._iszero(t::types) = isequal(0, t)



#=
About costs and constraints:
There are two major types, stage and trajectory.
- Stage takes in      x[t], u[t], p, t
- Trajectory takes in x[:], u[:], p, 1:T
- An objective consists of a running cost and a terminal cost

Ideas:
- Variables is the canonical source of truth, only this structure is allowed to be mutated


Questions:
- How do we handle varmap_to_vars when the user has provided symbolic constraints? Build explicit observed?
- Symbolics is handled in an outer layer that writes functions for the function interface
- Do we use slack variables? If so, who is responsible for them? This adds variables and must thus be part of the interface if we use them.
- Who is responsible for making constraints soft? The user, us or the optimizer?
    - Softness can be an outer layer or option to convenience constructors.
- Where do references and other known external signals enter? They need to appear in the cost.
    - do we categorize signals like r,d,w etc.? or the user is responsible for them?
    - Should we lump r and w together to represent all external signals?`
    - The dynamics may need to accept w as well
=#

default_autodiff(disc::MultipleShooting) = AutoSparseForwardDiff()
default_autodiff(disc::CollocationFinE) = AutoSparseForwardDiff() # AutoSymbolics()
default_autodiff(disc::Trapezoidal) = AutoSparseForwardDiff() # AutoSymbolics()

default_sym_lag_h(disc::MultipleShooting) = false
default_sym_lag_h(disc::CollocationFinE) = true
default_sym_lag_h(disc::Trapezoidal) = true


function unscale!(vars, scale_x, scale_u)
    for ri in 1:vars.n_robust
        x, u = get_xu(vars, nothing, ri)
        x .*= scale_x
        u .*= scale_u
    end
    nothing
end

function scale!(vars, scale_x, scale_u)
    for ri in 1:vars.n_robust
        x, u = get_xu(vars, nothing, ri)
        x ./= scale_x
        u ./= scale_u
    end
    nothing
end

function default_objective_input(dynamics::FunctionSystem, p0, x0, xr, N, Ts)
    if !ControlSystemsBase.isdiscrete(dynamics)
        dynamics = rk4(dynamics, Ts, supersample=10)
    end

    p1 = get_parameter_index(p0, 1)
    p = get_system_parameters(p1)

    nx, nu = dynamics.nx, dynamics.nu
    x = zeros(nx, N+1)
    x[:, 1] .= x0
    u = zeros(nu, N)
    MPC.rollout!(dynamics, x, u, p, 0)
    ObjectiveInput(x, u, xr)
end

struct GenericMPCProblem{F,O,C,S,P,OI,XR,OP,OP0,V,SX,SU} <: MPC.AbstractMPCProblem
    dynamics::F
    observer::O
    constraints::C
    solver::S
    N::Int
    Ts::Float64
    p::P
    objective_input::OI
    xr::XR
    optprob::OP
    optprob0::OP0
    vars::V
    scale_x::SX
    scale_u::SU
end

function Base.getproperty(prob::GenericMPCProblem, s::Symbol)
    s ∈ fieldnames(typeof(prob)) && return getfield(prob, s)
    if s === :nx
        return prob.dynamics.nx
    elseif s === :nu
        return prob.dynamics.nu
    elseif s === :ny
        return prob.observer.ny
    elseif s === :nz
        return prob.dynamics.nz
    elseif s === :nv
        return length(prob.constraints.min)
    elseif s === :Ts
        return prob.dynamics.Ts
    elseif s === :ur
        return zeros(prob.dynamics.nu)
    else
        throw(ArgumentError("GenericMPCProblem has no property named $s"))
    end
end

function Base.show(io::IO, mime::MIME"text/plain", prob::GenericMPCProblem{<:FunctionSystem{TE}}) where TE
    println(io, "GenericMPCProblem with:")
    @printf(io, " nx: %5d\n", prob.nx)
    @printf(io, " nu: %5d\n", prob.nu)
    @printf(io, " ny: %5d\n", prob.ny)
    @printf(io, " N:  %5d\n", prob.N)
    @printf(io, " Ts: %5.2g\n", prob.Ts)
    @printf(io, " solver:   %s\n", typeof(prob.solver))
    @printf(io, " dynamics: FunctionSystem{%s}\n", TE)
    @printf(io, " observer: %s\n", Base.nameof(typeof(prob.observer)))
    @printf(io, " discretization: %s\n", prob.constraints.constraints[2])
end

"""
    IpoptSolver(;
        verbose                    = false,
        printerval                 = 1,          # print this often if verbose
        tol                        = 0.0001,
        acceptable_tol             = 0.1,
        max_iter                   = 100,
        max_wall_time              = 180.0,
        max_cpu_time               = 2*max_wall_time,
        constr_viol_tol            = 0.0001,
        acceptable_constr_viol_tol = 0.1,
        acceptable_iter            = 5,
        exact_hessian              = true,
        mu_init                    = 0.1,
        mu_strategy                = "monotone", # can also be "adaptive" if problem has convergence issues
        lqg                        = false,      # Indicate that the problem has linear dynamics and constraints, with quadratic cost
        linear_inequality_constraints = false,   # Indicate that the problem has linear inequality constraints
        linear_system              = false,      # Indicate that the problem has linear dynamics and constraints
    )

A convenience constructor to create an `solver = Ipopt.Optimizer()` and set options.
See https://coin-or.github.io/Ipopt/OPTIONS.html for information about each option. The defaults provided here are more relaxed than Ipopt defaults.

Ipopt will try to meet `tol` and `constr_viol_tol`, but stops early if it has met `acceptable_tol` and `acceptable_constr_viol_tol` for `acceptable_iter` number of iterations.

For large problems, Ipopt may use multiple threads, in which case the argument `max_wall_time` is representative of the time the user experiences, while `max_cpu_time` is better set very large.

!!! note
    When solving MPC problems, it is often beneficial to favor a faster sample rate and a longer prediction horizon over accurate integration and optimization. The motivations for this are several

    - The dynamics model is often inaccurate, and solving an inaccurate model to high accuracy can be a waste of effort.
    - The performance is often dictated by the disturbances acting on the system, and having a higher sample rate may allow the controller to reject disturbances faster.
    - Feedback from measurements corrects for slight errors due to integration.
    - Increasing sample rate leads to each subsequent optimization problem being closer to the previous one, making warm-staring more efficient and a good solution being found in fewer iterations.

The `verbose` option can be a Bool or an integer, `true` is interpreted as the default Ipopt verbosity of 5.
"""
function IpoptSolver(;
        verbose                    = false,
        printerval                 = 1,
        tol                        = 1e-4,
        acceptable_tol             = 1e-1,
        max_iter                   = 500,
        max_wall_time              = 180.0,
        max_cpu_time               = 2*max_wall_time,
        constr_viol_tol            = 1e-4,
        acceptable_constr_viol_tol = 1e-1,
        acceptable_iter            = 5,
        exact_hessian              = true,
        linear_inequality_constraints = false,
        linear_system             = false,
        lqg                       = false,
        acceptable_obj_change_tol = 1e20,
        compl_inf_tol             = 0.0001,
        mu_init                   = 0.1,
        mu_strategy               = "monotone",
        warm_start_init_point     = "yes", # This benchmarked 25% faster
    )
    mu_strategy ∈ ("monotone", "adaptive") || throw(ArgumentError("""mu_strategy must be one of \"monotone\" or \"adaptive\""""))
    solver = Ipopt.Optimizer()

    MOI.set(solver, MOI.RawOptimizerAttribute("print_level"), verbose isa Bool ? verbose*5 : verbose) # Default 5, goes to 12
    MOI.set(solver, MOI.RawOptimizerAttribute("print_frequency_iter"), printerval)
    MOI.set(solver, MOI.RawOptimizerAttribute("tol"), tol)
    MOI.set(solver, MOI.RawOptimizerAttribute("acceptable_tol"), acceptable_tol)
    MOI.set(solver, MOI.RawOptimizerAttribute("max_iter"), max_iter)
    MOI.set(solver, MOI.RawOptimizerAttribute("max_cpu_time"), max_cpu_time) # in seconds
    MOI.set(solver, MOI.RawOptimizerAttribute("max_wall_time"), max_wall_time) # in seconds
    MOI.set(solver, MOI.RawOptimizerAttribute("constr_viol_tol"), constr_viol_tol)
    MOI.set(solver, MOI.RawOptimizerAttribute("acceptable_constr_viol_tol"), acceptable_constr_viol_tol)
    MOI.set(solver, MOI.RawOptimizerAttribute("acceptable_iter"), acceptable_iter)
    MOI.set(solver, MOI.RawOptimizerAttribute("compl_inf_tol"), compl_inf_tol)
    MOI.set(solver, MOI.RawOptimizerAttribute("mu_init"), mu_init)
    MOI.set(solver, MOI.RawOptimizerAttribute("mu_strategy"), mu_strategy)
    MOI.set(solver, MOI.RawOptimizerAttribute("acceptable_obj_change_tol"), float(acceptable_obj_change_tol))
    MOI.set(solver, MOI.RawOptimizerAttribute("warm_start_init_point"), warm_start_init_point) # This benchmarked 25% faster
    MOI.set(solver, MOI.RawOptimizerAttribute("warm_start_same_structure"), "no") # This benchmarked much slower 
    # MOI.set(solver, MOI.RawOptimizerAttribute("skip_finalize_solution_call"), "yes") # no noticeable difference
    # MOI.set(solver, MOI.RawOptimizerAttribute("alpha_for_y"), "min")

    
    if !exact_hessian
        MOI.set(solver, MOI.RawOptimizerAttribute("hessian_approximation"), "limited-memory")
    end
    linear_inequality_constraints && MOI.set(solver, MOI.RawOptimizerAttribute("jac_d_constant"), "yes")
    if lqg # Set this if solving robust MPC problems for linear systems with linear constraints
        linear_system = true
        MOI.set(solver, MOI.RawOptimizerAttribute("hessian_constant"), "yes")
    end
    linear_system && MOI.set(solver, MOI.RawOptimizerAttribute("jac_c_constant"), "yes")
    # NOTE: Ipopt has hessian_constant option to assume QP-style problem, requires linear constraints
    solver
end

function _check_arguments(solver, observer, objective_input, N)
    solver isa Union{MOI.AbstractOptimizer, MOI.OptimizerWithAttributes} ||
        throw(ArgumentError("We currently only support MathOptInterface solvers"))
    # We have overloaded lag_h to fit Ipopt, handle that generically before supporting other optimizers
    LowLevelParticleFilters.dynamics(observer) isa FunctionSystem{Continuous} && throw(ArgumentError("The observer is built using continuous-time dynamics, discretize the dynamics before constructing the observer. See https://help.juliahub.com/DyadControlSystemss/dev/mpc/#Discretization for more info."))
    size(objective_input.x, 2) == N+1 || error("Invalid size of x in objective_input, expected $(N+1), got $(size(objective_input.x, 2))")
    size(objective_input.u, 2) == N   || error("Invalid size of u in objective_input, expected $(N), got $(size(objective_input.u, 2))")
end

function setup_objective_input(disc::MultipleShooting, dynamics::FunctionSystem{TE}, objective_input) where TE
    TE <: Discrete || throw(ArgumentError("When using multiple shooting, the dynamics must be discrete time"))
    objective_input = remake(objective_input,discretization=disc)
end

function setup_objective_input(disc::CollocationFinE, dynamics::FunctionSystem{TE}, objective_input) where TE
    TE <: Continuous || throw(ArgumentError("When using collocation, the dynamics must be continuous time"))
    x = objective_input.x
    nx = size(x, 1)
    xn = [reshape(repeat(x[:, 1:end-1], disc.n_colloc, 1), nx,:) x[:, end]]
    objective_input = remake(objective_input,x=xn,discretization=disc)
end

function setup_objective_input(disc::Trapezoidal, dynamics::FunctionSystem{TE}, objective_input) where TE
    TE <: Continuous || throw(ArgumentError("When using Trapezoidal integration, the dynamics must be continuous time"))
    objective_input = remake(objective_input,discretization=disc)
end

function _setup_constraints(constraints, dynamics, objective_input, N, Ts, scale_x, scale_u, threads, disc, verbose)
    initial_state_constraint = InitialStateConstraint(copy(objective_input.x0), scale_x, dynamics)
    ii = dynamics.input_integrators
    i = findfirst(c->c isa BoundsConstraint, constraints)
    # translate between bounds constraints and stage constraints as needed depending on which inputs are associated with input integrators
    if i !== nothing
        bc = constraints[i]
        ii_with_ucon = filter(i->isfinite(bc.umin[i]) || isfinite(bc.umax[i]), ii)
        # We need to create StageConstraints for all bounds constraints on us that have input integrators
        if !isempty(ii_with_ucon)
            verbose && @info "Inputs $ii have input integrators and input bounds, the input bounds are being translated to StageConstraint."
            ucon = let ii = SVector(ii_with_ucon...)
                StageConstraint(bc.umin[ii], bc.umax[ii], N) do si, p, t
                    si.u[ii]
                end
            end
            umin, umax, dumin, dumax = copy(bc.umin), copy(bc.umax), copy(bc.dumin), copy(bc.dumax)
            umin[ii_with_ucon] .= bc.dumin[ii_with_ucon] # Move input-change bounds to input bounds for the inputs being integrated
            umax[ii_with_ucon] .= bc.dumax[ii_with_ucon]
            dumin[ii_with_ucon] .= -Inf
            dumax[ii_with_ucon] .= Inf
            constraints[i] = remake(bc; umin, umax, dumin, dumax)
            constraints = [constraints; ucon] # To handle eltype being too specific
        end
        not_ii = setdiff(1:dynamics.nu, ii)
        not_ii_with_ducon = filter(i->isfinite(bc.dumin[i]) || isfinite(bc.dumax[i]), not_ii)
        ii_with_ducon = filter(i->isfinite(bc.dumin[i]) || isfinite(bc.dumax[i]), ii)
        if !isempty(not_ii_with_ducon)
            verbose && @info "Inputs $not_ii_with_ducon have input-difference bounds, DifferenceConstraints are being added."
            # We need to create difference constraints for these variables
            ducon = let nii = SVector(not_ii_with_ducon...)
                getter = (si, p, t)->si.u[nii]
                DifferenceConstraint(getter, bc.dumin[nii], bc.dumax[nii], N) do du, p, t
                    du
                end
            end
            constraints = [constraints; ducon] # To handle eltype being too specific
        end
        if !isempty(ii_with_ducon)
            verbose && @info "Inputs $ii_with_ducon have input integrators and input-difference bounds, the input-difference bounds are converted to input bounds."
            bc = constraints[i] # This is the new bc
            umin, umax = copy(bc.umin), copy(bc.umax)
            umin[ii_with_ducon] .= bc.dumin[ii_with_ducon] # Move input-change bounds to input bounds for the inputs being integrated
            umax[ii_with_ducon] .= bc.dumax[ii_with_ducon]
            constraints[i] = remake(bc; umin, umax)
        end
    end
    constraints = CompositeMPCConstraint(N, Ts, scale_x, scale_u, threads, initial_state_constraint, disc, constraints...)
end

function safe_similar(x, ::Type{T}=Float64) where T
    y = similar(x, T)
    y.nzval .= 1
    y
end

_setup_constraints(constraints::CompositeMPCConstraint, dynamics, objective_input, N, Ts, scale_x, scale_u, threads, disc, verbose) = constraints

include("symbolic_ad.jl")

function _test_and_precompile_functions(optfun, vars, nc, p, verbose)
    var0 = copy(vars.vars)
    nvars = length(var0)
    nc = nc*vars.n_robust + vars.nu*optfun.cons.f.cons.robust_horizon*(vars.n_robust-1)
    verbose && @info "Testing loss function"
    optfun.f(var0, p)
    verbose && @info "Testing gradient function"
    optfun.grad(zeros(nvars), var0)
    verbose && @info "Testing hessian function"
    optfun.hess_prototype !== nothing && optfun.hess(safe_similar(optfun.hess_prototype), var0, p)
    verbose && @info "Testing constraint function"
    optfun.cons(zeros(nc), var0)
    verbose && @info "Testing constraint jacobian function"
    optfun.cons_jac_prototype !== nothing && optfun.cons_j(safe_similar(optfun.cons_jac_prototype), var0)
    # verbose && @info "Testing lagrangian hessian function"
    # optfun.lag_h(similar(optfun.lag_hess_prototype.nzval), var0, 1.0, zeros(nc), p)
    verbose && @info "Done testing"
end

function _build_OptimizationFunction(objective, constraints, objective_input,
    vars, p, scale_x, scale_u, robust_horizon, autodiff, symbolic_lag_h, verbose,
)

    loss = LossFunction(objective, objective_input, vars, p, 0.0, scale_x, scale_u)
    t = 0.0 # TODO
    cons = ConstraintEvaluator(constraints, deepcopy(objective_input), vars, p, t, scale_x, scale_u, robust_horizon)
    if symbolic_lag_h
        verbose && @info "Building symbolic lagraingian hessian"
        lag_h, hp = build_symbolic_lag_hessian(cons, loss, vars, robust_horizon, p; verbose=false)
        verbose && @info "Done building symbolic lagraingian hessian"
        OptimizationFunction{true}(loss, autodiff; cons, cons_vjp = false, lag_h, lag_hess_prototype=hp);
    else
        OptimizationFunction{true}(loss, autodiff; cons, cons_vjp = false);
    end

end

function _build_OptimizationProblem(optfun, constraints, objective_input, p, scale_x, scale_u, int_x, int_u, Nint, vars)
    var0 = copy(vars.vars)
    lcons, ucons = get_bounds(constraints)
    lcons = repeat(lcons, vars.n_robust)
    ucons = repeat(ucons, vars.n_robust)

    @unpack N, nx, nu, n_robust = vars
    robust_horizon = optfun.cons.robust_horizon
    lcons = [lcons; zeros(nu*robust_horizon*(n_robust-1))] # add coupling between initial contorl signals due to robust MPC
    ucons = [ucons; zeros(nu*robust_horizon*(n_robust-1))]

    lvar, uvar = get_variable_bounds(constraints, objective_input, scale_x, scale_u, vars.n_robust)

    disc = objective_input.discretization
    xinds = all_x_indices(disc, N, nx, nu, n_robust, 1)
    uinds = all_u_indices(disc, N, nx, nu, n_robust, 1)
    Nx = length(xinds) ÷ nx
    Nu = length(uinds) ÷ nu
    if any(int_x) || any(int_u)
        int = [
            repeat(repeat(int_x, Nx), n_robust)
            repeat([
                repeat(int_u, Nint*nu)
                falses(Nu-Nint*nu)
            ], n_robust)
        ]
    else
        int = nothing
    end
    OptimizationProblem(optfun, var0, p; lcons, ucons, lb=lvar, ub=uvar, int);
end

"""
    GenericMPCProblem(dynamics; N, observer, objective, constraints::Union{AbstractVector{<:AbstractGenericMPCConstraint}, CompositeMPCConstraint}, solver = IpoptSolver(), p = DiffEqBase.NullParameters(), objective_input, xr, presolve = true, threads = false)

Defines a generic, nonlinear MPC problem. 

# Arguments:
- `dynamics`: An instance of [`FunctionSystem`](@ref)
- `N`: Prediction horizon in the MPC problem (number of samples, equal to the control horizon)
- `Ts`: Sample time for the system, the control signal is allowed to change every `Ts` seconds.
- `observer`: An instance of `AbstractObserver`, defaults to `StateFeedback(dynamics, zeros(nx))`.
- `objective`: An instance of [`Objective`](@ref).
- `constraints`: A vector of [`AbstractGenericMPCConstraint`](@ref).
- `solver`: An instance of an optimizer recognized by Otpimazation.jl. The recommended default choice is the Ipopt.Optimizer, which can be created using [`IpoptSolver`](@ref).
- `p`: Parameters that will be passed to the dynamics, cost and constraint functions. It is possible to provide a different set of parameters for the cost and constraint functions by passing in an instance of [`MPCParameters`](@ref). 
- `objective_input`: An instance of [`ObjectiveInput`](@ref) that contains initial guesses of states and control trajectories.
- `xr`: Reference trajectory
- `presolve`: Solve the initial optimal-control problem already in the constructor. This may provide a better initial guess for the first online solve of the MPC controller than the one provided by the user.
- `threads`: Use threads to evaluate the dynamics constraints. This is beneficial for large systems.
- `scale_x = ones(dynamics.nx)`: Scaling factors for the state variables. These can be set to the "typical magnitude" of each state to improve numerical performance in the solver.
- `scale_u = ones(dynamics.nu)`: Scaling factors for the input variables.
- `disc = MultipleShooting(dynamics)`: Discretization method. Defaults to [`MultipleShooting`](@ref). See [`Discretization`](@ref) for options.
- `robust_horizon`: Set to a positive integer to enable robust MPC. The robust horizon is the number of initial control inputs that are constrained to be equal for robust optimization problems. Defaults to `0` (no robust MPC). When robust MPC is used, the parameter argument `p` is expected to be a vector of parameter structures, e.g., `Vector{<:Vector}` or `Vector{Dict}` etc. See the tutorial [Robust MPC with uncertain parameters](@ref) for additional information. For robust MPC, `robust_horizon = 1` is the most common choice, for trajectory optimization in open loop, `robust_horizon = N` is recommended. For hierarchical controllers where several (`Nh`) time steps of the optimized MPC trajectory are exectured before re-optimizing, set the robust horizon to `Nh`. In situations where several consecutive steps of the optimized trajectory are executed before re-optimizing, set `robust_horizon` to the maximum number of steps before re-optimization.
- `symbolic_lag_h`: Set to `true` to use symbolic differentiation for the Lagrangian Hessian. Defaults to `true` for collocation methods and `Trapezoidal`, and `false` for multiple shooting. 
- `int_x::Vector{Bool}`: Vector of booleans indicating which state variables should be treated as integer variables. Defaults to `falses(dynamics.nx)`. See https://docs.sciml.ai/Optimization/dev/optimization_packages/mathoptinterface/#Using-Integer-Constraints for more details.
- `int_u::Vector{Bool}`: Vector of booleans indicating which input variables should be treated as integer variables. Defaults to `falses(dynamics.nu)`.
- `Nint = N`: Number of control inputs that should be treated as integer variables. Defaults to all control inputs. The computational effort required to solve control problems can sometimes be reduced significantly by relaxing the problem to only require the first ``N_{int} < N`` control inputs to be integer.
"""
function GenericMPCProblem(
    dynamics::FunctionSystem{TE};
    N,
    Ts = hasproperty(dynamics, :Ts) ? dynamics.Ts : error("When continuous-time dynamics is used, the sample time Ts must be provided to GenericMPCProblem"),
    observer,
    objective,
    constraints::Union{AbstractVector{<:AbstractGenericMPCConstraint}, CompositeMPCConstraint} = AbstractGenericMPCConstraint[],
    solver = IpoptSolver(),
    p = DiffEqBase.NullParameters(),
    xr,
    presolve = true,
    threads = false,
    verbose = false,
    scale_x = ones(dynamics.nx),
    scale_u = ones(dynamics.nu),
    disc = MultipleShooting(dynamics, N, scale_x, threads),
    autodiff = default_autodiff(disc),
    symbolic_lag_h = default_sym_lag_h(disc),
    robust_horizon = 0,
    int_x::AbstractVector{Bool} = falses(dynamics.nx),
    int_u::AbstractVector{Bool} = falses(dynamics.nu),
    Nint::Int = N,
    objective_input::ObjectiveInput = default_objective_input(dynamics, get_system_parameters(get_parameter_index(p, 1)), state(observer), xr, N, Ts),
    build_optprob = true,
) where TE

    if robust_horizon > 0
        p isa Vector || error("When using robust MPC (`robust_horizon` > 0), the parameter p must be a vector of parameters.")
        n_robust = length(p)
    else 
        robust_horizon, n_robust = 0, 1
    end
    _check_arguments(solver, observer, objective_input, N)
    length(int_x) == dynamics.nx || error("The length of int_x must be equal to the number of states.")
    length(int_u) == dynamics.nu || error("The length of int_u must be equal to the number of inputs.")
    objective_input = setup_objective_input(disc, dynamics, deepcopy(objective_input))
    
    x = objective_input.x
    u = objective_input.u
    vars = Variables(copy(x), copy(u), disc, n_robust)
    if build_optprob
        constraints = _setup_constraints(constraints, dynamics, objective_input, N, Ts, scale_x, scale_u, threads, disc, verbose)

        optfun = _build_OptimizationFunction(objective, constraints, objective_input, vars, p, scale_x, scale_u, robust_horizon, autodiff, symbolic_lag_h, verbose)

        optprob0 = _build_OptimizationProblem(optfun, constraints, objective_input, p, scale_x, scale_u, int_x, int_u, Nint, vars)
        optprob = init(optprob0, solver)
        _test_and_precompile_functions(optprob.f, vars, length(constraints), p, verbose)
    else
        optprob = optprob0 = (; objective, constraints)
        presolve = false
    end

    # scale!(vars, scale_x, scale_u)
    # update_initial_guess!(optprob, vars.vars)
    # if presolve
    #     verbose && @info "Presolving for initial trajectory"
    #     MOI.empty!(solver)
    #     sol = solve(optprob, solver);
    #     vars = remake(vars; vars=sol.u)
    # end
    # unscale!(vars, scale_x, scale_u)

    prob = GenericMPCProblem(
        dynamics,
        observer,
        constraints,
        solver,
        N,
        Float64(Ts),
        p,
        objective_input,
        copy(xr),
        optprob,
        optprob0,
        vars,
        scale_x,
        scale_u,
    )
    if presolve
        verbose && @info "Presolving for initial trajectory"
        controllerinput = ControllerInput(MPC.state(prob.observer), prob.xr, [], u[:, 1])
        MPC.optimize!(prob, controllerinput, p, 0.0)
    end
    prob
end

empty_solver!(solver::MOI.AbstractOptimizer) = MOI.empty!(solver)
empty_solver!(solver::Any) = nothing

get_xu(prob::GenericMPCProblem, ri::Int=1) = get_xu(prob.vars, prob.objective_input.u0, ri)

update_initial_guess!(optprob, vars::AbstractVector) = optprob.u0 .= vars

function update_t!(prob, t)
    # Update the time variable in both constraint evaluator and loss function
    prob.optprob.f.cons.f.cons.t = t
    prob.optprob.f.f.t = t
end

function update_u0!(prob, u0)
    # Update the time variable in both constraint evaluator and loss function
    # Main.a = prob.optprob.f.cons.f.cons

    prob.optprob.f.cons.f.cons.objective_input.u0 .= u0
    prob.optprob.f.f.objective_input.u0 .= u0
end

function maybe_update!(dest, src)
    if dest !== src
        size(dest) == size(src) || throw(DimensionMismatch("Cannot update ObjectiveInput array `r`, got sizes $(size(dest)) and $(size(src))"))
        Base.ismutable(dest) || throw(ArgumentError("Cannot update ObjectiveInput array `r`, it is not mutable ($(typeof(dest))"))
        dest .= src
    end
end


"""
    controlleroutput = MPC.optimize!(prob::GenericMPCProblem, controllerinput::ControllerInput, p, t)

Perform optimization for the MPC problem `prob` at time `t` with parameters `p` and initial state `controllerinput`.
"""
function MPC.optimize!(prob::GenericMPCProblem, controllerinput::ControllerInput, p, t; verbose = true)
    (; optprob, solver, vars, objective_input, constraints) = prob
    # (; scale_x, scale_u) = optprob.f.f
    (; scale_x, scale_u) = prob

    # Variables is the canonical source of truth, only this structure is allowed to be mutated
    # This happens inside of advance!
    t > 0 && advance!(prob, controllerinput) # This updates vars
    update_t!(prob, t)
    update_u0!(prob, controllerinput.u0)
    for ri = 1:vars.n_robust
        x,u = get_xu(vars, controllerinput.u0, ri)
        @assert !(controllerinput.u0 isa SubArray || controllerinput.x isa SubArray) # If it had been, it might have been advanced incorrectly.
        
        # The remake does not change the objective input in the constraint evaluator, the CE needs updated reference trajectory for the constraints that depend on the reference
        # objective_input = remake(objective_input; x, u, r=prob.xr, u0=controllerinput.u0, x0 = controllerinput.x)
        objective_input.x .= x
        objective_input.u .= u
        objective_input.u0 .= controllerinput.u0
        objective_input.x0 .= controllerinput.x
        maybe_update!(objective_input.r, prob.xr)
        maybe_update!(optprob.f.cons.f.cons.objective_input.r, objective_input.r)

        # Update optprob bounds and initial guess
        (; lcons, ucons) = optprob
        nc = length(constraints)
        @views update_bounds!(lcons[(1:nc) .+ (ri-1)*nc], ucons[(1:nc) .+ (ri-1)*nc], constraints, objective_input)
    end
    
    # The scaling is handled like this
    # We scale just before solving the problem, and unscale immediately afterwards. Inside the loss function and constraint evaluator, the variables are unscaled to handle nonlinearities. The dynamics constraints and initial-value constraint performs scaling again. 
    scale!(vars, scale_x, scale_u)
    update_initial_guess!(optprob, vars.vars)
    empty_solver!(solver)
    if optprob.p !== p
        optprob = init(remake(prob.optprob0; p), solver) # This cannot be done anymore since optprob has been `init!`ed
        optprob.f.cons.f.cons.p = p
        optprob.f.f.p = p
    end
    sol = OptimizationMOI.solve!(optprob); 
    vars.vars .= sol.u
    unscale!(vars, scale_x, scale_u)
    x,u = get_xu(vars, objective_input.u0) # These are now views, so we may not advance the problem until we are done using the views
    ControllerOutput(x, u, sol)
end


 
"""
    MPC.step!(prob::GenericMPCProblem, observerinput, p, t; kwargs)

# Arguments:
- `observerinput`: An instance of type [`ObserverInput`](@ref).
"""
@views function MPC.step!(prob::GenericMPCProblem, observerinput, p, t; p_actual = p, kwargs...)
    (; u,y,r,w) = observerinput
    # n_robust = prob.vars.n_robust
    p_observer = get_parameter_index(p_actual, 1) |> get_system_parameters
    MPC.mpc_observer_correct!(prob.observer, u, y, r, p_observer, t)
    controllerinput = ControllerInput(MPC.state(prob.observer), prob.xr, w, u)
    controlleroutput = optimize!(prob, controllerinput, p, t; kwargs...)
    u0 = get_first(controlleroutput.u)
    MPC.mpc_observer_predict!(prob.observer, u0, get_first(r), p_observer, t)
    controlleroutput
end

"""
    advance_xr!(prob::GenericMPCProblem, controllerinput = nothing)

Advance the reference trajectory stored in the problem by one time step (shifting it to the left). If a `controllerinput::ControllerInput` containing a new reference is provided, this reference
- Replaces the old reference trajectory if `controllerinput.r` is a matrix of the same size as the reference trajectory stored in the problem.
- Replaces the _last_ element of the reference trajectory if `controllerinput.r` is a vector of the same size as the last element of the reference trajectory stored in the problem. If the reference opint stored in the problem is a static array, this update is not supported.
"""
function MPC.advance_xr!(prob::GenericMPCProblem, ci::Union{ControllerInput, Nothing} = nothing)
    @unpack nx, nu, N, xr, ur = prob
    # u0 = prob.u[:, 1]
    copyto!(xr, 1, xr, nx+1, length(xr)-nx) # advance xr
    copyto!(ur, 1, ur, nu+1, length(ur)-nu) # advance ur
    if ci !== nothing
        if ci.r isa AbstractVector
            if ci.r isa SVector
                isequal(prob.xr, ci.r) || throw(ArgumentError("The reference stored in the GenericMPCProblem is a static array, it cannot be updated with a new reference point."))
                return
            end
            length(ci.r) == size(xr, 1) || throw(DimensionMismatch("The length of the updated reference vector `r` must match the size of the reference trajectory stored in the problem."))
            xr[:, end] .= ci.r
        elseif ci.r isa AbstractMatrix
            size(ci.r) == size(xr) || throw(DimensionMismatch("The size of the updated reference trajectory matrix `r` must match the size of the reference trajectory stored in the problem."))
            xr .= ci.r
        end
    end
    nothing
end

function MPC.advance!(prob::GenericMPCProblem, ci::Union{ControllerInput, Nothing} = nothing)
    @unpack nx, nu, N = prob
    MPC.advance_xr!(prob, ci) # This takes u0 from prob.u and must be done before advancing u
    x, u = get_xu(prob.vars)
    if prob.vars.n_robust == 1 # TODO: advance robust problems
        if prob.vars isa Variables{<:Union{<:MultipleShooting, <:Trapezoidal}}
            copyto!(x, 1, x, nx+1, N*nx) # advance the state trajectory
        else
            nc = prob.vars.disc.n_colloc
            copyto!(x, 1, x, nc*nx+1, (N-1)*nx*nc) # advance the state trajectory
        end
        copyto!(u, 1, u, nu+1, (N-1)*nu) # advance the control trajectory
    end
    nothing
end

"""
    MPC.solve(prob::GenericMPCProblem, alg = nothing; x0,
        T,
        verbose         = false,
        p               = MPC.parameters(prob),
        callback        = (actual_x, u, x, X, U)->nothing,
        sqp_callback,
        noise           = 0,
        reset_observer  = true,
        dyn_actual      = deepcopy(prob.dynamics),
        x0_actual       = copy(x0),
        p_actual        = p,
        disturbance     = (u,t)->0,
    )

Simulate an MPC controller in feedback loop with a plant for `T` steps. To step an MPC controller forward one step, see [`step!`](@ref).

# Arguments:
- `x0`: Initial state
- `T`: Number of time steps to perform the simulation.
- `verbose`: Print stuff
- `p`: Parameters
- `callback`: Called after each iteration
- `noise`: Add measurement noise to the simulation, the noise is Gaussian with `σ = noise` (can be a vector or a scalar).
- `reset_observer`: Set the intitial state of the observer to `x0` before each simulation.
- `dyn_actual`: Actual dynamics. This defaults to `prob.dynamics`, but can be set to any other dynamics in order to simulate model errors etc. If this is set, set also `x0_actual` and `p_actual`.
- `x0_actual`: Initial state for the actual dynamics.
- `p_actual`: Parameters for the actual dynamics.
- `disturbance`: A function (u_out,t)->0 that takes the control input `u` and modifies this in place to compute the disturbance.
"""
function MPC.solve(prob::GenericMPCProblem, alg = nothing; x0, T, verbose = false, p = MPC.parameters(prob), callback=(args...)->(), noise=0, reset_observer=true, dyn_actual = deepcopy(prob.dynamics), x0_actual = copy(x0), p_actual=p, disturbance = (u,t)->0)

    observer = prob.observer
    if reset_observer # TODO: reconsider this strategy
        MPC.reset_observer!(observer, x0)
    end
    update_u0!(prob, prob.objective_input.u0) # This updates the u0 inside loss and constraint evaluator
    hist = MPC.NonlinearMPCHistory(prob)

    actual_xr = Vector(get_first(prob.xr))
    actual_ur = Vector(get_first(prob.ur))
    actual_x = copy(x0_actual)
    u = copy(prob.objective_input.u0) # This will be updated in-place so we make a copy
    w = [] # TODO: implement support for this

    for t = 1:T
        verbose && println("t = $t")
        if dyn_actual isa AbstractStateSpace
            msys = dyn_actual
            y = msys.C*actual_x + msys.D*u
        else
            dyn_actual isa FunctionSystem{Continuous} && error("Cannot simulate a continuous-time system, provide a discrete-time dyn_actual to MPC.solve.")
            g = dyn_actual.measurement
            y = g(actual_x, u, p_actual, t)
        end
        if noise != false
            y = y .+ noise .* randn.() # not in-place if y is static
        end
        
        r = get_first(prob.xr)
        observerinput = ObserverInput(u,y,r,w)
        treal = (t-1)*prob.Ts
        co = MPC.step!(prob, observerinput, p, treal; p_actual = observer isa StateFeedback ? p_actual : p, verbose)
        push!(hist, actual_x, co.u, copy(actual_xr), copy(actual_ur), y)
        callback(actual_x, co.u, co.x, hist.X, hist.U)

        d_u = disturbance(get_first(co.u), treal) # Deliberate copy
        sim_u = co.u[:,1] .+ d_u
        actual_x = dyn_actual(actual_x, sim_u, p_actual, treal) # Advance actual state

        @views actual_xr .= get_first(prob.xr) # prob.xr has already been advanced in step!
        @views actual_ur .= get_first(prob.ur) # prob.ur has already been advanced in step!
        # advance!(hist) # This changes u
        @views u .= co.u[:, 1] 
    end
    push!(hist.X, actual_x) # add one last state to make the state array one longer than control

    hist
end

get_first(a::AbstractMatrix) = a[:,1]
get_first(a::AbstractVector) = a

import RobustAndOptimalControl.MonteCarloMeasurements as MCM
VecOrScalar = Union{AbstractVector, Number}

function expand_uncertain_operating_points(op::MPCParameters{<:VecOrScalar, <:VecOrScalar})
    # check the validity of uncertain parameters
    paths = MCM.particle_paths(op)
    found_particle_numbers = Set{Int}(getindex.(paths, 3))
    if length(found_particle_numbers) > 1
        error("The number of samples (particles) for all uncertain parameters must be the same, but I found $(found_particle_numbers)")
    elseif isempty(found_particle_numbers)
        error("No uncertain parameters found")
    end
    N = only(found_particle_numbers)
    N > 100 && @warn "You have a lot of particles ($N), this may result in a computationally heavy MPC problem." maxlog=1
    p = map(1:N) do i
        MCM.vecindex(op.p, i)
    end
    p_mpc = map(1:N) do i
        MCM.vecindex(op.p_mpc, i)
    end
    MPCParameters.(p, p_mpc)
end