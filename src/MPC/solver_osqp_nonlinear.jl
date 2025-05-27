#=
OSQP MPC solver implementation
While the interface to the solver is supposed to be generic and additional solvers could be provided, OSQP was chosen because
- It's developed by a control group => custom-made and battle tested for MPC
- Supports warm starting
- Supports sparse matrices
- Supports C-code generation
- Division free algorithm (important for embedded devices where / is expensive

While tools like JuMP or Convex could in theory be used to interface the solver,
they both break down for large problem sizes, JuMP in particular only handles toy problems due to their lack of array variables. MPC also requires low-level control over the solver which is awkward at best with these tools. 
=#

"""
    OSQPSolver <: AbstractMPCSolver

A solver which uses sequential quadratic programming on a system linearized around a trajectory computed using discretized dynamics.

# Arguments:
- `verbose::Bool = false`
- `eps_rel::Float64 = 1e-5`
- `eps_abs::Float64 = 1e-5`
- `max_iter::Int = 5000`
- `check_termination::Int = 15`: The interval at which termination criteria is checked
- `sqp_iters::Int = 2`: Number of sequential QP iterations to run. Oftentimes 1-5 is enough, but hard non-linear problems can be solved more robustly with more iterations. This option has no effect for linear MPC.
- `dynamics_interval::Int = 1`: How often to update the linearization. This is an expensive step, and the (average) solution time can be sped up by not updating every step. This must be set to 1 if `sqp_iters > 1`. This option has no effect for linear MPC.
- `polish::Bool = true`: Try to obtain a high-accuracy solution, increases computational time.
"""
Base.@kwdef struct OSQPSolver <: AbstractMPCSolver
    verbose           :: Bool    = false
    eps_rel           :: Float64 = 1e-5
    eps_abs           :: Float64 = 1e-5
    max_iter          :: Int     = 5000
    check_termination :: Int     = 15
    sqp_iters         :: Int     = 2
    dynamics_interval :: Int     = 1
    polish            :: Bool    = true
    time_limit        :: Float64 = 1e10
end
mutable struct OSQPSolverWorkspace
    model::OSQP.Model
    solver::OSQPSolver
end

function init_solver_workspace(solver::OSQPSolver; P, q, A, lb, ub)
    # Create an OSQP model
    model = OSQP.Model()
    # Setup workspace
    OSQP.setup!(
        model;
        P,
        q,
        A,
        l=lb,
        u=ub,
        solver.verbose,
        solver.eps_rel,
        solver.max_iter,
        solver.check_termination,
        solver.eps_abs,
        solver.polish,
        solver.time_limit,
    )
    OSQPSolverWorkspace(model, solver)
end

function update_hessian!(solver::OSQPSolverWorkspace, P)
    OSQP.update!(solver.model; Px=triu(P).nzval)
end

function update_constraints!(solver::OSQPSolverWorkspace, lb, ub)
    OSQP.update!(solver.model; l=lb, u=ub)
end

function update_constraints!(solver::OSQPSolverWorkspace, A)
    OSQP.update!(solver.model; Ax = A.nzval)
end

function update_q!(solver::OSQPSolverWorkspace, q)
    OSQP.update!(solver.model; q = q)
end

is_sqp_problem(prob) = prob.solver.solver.sqp_iters > 1

function optimize!(solver::OSQPSolverWorkspace, prob::QMPCProblem, p, t; verbose = true, callback=nothing)
    @unpack x, u, s, xr, ur, N, nx, nu, ns, dynamics = prob
    n_tot = nx+nu+2ns # Total number of variables for each time step
    update_dynamics = true
    sqp_iters = solver.solver.sqp_iters
    dynamics_interval = solver.solver.dynamics_interval

    if dynamics_interval > 1
        sqp_iters <= 1 || throw(ArgumentError("It does not make sense to have sqp_iters > 1 if dynamics is not updated every interval."))
        update_dynamics = mod1(t, dynamics_interval) == 1
    end

    # rollout!(prob)
    u0 = u[:,1]
    x0 = x[:,1]


    local res
    xlin = copy(x)
    ulin = copy(u)
    # update_xr!(prob, xr, force=true)
    rollout!(dynamics, x, u, p, t) # this one can be moved outside of SQP iterations since there is an additional one below
    for sqp_iter = 1:sqp_iters
        # prob.xri .= prob.xr
        if is_sqp_problem(prob)
            xlin .= x
            ulin .= u
            update_constraints_sqp!(prob, x, u, x0, s; update_dynamics)
            update_xr_sqp!(prob, xr, ur, xlin, ulin, p)
        else
            update_constraints!(prob, x, u, x0; update_dynamics)
        end
        # x[:, 1] .= x0
        res = OSQP.solve!(solver.model)
        c = res.info.obj_val

        @views XUS = reshape(res.x[1:N*n_tot], n_tot, :)


        xopt = copy_x!(copy(x), res.x, N, nx, nu, ns) # deliberate copy
        uopt = copy_u!(copy(u), res.x, N, nx, nu, ns) # deliberate copy
        
        if is_sqp_problem(prob)
            copy_s!(s, res.x, N, nx, nu, ns) # deliberate copy
            # form x = xopt + xr, u = uopt + ur
            x += xopt*(1/sqrt(sqp_iter)) # Note, do not overwrite x
            x[:, 1] .= x0
            # @show norm(xopt), norm(uopt)
            u += uopt*(1/sqrt(sqp_iter)) # NOTE: do not overwrite in-place at the moment
        else
            # x += xopt # Note, do not overwrite x
            u = uopt # NOTE: do not overwrite in-place at the moment
            rollout!(dynamics, x, u, p, t) # update_constraints calls the dynamics
        end


        if verbose
            if res.info.status != :Solved
                @error("OSQP did not solve the problem!", res.info.status)
            end
            @info "Objective: $c"
            if ns > 0
                sinds = [repeat([falses(nx+nu); trues(2ns)], N); falses(nx)]
                slacknorm = norm(res.x[sinds])
                slackextrema = extrema(res.x[sinds])
                printstyled("Slack norm: ", round(slacknorm, digits=6), " extreme slacks: ", slackextrema, '\n', color = slackextrema[2] > 1e-3 ? :red : :normal)
            end
            
            # xc = copy(x)
            # rollout!(dynamics, xc, u) # do an additional rollout for the modelfit diagnostic
            if is_sqp_problem(prob)
                residual = mean(abs2, xopt ./ x)
                if sqp_iter == sqp_iters && residual > 1e-3
                    @warn "The linearized trajectory followed by the optimizer was far from the trajectory of the nonlinear integrator (residual = $residual). The problem might be highly nonlinear. Try\n1) increasing the number of SQP iterations (keyword arg. sqp_iters to QMPCProblem) \n2) lower the discretization time and or the optimization horizon."
                end
            else
                fit = modelfit(x, xopt)
                if sqp_iter == sqp_iters && mean(fit) < 75
                    @warn "The linearized trajectory followed by the optimizer was far from the trajectory of the nonlinear integrator (fit = $fit)%. The problem might be highly nonlinear. Try\n1) increasing the number of SQP iterations (keyword arg. sqp_iters to QMPCProblem) \n2) lower the discretization time and or the optimization horizon."
                end
            end
        end
        all(isfinite, res.x) || error("Solver produced non-finite values at time $t. The solver status was $(res.info.status). If the status was inaccurate, try increasing the maximum number of iterations (max_iter keyword arg to OSQPSolver).")
        if callback !== nothing
            # xc = copy(x)
            # rollout!(dynamics, x, u, p, t) # do an additional rollout for the callback
            callback(x, u, xopt, t, sqp_iter, prob)
        end
        if is_sqp_problem(prob) && maximum(abs, xopt) < solver.solver.eps_abs
            verbose && @info "Breaking at SQP-iteration $sqp_iter"
            break
        end
    end
    xopt = copy_x!(copy(x), res.x, N, nx, nu, ns) # deliberate copy
    if !is_sqp_problem(prob)
        xsol = res.x
        copyto!(xsol, 1, xsol, n_tot+1, (N-1)*n_tot) # forward opt variables to next iteration
        @timeit to "warm_start" OSQP.warm_start_x!(solver.model, xsol)
    end
    ControllerOutput(xopt, u, res)
end



function copy_x!(x, sol, N, nx, nu, ns)
    xinds = 1:nx
    n_tot = nx + nu + 2ns
    @views for i = 1:N+1
        x[:, i] .= sol[xinds]
        xinds = xinds .+ n_tot
    end
    x
end
function copy_u!(u, sol, N, nx, nu, ns)
    uinds = (1:nu) .+ nx
    n_tot = nx + nu + 2ns
    @views for i = 1:N
        u[:, i] .= sol[uinds]
        uinds = uinds .+ n_tot
    end
    u
end
function copy_s!(s, sol, N, nx, nu, ns)
    sinds = (1:2ns) .+ (nx + nu)
    n_tot = nx + nu + 2ns
    @views for i = 1:N
        s[:, i] .= sol[sinds]
        sinds = sinds .+ n_tot
    end
    s
end