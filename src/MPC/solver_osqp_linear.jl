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


function optimize!(solver::OSQPSolverWorkspace, prob::LQMPCProblem, p, t; verbose = true, callback=nothing)
    @unpack x, u, r, N, nx, nu, dynamics, qs, qs2 = prob
    ns = length(dynamics.soft_indices) * (qs==qs2==0 ? 0 : 1) # Number of soft constraints (num slack is 2ns)
    n_tot = nx+nu+2ns # Total number of variables for each time step
    update_dynamics = false

    # rollout!(prob)
    u0 = u[:,1]
    x0 = x[:,1]
    xinds = 1:nx
    uinds = (1:nu) .+ nx
    local res
    # update_xr!(prob, xr, force=true)
    rollout!(dynamics, x, u, p, t) # this one can be moved outside of SQP iterations since there is an additional one below
    
    update_constraints!(prob, x, u, x0; update_dynamics)

    # x[:, 1] .= x0
    res = OSQP.solve!(solver.model)
    c = res.info.obj_val

    xopt = copy_x!(copy(x), res.x, N, nx, nu, ns) # deliberate copy
    uopt = copy_u!(copy(u), res.x, N, nx, nu, ns) # deliberate copy

    # x += xopt # Note, do not overwrite x
    u = uopt # NOTE: do not overwrite in-place at the moment
    rollout!(dynamics, x, u, p, t) # update_constraints calls the dynamics

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
        
    end
    all(isfinite, res.x) || error("Solver produced non-finite values at time $t. The solver status was $(res.info.status). If the status was inaccurate, try increasing the maximum number of iterations (max_iter keyword arg to OSQPSolver).")
    if callback !== nothing
        # xc = copy(x)
        # rollout!(dynamics, x, u, p, t) # do an additional rollout for the callback
        callback(x, u, xopt, t, prob)
    end
    
    xopt = copy_x!(copy(x), res.x, N, nx, nu, ns) # deliberate copy

    xsol = res.x
    copyto!(xsol, 1, xsol, n_tot+1, (N-1)*n_tot) # forward opt variables to next iteration
    @timeit to "warm_start" OSQP.warm_start_x!(solver.model, xsol)

    ControllerOutput(xopt, u, res)
end