using Symbolics
using DyadControlSystems.MPC
using DyadControlSystems.MPC: AbstractMPCProblem
using UnPack
using OSQP


##


function quick_trim(dynamics, xr, ur)
    nx = length(xr)
    optres = Optim.optimize([xr;ur], BFGS(), Optim.Options(iterations=1000)) do xu
        x,u = xu[1:nx],xu[nx+1:end]
        @views sum(abs, dynamics(x,u,0,0)-x) + 0.0001sum(abs2, x-xr)
    end
	@info optres
	optres.minimizer[1:nx], optres.minimizer[nx+1:end]
end


function XUcost(hist)
    X,E,R,U,Y = reduce(hcat, hist)
    X, U, MPC.lqr_cost(hist)
end
function sqp_callback(x,u,xopt,t,sqp_iter,prob)
    nx, nu = size(x,1), size(u,1)
    c = MPC.lqr_cost(x,u,prob)
    plot(x[:, 1:end-1]', layout=2, sp=1, c=(1:nx)', label="x nonlinear", title="Time: $t SQP iteration: $sqp_iter")
    plot!(xopt[:, 1:end-1]', sp=1, c=(1:nx)', l=:dash, label="x opt")
    plot!(u', sp=2, c=(1:nu)', label = "u", title="Cost: $(round(c, digits=5))") |> display
    sleep(0.001)
end


function vvariable(name, length)
    # un = Symbol(name)
    # u = @variables $un[1:length]
    # collect(u[])
    map(1:length) do i
        un = Symbol(name*string(i))
        u = Symbolics.@variables $un
        u[]
    end
    # u[]
end
function loss(x, u, ulast, n, prob)
    c = 0.5*(dot(x, prob.Q1, x) + dot(u, prob.Q2, u))
    if prob.Q3 !== nothing && n > 1
        du = u - ulast
        c +=  0.5dot(du, prob.Q3, du)
    end
    c
end
function final_cost(x, prob)
    0.5x'prob.QN*x
end

struct SymbolicMPC{PT, QT, AT, GT, LBT, UBT}
    P::PT
    q::QT
    A::AT
    g::GT
    lb::LBT
    ub::UBT
end


function warmstart(model, res, prob)
    @unpack nx, nu, N = prob
    xsol = res.x
    copyto!(xsol, 1, xsol, nx+1, N*nx) # forward states
    copyto!(xsol, (N+1)*nx+1, xsol, (N+1)*nx+1+nu, (N-1)*nu) # forward control
    OSQP.warm_start_x!(model, xsol)
end

function build_symbolic_functions(prob::AbstractMPCProblem, constraints)
    X   = Num[] # variables
    U   = Num[] # variables
    lbX = Num[] # lower bound on w
    ubX = Num[] # upper bound on w
    lbU = Num[] # lower bound on w
    ubU = Num[] # upper bound on w
    g = Num[]   # equality constraints
    L = 0
    x0 = vvariable("x0", nx)
    x = vvariable("x1", nx) # initial value variable
    xr = vvariable("xr", (N+1)*nx)
    ur = vvariable("ur", N*nu)

    xg = vvariable("xg", (N+1)*nx) # x_guess in paper
    ug = vvariable("ug", N*nu)

    xr = reshape(xr, nx, :)
    ur = reshape(ur, nu, :)
    xg = reshape(xg, nx, :)
    ug = reshape(ug, nu, :)

    append!(g, x0 - xg[:,1] - x)
    append!(X, x)
    append!(lbX, x) # Initial state is fixed
    append!(ubX, x)

    u = zeros(nu)#vvariable("u0", nu)
    n = 1
    for n = 1:prob.N # for whole time horizon N
        ulast = u
        u = vvariable("u$n", nu)
        append!(U, u)
        append!(lbU, constraints.umin)
        append!(ubU, constraints.umax)

        L += loss(x+xg[:,n]-xr[:,n], u+ug[:,n]-ur[:,n], ulast, n, prob) # discrete integration of loss instead of rk4 integration, this makes the hessian constant
        # append!(lbX, constraints.xmin)
        # append!(ubX, constraints.xmax)
        f_nl = prob.dynamics(xg[:,n], ug[:,n], 0, 0)
        A = Symbolics.jacobian(f_nl, xg[:,n])
        B = Symbolics.jacobian(f_nl, ug[:,n])
        xp = A*x + B*u
        x = vvariable("x$(n+1)", nx) # x in next time point
        append!(X, x)
        r = prob.dynamics(xg[:,n], ug[:,n], 0, 0) - xg[:,n+1] # stationary point residual
        append!(g, xp - x + r) # propagated x is x in next time point
    end
    L += final_cost(x+xg[:,N+1]-xr[:,N+1], prob)

    # append!(g, lbX - X)
    # append!(g, U - lbU)

    w = [X;U]
    P = Symbolics.sparsehessian(L, w)
    q = Symbolics.gradient(L, w) - P*w # if xg and xr are included everywhere above, this line is equivalent to the line below, otherwise not
    # q = P*([vec(xg); vec(ug)] - [vec(xr); vec(ur)]) # we remove the linearization point x due to (12c) in http://cse.lab.imtlucca.it/~bemporad/publications/papers/ijc_rtiltv.pdf


    Aeq = Symbolics.sparsejacobian(g, w)
    leq = [-x0; zeros((N+1) * nx)] # these are currently not used
    ueq = leq
    
    # lineq = [lbX; lbU]
    # uineq = [ubX; ubU]
    # Aineq = Symbolics.jacobian(uineq, w, x0)

    # A, l, u = [Aeq; Aineq], [leq; lineq], [ueq; uineq]
    A,l,u = Aeq, leq, ueq

    lossfun = build_function(L, w, x0, xr, ur, xg, ug, expression=Val{false})

    SymbolicMPC(
        build_function(P, w, xg, ug, expression=Val{false})[1],
        build_function(q, w, xr, ur, xg, ug, expression=Val{false})[1],
        build_function(A, w, x0, xg, ug, expression=Val{false})[1],
        build_function(g, w, x0, xg, ug, expression=Val{false})[1],
        build_function(l, w, x0, expression=Val{false})[1],
        build_function(u, w, x0, expression=Val{false})[1],
    ), (; P,q,A,g,w,xr,ur,xg,ug,x0,L,lossfun)
end

function symbolic_solve(prob, funs, x0, xr=zeros(length(x0), prob.N+1), ur=zeros(prob.nu, prob.N); T)
    @unpack nx, nu, N = prob
    np = (N+1)*nx + N*nu
    xinds = (1:(N+1)*nx)
    uinds = (1:N*nu) .+ (N+1)*nx
    @assert uinds[end] == np
    wk = [vec(xr); zeros(N*nu)]
    @assert length(wk) == np

    # wk = [vec(xr); -9.33; zeros((N-1)*nu)]
    xg = reshape(wk[xinds], nx, :)
    ug = reshape(wk[uinds], nu, :)
    x_current = x0
    x_all = []
    u_all = []
    model = OSQP.Model()
    local res
    gi = funs.g(0*wk, x_current, xg, ug)

    OSQP.setup!(
        model;
        P = funs.P(wk, xg, ug),
        q = funs.q(wk, xr, ur, xg, ug),
        A = funs.A(wk, x_current, xg, ug),
        l=-gi,
        u=-gi,
        verbose=false,
        eps_rel=1e-10,
        eps_abs=1e-10,
        polish=true,
        max_iter=10000,
    )
    n = 1
    for n in 1:T
        # @show n
        push!(x_all, x_current)
        xg[:, 1] .= x_current
        # n > 1 && warmstart(model, res, prob)
        MPC.rollout!(prob.dynamics, xg, ug)
        for sqp_iter in 1:prob.solver.solver.sqp_iters
            #         # Evaluate jacobians and constraints
            gi = funs.g(0*wk, x_current, xg, ug) # gi is the constraint evaluated at (Δx = Δu = 0)
            OSQP.update!(
                model;
                Px = funs.P(wk, xg, ug).nzval,
                q  = funs.q(wk, xr, ur, xg, ug),
                Ax = funs.A(wk, x_current, xg, ug).nzval,
                l  = -gi,
                u  = -gi,
            )
            res = OSQP.solve!(model)
            w_opt =  res.x
            # @show norm(w_opt)
            any(!isfinite, w_opt) && error("Infinite")
            wk += w_opt
            xg = reshape(wk[xinds], nx, :)
            ug = reshape(wk[uinds], nu, :)
            if sqp_iter == prob.solver.solver.sqp_iters
                # plot(vec(ug)) |> display
                # sleep(0.01)
            end
        end
        @assert norm(xg[:,1] - x_current) < 1e-10
        @assert maximum(gi) < 1e-6
        u_opt = ug[1:nu]
        x_current = prob.dynamics(x_current, u_opt, 0, 0)
        push!(u_all, u_opt)

        # wk .= 0
        copyto!(wk, 1, wk, nx+1, N*nx) # forward states
        copyto!(wk, (N+1)*nx+1, wk, (N+1)*nx+1+nu, (N-1)*nu) # forward control
    end

    reduce(hcat, x_all), reduce(hcat, u_all)
end
