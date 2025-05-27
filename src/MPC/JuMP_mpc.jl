using JuMP
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


#=
@variable(model, begin
    A[1:nx, 1:nx]
    B[1:nx, 1:nu]
    C[1:ny, 1:nx]
    D[1:ny, 1:nu]
    K[1:nx, 1:ny]
    x[1:nx, 1:N]
end)
=#

function build_jump_problem(prob::AbstractMPCProblem, xr, ur, x0)
    model = Model(OSQP.Optimizer)

    @variable(model, x[1:nx, 1:N+1]) 
    @variable(model, u[1:nu, 1:N]) 


    @constraint(model, x0 .== x[:,1])

    u0 = zeros(nu)#@variable(model, u0", nu]
    n = 1
    L = 0
    for n = 1:prob.N # for whole time horizon N
        if n > 1
            L += loss(x[:,n]-xr[:,n], u[:,n]-ur[:,n], u[:, n-1], n, prob) 
        else
            L += loss(x[:,n]-xr[:,n], u[:,n]-ur[:,n], u0, n, prob) 
        end
        # JuMP.@NLexpression(model, xp, prob.dynamics(x[:,n], u[:,n], 0, 0))
        # @constraint(model, x[:, n+1] .== xp)

        
        @NLconstraint(model, x[1, n+1] == prob.dynamics(x[:,n], u[:,n], 0, 0)[1])
    end
    L += final_cost(x[:, N+1]-xr[:,N+1], prob)
    @objective(model, Min, L)
    model

end

function jump_solve(prob, x0, xr=zeros(length(x0), prob.N+1), ur=zeros(prob.nu, prob.N); T)
    @unpack nx, nu, N = prob
    np = (N+1)*nx + N*nu

    x_current = x0
    x_all = []
    u_all = []

    xg = repeat(x0, 1, N+1)
    ug = zeros(nu, N)

    local res

    n = 1
    for n in 1:T
        # @show n
        push!(x_all, x_current)
        MPC.rollout!(prob.dynamics, xg, ug)

        model = build_jump_problem(prob, xr,ur,x_current)
        JuMP.optimize!(model)
            
        u_opt = ug[1:nu]
        x_current = prob.dynamics(x_current, u_opt, 0, 0)
        push!(u_all, u_opt)

        # wk .= 0
        # copyto!(wk, 1, wk, nx+1, N*nx) # forward states
        # copyto!(wk, (N+1)*nx+1, wk, (N+1)*nx+1+nu, (N-1)*nu) # forward control
    end

    reduce(hcat, x_all), reduce(hcat, u_all)
end

# X, U = jump_solve(prob, ones(2); T=10)