function get_symvars(vars)
    x_, u_ = get_xu(vars)
    x = variables(:x, 1:size(x_, 1), 1:size(x_, 2)*vars.n_robust)
    u = variables(:u, 1:size(u_, 1), 1:size(u_, 2)*vars.n_robust)
    x,u
end

function butcher_triangle(m)
    m = copy(m)
    for i in CartesianIndices(m)
        if i.I[1] > i.I[2]
            m[i] = false # Nullify lower triangle
        end
    end
    dropzeros(m)
end

function build_symbolic_lag_hessian(cons, loss, vars, robust_horizon, p; verbose=false)
    nu = vars.nu
    n_robust = vars.n_robust

    nc = length(cons.constraint)*n_robust + nu*robust_horizon*(n_robust-1)

    x_,u_ = get_symvars(vars)
    symvars = [vec(x_); vec(u_)]
    mu = variables(:mu, 1:nc)
    @variables sigma
    cache = similar(mu)
    lag = (x, sigma, mu, args...)->sigma*loss(x, args...) + cons(cache, x, args...)'mu
    l = lag(symvars, sigma, mu, p)
    hs = Symbolics.sparsehessian(l, symvars) |> butcher_triangle
    lag_hess_prototype = safe_similar(hs, Float64)
    verbose && @info "Done"
    lag_h = build_function(hs.nzval, symvars, sigma, mu; cse=true, expression=Val{false}, iip_config = (false, true))[2];

    lag_h, lag_hess_prototype
end