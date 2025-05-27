mutable struct ModelReducer
    method::String
    sys
    sys_orig
    ps
    stab
    unstab
    n_unstable::Int
    coprime_info
    stable_inverse::Bool
end

function ModelReducer(method::String, sys)

    sys_schur, _, schurfact = schur_form(sminreal(sys))
    ps = schurfact.values
    n_unstable = if ControlSystemsBase.isdiscrete(sys)
		count(e->abs(e) > 1-sqrt(eps()), ps) # √ϵ is the default tolerance used by https://andreasvarga.github.io/DescriptorSystems.jl/dev/advanced_operations.html#DescriptorSystems.gsdec
	else
		count(e->real(e) > -sqrt(eps()), ps)
	end
    z = tzeros(sys_schur)
    stable_inverse = if ControlSystemsBase.isdiscrete(sys)
		all(e->abs(e) < 1-sqrt(eps()), z)
	else
		all(e->real(e) < -sqrt(eps()), z)
	end
    ModelReducer(method, sys_schur, sys, ps, nothing, nothing, n_unstable, nothing, stable_inverse)
end

available_orders(r::ModelReducer) = max(1, r.n_unstable):r.sys.nx

function model_reduction(r::ModelReducer, sys, n::Int; W, frequency_focus, residual, kwargs...)

    if r.sys_orig !== sys
        # Invalidate cache
        # r.stab = nothing
        # r.unstab = nothing
        # r.n_unstable = 0
        # r.coprime_info = nothing
        # r.sys = sys
        error("system has changed, recreate the model reducer")
    end

    n >= r.n_unstable || throw(ArgumentError("n must be at least the number of unstable modes ($(r.n_unstable))"))
    sysr, hs, err = if r.method == "coprime"
        # In this case we do not perform the stable/unstable decomposition
        sysr, hs, info = baltrunc_coprime(sys, r.coprime_info; n, residual, kwargs...)
        r.coprime_info = info
        err = RobustAndOptimalControl.DescriptorSystems.ghinfnorm(info.NMr-info.NM)[1]
        sysr, hs, err
    else
        if r.stab === nothing || r.unstab === nothing
            stab, unstab = RobustAndOptimalControl.DescriptorSystems.gsdec(dss(sys); job="stable")
            r.stab = stab   # Cache these for later use
            r.unstab = unstab
        end

        sysr, hs = if frequency_focus
            frequency_weighted_reduction(ss(r.stab), W, 1, n; residual, kwargs...)
        else
            sysr, hs = RobustAndOptimalControl.DescriptorSystems.gbalmr(r.stab; matchdc=residual, ord=n-r.n_unstable, kwargs...)
            ss(sysr), hs
        end
        @assert r.unstab.nx == r.n_unstable
        if r.unstab.nx > 0
            sysr = sysr + ss(r.unstab)
            hs = [fill(Inf, r.unstab.nx); hs]
        end
        err = if frequency_focus
            linfnorm2((W * I(sys.ny)) * (sys-sysr))[1]
        else
            linfnorm2(sys-sysr)[1]
        end
        sysr, hs, err
    end


    sysr, hs, err
end

function model_reduction(r::ModelReducer, sys, orders::AbstractVector{<:Integer}; kwargs...)
    sysr, hs, e = model_reduction(r, sys, orders[1]; kwargs...)
    systems = [sysr]
    errors = [e]
    for n in orders[min(length(orders), 2):end]
        sysr, _, e = model_reduction(r, sys, n; kwargs...)
        push!(systems, sysr)
        push!(errors, e)
    end
    systems, hs, errors
end
