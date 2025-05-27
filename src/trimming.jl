using ModelingToolkit, Optimization, OrdinaryDiffEq
using OptimizationOptimJL
using ModelingToolkit:AbstractSystem, io_preprocessing
using Symbolics:canonical_form

"""
    observed_state_substituter(sys::AbstractSystem)

Arguments: 
- `sys::AbstractSystem`: Simplified system attained after `structural_simplify`.

Returns a function that takes in any state and returns its equivalent expression in terms of the simplified states of the system given by `unknowns(sys)`.
Full behaviour of returned function (`r_func = observed_state_substituter(sys)`):
- `r_func(reduced_state)`: returns the `reduced_state` unchanged
- `r_func(observed_state)`: returns expression for `observed_state` in terms of reduced states
- `r_func(reduced_derivative)`: returns expression for `reduced_derivative` in terms of reduced states (also the corresponding rhs in `full_equations(sys)`)
- `r_func(observed_derivative)`: returns expression for `observed_derivative` in terms of reduced states
- `r_func(io_state)`: returns the `io_state` unchanged
- `r_func(param)`: returns the `param` unchanged
"""
function observed_state_substituter(sys::AbstractSystem)
    return Base.Fix2(ModelingToolkit.fixpoint_sub, Dict(eq.lhs => eq.rhs for eq in vcat(observed(sys), equations(sys))))
end

"""
    trim(sys::AbstractSystem;
        inputs = [], outputs = [], params = Dict(),
        desired_states = Dict(), hard_eq_cons = Equation[], soft_eq_cons = Equation[], hard_ineq_cons = Inequality[], soft_ineq_cons = Inequality[], 
        penalty_multipliers = Dict(:desired_states => 1.0, :trim_cons => 1.0, :sys_eqns => 10.0), dualize = false,
        solver = IPNewton(), verbose = false, simplify = false, kwargs...)  
    
Determine a trim point which is a feasible stationary point of the ODE closest to the specified desired state, given hard and soft constraints.

# Arguments:
- `sys`::AbstractSystem`: ODESystem to get a trim point for. Note that this should be the original system prior to calling `structural_simplify`
- `inputs = []`: array of (control) inputs to the system
- `outputs = []`: array of outputs from the system.
- `params`: Dictionary of parameters that will override defaults.
- `desired_states`: Dictionary specifying the desired operating point or derivatives.
- `hard_eq_cons = Equation[]`: User-defined hard equality constraints.   
- `soft_eq_cons = Equation[]`: User-defined soft equality constraints.
- `hard_ineq_cons = Inequality[]`: User-defined hard inequality constraints.
- `soft_ineq_cons = Inequality[]`: User-defined soft inequality constraints.
- `penalty_multipliers::AbstractDict`: The weighting of the `:desired_states`, `:trim_cons` (equality and inequality constraints), and `:sys_eqns` to the objective of the trimming optimization problem. Setting `dualize` to `true` and changing these multiples is useful in getting a trim point closer to user specifications.
- `dualize = false`: To dualize any user constraints as well as the system equations, set this to `true`. This solves an unconstrained optimization problem with the default solver changed to `Newton()`. 
- `solver = IPNewton()`: Specified optimization solver compatible with Optimization.jl. Note that the package for the interface to the solver needs to be loaded. By default only `OptimizationOptimJL` is added. In general, use a constrained optimization solver, so NOT `Newton()`.
- `verbose = false`: Set to `true` for information prints.
- `abstol = 0.0`: User-defined absolute tolerance for constraints and `ODESystem` equations. 
- `simplify = false`: Argument passed to `ModelingToolkit.structural_simplify`. Refer to documentation of `ModelingToolkit.jl`
- `kwargs`: Other keyword arguments passed to `ModelingToolkit.find_solvables!`

# Note:
- If several hard and soft-constraints are imposed, it is important to give a good (feasible) initial point e.g., through `desired` for `IPNewton()` or set `dualize` to `true`.

# Returns:
- 'sol::SciMLBase.OptimizationSolution': This provides information for the optimization solver. Refer to documentation of `SciMLBase.jl`  
- 'trim_states`: This provides a dictionary of all the states and derivatives of `sys` as well as their corresponding values at the trim point.     
"""
function trim(sys::AbstractSystem;
    inputs = [], outputs = [], params = Dict(),
    desired_states = Dict(), hard_eq_cons = Equation[], soft_eq_cons = Equation[], hard_ineq_cons = Inequality[], soft_ineq_cons = Inequality[], 
    penalty_multipliers = Dict(:desired_states => 1.0, :trim_cons => 1.0, :sys_eqns => 10.0), dualize = false,
    solver = IPNewton(), verbose = false, simplify = false, kwargs...)

        #0. Check DOFs available for optimization. NOTE: It is not always an issue since sys_equations with derivative terms on the lhs that the user specifies are removed from constraints.
        size(inputs, 1) + size(outputs, 1) ≥ size(hard_eq_cons, 1) || @warn "The trim specifications may result in an over-constrained system. Consider removing a `hard_eq_cons` equation or setting `dualize` to `true`."

        # 1. Get all user-defined states. Logic: For any derivative term that appears in `user_specified_variables`, discard the corresponding reduced system equation that has that specified derivative (or variable for DAE's) if it exists from the optimization problem. 
        # E.g., if the user includes a derivative in the `desired_states` dictionary such as `D(H) => 2.0`, then if `D(H)` appears in the reduced system equations, its `rhs` should NOT be constrained to 0.    
        # This logic should also hold if any of the reduced equations are algebraic i.e., when the lhs is not a derivative term. NOT collect_differential_variables, maybe with vars! (TBD??).
        user_trim_eqn_form = vcat(keys(desired_states) .~ 0.0, hard_eq_cons, soft_eq_cons, hard_ineq_cons, soft_ineq_cons)  # Complete with 0.0 to get equation forms.  
        # user_specified_variables = ModelingToolkit.collect_differential_variables(user_trim_eqn_form) [vars() below more general.]
        user_specified_variables = ModelingToolkit.vars(user_trim_eqn_form)

        # 2. Simplify the system. `obs_state_substituter` provides a map from any state to its equivalent expression in terms of simplified states. 
        sys, diff_idxs, alge_idxs, input_idxs = io_preprocessing(sys, inputs, outputs; simplify, kwargs...)
        ssys_eqns = equations(sys)
        free_sys_eqns = filter(eq -> eq.lhs ∉ user_specified_variables, ssys_eqns) # These are the simplified equations where the lhs derivative terms are not user specified, would be set to 0.0

        obs_state_substituter = observed_state_substituter(sys) # TODO: Some original variables (e.g., 2nd order derivatives) are renamed by structural_simplify. The user CANNOT specify these. Fixed point sub convergence/speed issues if equations(full_sys) are also included.

        # 3. Getting the input & output variables. These are optimization variables as well. 
        io_dict = merge(Dict(inputs .=> 0.0), Dict(outputs .=> 0.0)) # Ensures initial value.

        # Override with default values for io variables
        default_io = filter(ModelingToolkit.defaults(sys)) do (k,v)
            k in keys(io_dict)
        end
        io_dict = merge(io_dict, default_io)

        # If the user puts an io state in `desired_states` remove it and set the desired value as its initial value for optimization.
        # Any io states in the equality and inequality constraints will pass through `obs_state_substituter` unchanged as they are parameters(sys). 
        for (k,v) in desired_states
            if k in keys(io_dict)
                io_dict[k] = v
                pop!(desired_states, k)
            end
        end

        # 4. Substituting in the desired params defined by user
        all_params = filter(x -> x ∉ keys(io_dict), parameters(sys)) # io_preprocessing converts io states to params, but they are variables of the optimization problem.  
        params_dict = Dict(all_params .=> nothing)
        default_params = filter(ModelingToolkit.defaults(sys)) do (k,v)
            k in keys(params_dict)
        end
        params_dict = merge(params_dict, default_params)
        params_dict = merge(params_dict, params)

        undef_params = filter(params_dict) do (k,v)
            v === nothing
        end 
        
        isempty(undef_params) || error("The following parameters are not fully defined: $(keys(undef_params))")

        # 5. Reduced states and io states are the optimization variables. Setting initial values.  
        optim_vars_states0 = merge(Dict(unknowns(sys) .=> 0.0), io_dict)
        default_states = filter(ModelingToolkit.defaults(sys)) do (k,v)
            k in keys(optim_vars_states0)
        end
        optim_vars_states0 = merge(optim_vars_states0, default_states)
        desired_state_updates = filter(desired_states) do (k,v)
            k in keys(optim_vars_states0) # Don't want to have derivatives merged in as optimization variables.
        end
        optim_vars_states0 = merge(optim_vars_states0, desired_state_updates) # Override with desired states and defaults. 
        # TODO: Refactor with function that does these override steps with default states then desired states. 

        # 6. Formulate objective.
        # 6.a. Deal with desired states
        objective = sum(abs2(k - v)*penalty_multipliers[:desired_states] for (k,v) in desired_states; init = 0)

        #6.b. Dealing with soft equality constraints. 
        objective += sum(abs2(eq.lhs - eq.rhs)*penalty_multipliers[:trim_cons] for eq in soft_eq_cons; init = 0)

        #6.c. Dealing with soft inequality constraints.
        objective += sum(max(0.0, row.lhs)*penalty_multipliers[:trim_cons] for row in Symbolics.canonical_form.(soft_ineq_cons); init = 0)


        if dualize
            verbose && @warn "Hard constraints and system equations are dualized "
            # 7a. Dualizing hard equality constraints
            objective += sum(abs2(eq.lhs - eq.rhs)*penalty_multipliers[:trim_cons] for eq in hard_eq_cons; init = 0)

            # 7b. Dualizing hard inequality constraints
            objective += sum(max(0, abs2(row.lhs))*penalty_multipliers[:trim_cons] for row in Symbolics.canonical_form.(hard_ineq_cons); init = 0)

            # 7c. Dualizing free system equations
            objective += sum(abs2(eq.rhs)*penalty_multipliers[:sys_eqns] for eq in free_sys_eqns; init = 0)
            objective = obs_state_substituter(objective)
            @named opt_sys = OptimizationSystem(objective, collect(keys(optim_vars_states0)), collect(keys(params_dict)))
            # Handle the case when users provide some Ints and some Floats, varmap_to_vars doesnis called with tofloat=false in the OptimizationSystem constructor and this causes an assertion error.
            if !isconcretetype(valtype(optim_vars_states0))
                for (k,v) in pairs(optim_vars_states0)
                    optim_vars_states0[k] = float(v)
                end
            end
            if !isconcretetype(valtype(params_dict))
                for (k,v) in pairs(params_dict)
                    params_dict[k] = float(v)
                end
            end
            prob = OptimizationProblem(complete(opt_sys), optim_vars_states0, params_dict; grad = true, hess = true, sense = Optimization.MinSense)
            if solver == IPNewton()
                verbose && @warn "Changing solver to `Newton()`"
                solver = Newton()
            end
            sol = solve(prob, solver)
            u_opt = Dict(keys(optim_vars_states0) .=> sol.u)
            trim_states = merge(Dict(), u_opt) # Need to keep copy of u_opt
            for eq in ssys_eqns
                value = substitute(obs_state_substituter(eq.rhs), u_opt)
                value = substitute(value, params_dict)
                trim_states[Symbol(eq.lhs)] = value
            end
            for eq in observed(sys)
                value = substitute(obs_state_substituter(eq.rhs), u_opt)
                value = substitute(value, params_dict)
                trim_states[eq.lhs] = value
            end
            return sol, trim_states
        end

        # If not dualized
        objective = obs_state_substituter(objective)

        # 7. The constraints are now just carried forwards. No need to run a Symbolics.scalarize(constraints) here. Done in OptimizationSystem constructor.
        constraints = vcat(hard_eq_cons, hard_ineq_cons, map(eq -> eq.rhs ~ 0.0, free_sys_eqns))
        #=
        #7.a. Dealing with hard equality constraints. 
        constraints = map(eq -> eq.lhs - eq.rhs ~ 0.0, hard_eq_cons)
        lcons = zeros(length(hard_eq_cons))
        ucons = zeros(length(hard_eq_cons))

        #7.b. Dealing with hard inequality constraints.
        constraints = vcat(constraints, map(row -> 0.0 ~ row, hard_ineq_cons)) # Weird notation of Optimization.jl. row ~ 0.0 doesn't work, needs to be on rhs.
        lcons = vcat(lcons, fill(-Inf, length(hard_ineq_cons)))
        ucons = vcat(ucons, zeros(length(hard_ineq_cons)))

        #7.c Dealing with the free system equations
        constraints = vcat(constraints, map(eq -> eq.rhs ~ 0.0, free_sys_eqns))
        lcons = vcat(lcons, zeros(length(free_sys_eqns)))
        ucons = vcat(ucons, zeros(length(free_sys_eqns)))

        
        # Allowing some user-specified tolerances for convergence
        lcons = lcons .- abstol
        ucons = ucons .+ abstol
        =#

        constraints = obs_state_substituter.(constraints)

        # Note that for the purposes of obs_state_substituter, the io variables still pass through as parameters(sys)

        # Formulating the OptimizationSystem and OptimizationProblem
        @named opt_sys = OptimizationSystem(objective, collect(keys(optim_vars_states0)), collect(keys(params_dict)); constraints = constraints)#, defaults = ModelingToolkit.defaults(sys))
        # copt_sys = complete(opt_sys)
        prob = OptimizationProblem(complete(opt_sys), optim_vars_states0, params_dict; grad = true, hess = true, cons_j = true, cons_h = true, sense = Optimization.MinSense)
        sol = solve(prob, solver)
        u_opt = Dict(keys(optim_vars_states0) .=> sol.u)
        trim_states = merge(Dict(), u_opt) # Need to keep copy of u_opt
        for eq in ssys_eqns
            value = substitute(obs_state_substituter(eq.rhs), u_opt)
            value = substitute(value, params_dict)
            trim_states[eq.lhs] = value
        end
        for eq in observed(sys)
            value = substitute(obs_state_substituter(eq.rhs), u_opt)
            value = substitute(value, params_dict)
            trim_states[eq.lhs] = value
        end
    return sol, trim_states

    # - TBD: One cannot have terms of type `ODESystem` in the `desired_states`dictionary, hard or soft constraint equations e.g., one cannot set ```model.output ~ 1.4 ``` , where `model.output::RealOutput`
end

#= DEPRECATED CODE

"""
    trim(sys, adtype::SciMLBase.AbstractADType; desired_states = Dict(), solver = Newton(), verbose = false, kwargs...)
    
Determine a trim point which is a feasible stationary point of the ODE closest to the specified desired state.
# Arguments:
- `sys`::ODESystem`: ODESystem to get a trim point for.
- `desired`: dictionary of desired operating states.
- `solver = Newton()`: Specified optimization solver. Note that the package for the interface to the solver needs to be loaded. By default only `OptimizationOptimJL` is added.
- `verbose = false`: Set to `true` for information prints.
Returns:
- `sol`: The solution of the trimming optimization problem 
"""
function trim(sys, adtype::SciMLBase.AbstractADType; desired_states = Dict(), solver = Newton(), verbose = false, kwargs...)
    @warn "This function is deprecated. Use `trim(sys::AbstractSystem; inputs = [], outputs = [], params = Dict(), desired_states = Dict(), hard_eq_cons = Equation[], soft_eq_cons = Equation[], hard_ineq_cons = Num[], soft_ineq_cons = Num[], penalty_multipliers = Dict(:desired_states => 1.0, :trim_cons => 1.0, :sys_eqns => 10.0), dualize = false, solver = IPNewton(), verbose = false, abstol = 0.0, simplify = false, kwargs...)`"
    st_p_in = merge(ModelingToolkit.defaults(sys), desired_states)
    all_st = unknowns(sys)
    all_ps = parameters(sys)
    st_in = Dict() # Filters out only states
    ps_in = Dict() # Filter out only parameters
    x0 = Float64[]
    for st in all_st
        if haskey(st_p_in,st)
            st_in[st] = st_p_in[st]
        else
            verbose && @info("Initializating state $st to 0.0")
            st_in[st] = 0.0
        end
        push!(x0, convert(Float64 ,st_in[st]))
    end

    for p in all_ps
        if haskey(st_p_in,p)
            ps_in[p] = st_p_in[p]
        else
            error("Parameters $p undefined")
        end
    end
    eqs = equations(sys)
    diffeqs = filter(ModelingToolkit.isdiffeq, eqs) # To generalize later to DAE case.
      # Desired states are starting points for optimization
    obj_s = sum(abs2(ode.rhs) for ode in diffeqs; init = 0)
    for st in all_st
        if st in keys(desired_states)
            obj_s += abs2(st - st_in[st])
        end
    end
    obj_sub = substitute(obj_s, ps_in)
    obj_num = build_function(obj_sub, all_st; expression = Val{false})
    f = OptimizationFunction(obj_num, adtype)
    prob = OptimizationProblem(f, x0; sense = Optimization.MinSense)
    sol = solve(prob, solver)
    u_opt = Dict(all_st .=> sol.u)
    return sol, u_opt
end


# Trimming tool utilities
"""
    Internal function to extract out the states given a dictionary that may contain states or params.
    returns: `Set{Num}` containing the extracted states. 
"""
function get_participating_states(spec::AbstractDict, states_set)
    participating_st = filter(keys(spec)) do k
        k in states_set    
    end
    return participating_st
end

"""
    Internal function to extract out the states a vector of equations
    returns: `Set{Num}` containing the extracted states. 
"""
function get_participating_states(spec::Vector{Equation}, states_set)
    all_extracted_vars = []
    for eq in spec
        all_extracted_vars = union(all_extracted_vars, get_variables(eq.lhs), get_variables(eq.rhs))
    end
    filter!(all_extracted_vars) do elem
        elem in states_set# Filter out the parameters 
    end
    return all_extracted_vars
end

"""
    Returns all vars and derivatives. HOW TO IMPLEMENT THIS?
"""
function get_all_vars(spec::Vector{Equation})
end

function get_participating_states(spec::Vector{Num}, states_set)
    all_extracted_vars = []
    for row in spec
        all_extracted_vars = union(all_extracted_vars, get_variables(row))
    end
    filter!(all_extracted_vars) do elem
        elem in states_set# Filter out the parameters 
    end
    return all_extracted_vars
end

function get_participating_states(spec::Vector, states_set) # Fallback
end


"""
NOT TESTED! Marks the states provided in states are participating. 
"""
function markstates!(tearing_struct, states)
    fullvars = tearing_struct.fullvars
    participating_states = Dict(states .=> false)
    for (i, v) in enumerate(fullvars)
        if v in keys(participating_states)
            v = setmetadata(v, ModelingToolkit.VariableIrreducible, true)
            participating_states[v] = true
            fullvars[i] = v
        end
    end
    all(values(participating_states)) ||
        error("Some states were not correctly marked as participating ", participating_states)
    tearing_struct
end

"""
    io_state_preprocessing(sys::ModelingToolkit.AbstractSystem, inputs, outputs, irr_states; simplify = false, kwargs...)
    
This internal function marks the `inputs`, `outputs` array of states as not part of the `structural_simplify`. It also marks the `states` as participating.
    Based on `ModelingToolkit.io_preprocessing`. 
    
`kwargs` are passed to `ModelingToolkit._structural_simplify`.
"""
function io_state_preprocessing(sys::ModelingToolkit.AbstractSystem, inputs, outputs, irr_states; simplify = false, kwargs...)
    sys = expand_connections(sys)
    tearing_struct = TearingState(sys);
    ModelingToolkit.markio!(tearing_struct, inputs, outputs);
    markstates!(tearing_struct, irr_states);
    sys, input_idxs = ModelingToolkit._structural_simplify(sys, tearing_struct; simplify, check_bound = false,
                        kwargs...)

    eqs = equations(sys)
    ModelingToolkit.check_operator_variables(eqs, Differential)
    # Sort equations and states such that diff.eqs. match differential states and the rest are algebraic
    diffstates = ModelingToolkit.collect_operator_variables(sys, Differential)
    eqs = sort(eqs, by = e -> !isoperator(e.lhs, Differential),
    alg = Base.Sort.DEFAULT_STABLE)
    @set! sys.eqs = eqs
    diffstates = [arguments(e.lhs)[1] for e in eqs[1:length(diffstates)]]
    sts = [diffstates; setdiff(unknowns(sys), diffstates)]
    @set! sys.states = sts
    diff_idxs = 1:length(diffstates)
    alge_idxs = (length(diffstates) + 1):length(sts)

    sys, diff_idxs, alge_idxs, input_idxs
end
=#