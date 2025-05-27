# Same model as RCAM but without variables in sub-systems, more suitable for webinars/documentation. 

using ModelingToolkit
using OrdinaryDiffEq
using DyadControlSystems
using LinearAlgebra:cross, inv
using Plots
#using OptimizationMOI, Ipopt
#using DyadControlSystems: observed_state_substituter
#using Optimization
#using OptimizationOptimJL
#using ModelingToolkit:AbstractSystem, io_preprocessing

# Define a dictionary of constants 

rcam_constants = Dict{Symbol, Any}(:m => 120000, :c̄ => 6.6, :lt => 24.8, :S => 260, :St => 64, :Xcg => 0.23 * 6.6,:Ycg => 0.0, :Zcg => 0.10 * 6.6, :Xac => 0.12 * 6.6, :Yac => 0.0, :Zac => 0.0, :Xapt1 => 0, :Yapt1 => -7.94, :Zapt1 => -1.9, :Xapt2 => 0, :Yapt2 => 7.94, :Zapt2 => -1.9, :g => 9.81, :depsda => 0.25, :α_L0 => deg2rad(-11.5), :n => 5.5, :a3 => -768.5, :a2 => 609.2, :a1 => -155.2, :a0 => 15.212, :α_switch => deg2rad(14.5), :ρ => 1.225, 
)


Ib_mat =  [40.07          0.0         -2.0923
            0.0            64.0        0.0  
            -2.0923        0.0         99.92]*rcam_constants[:m]
Ib = Ib_mat
invIb = inv(Ib_mat)

ps = @parameters(
        ρ             = rcam_constants[:ρ], [description="kg/m3 - air density"],
        m             = rcam_constants[:m], [description="kg - total mass"],
        c̄             = rcam_constants[:c̄], [description="m - mean aerodynamic chord"],
        lt            = rcam_constants[:lt], [description="m - tail AC distance to CG"],
        S             = rcam_constants[:S], [description="m2 - wing area"],
        St            = rcam_constants[:St], [description="m2 - tail area"],
        Xcg           = rcam_constants[:Xcg], [description="m - x pos of CG in Fm"],
        Ycg           = rcam_constants[:Ycg], [description="m - y pos of CG in Fm"],
        Zcg           = rcam_constants[:Zcg], [description="m - z pos of CG in Fm"],
        Xac           = rcam_constants[:Xac], [description="m - x pos of aerodynamic center in Fm"],
        Yac           = rcam_constants[:Yac], [description="m - y pos of aerodynamic center in Fm"],
        Zac           = rcam_constants[:Zac], [description="m - z pos of aerodynamic center in Fm"],
        Xapt1         = rcam_constants[:Xapt1], [description="m - x position of engine 1 in Fm"],
        Yapt1         = rcam_constants[:Yapt1], [description="m - y position of engine 1 in Fm"],
        Zapt1         = rcam_constants[:Zapt1], [description="m - z position of engine 1 in Fm"],
        Xapt2         = rcam_constants[:Xapt2], [description="m - x position of engine 2 in Fm"],
        Yapt2         = rcam_constants[:Yapt2], [description="m - y position of engine 2 in Fm"],
        Zapt2         = rcam_constants[:Zapt2], [description="m - z position of engine 2 in Fm"],
        g             = rcam_constants[:g], [description="m/s2 - gravity"],
        depsda        = rcam_constants[:depsda], [description="rad/rad - change in downwash wrt α"],
        α_L0          = rcam_constants[:α_L0], [description="rad - zero lift AOA"],
        n             = rcam_constants[:n], [description="adm - slope of linear ragion of lift slope"],
        a3            = rcam_constants[:a3], [description="adm - coeff of α^3"],
        a2            = rcam_constants[:a2], [description="adm -  - coeff of α^2"],
        a1            = rcam_constants[:a1], [description="adm -  - coeff of α^1"],
        a0            = rcam_constants[:a0], [description="adm -  - coeff of α^0"],
        α_switch      = rcam_constants[:α_switch], [description="rad - kink point of lift slope"],
    )

    @parameters t
    D =  Differential(t)
    
    # States 
    V_b = @variables(
        u(t), [description="translational velocity along x-axis [m/s]"],
        v(t), [description="translational velocity along y-axis [m/s]"],
        w(t), [description="translational velocity along z-axis [m/s]"]
        )

    wbe_b = @variables(
        p(t), [description="rotational velocity about x-axis [rad/s]"], 
        q(t), [description="rotational velocity about y-axis [rad/s]"],
        r(t), [description="rotational velocity about z-axis [rad/s]"]
        )
    
    rot = @variables(
            ϕ(t), [description="rotation angle about x-axis/roll or bank angle [rad]"], 
            θ(t), [description="rotation angle about y-axis/pitch angle [rad]"], 
            ψ(t), [description="rotation angle about z-axis/yaw angle [rad]"]
            )
    
    # Controls
    U = @variables(
        uA(t), [description="aileron [rad]"],
        uT(t), [description="tail [rad]"],
        uR(t), [description="rudder [rad]"],
        uE_1(t), [description="throttle 1 [rad]"],
        uE_2(t), [description="throttle 2 [rad]"],
    )


    # Auxiliary Variables to define model.
    Auxiliary_vars = @variables Va(t) α(t) β(t) Q(t) CL_wb(t) ϵ(t) α_t(t) CL_t(t) CL(t) CD(t) CY(t) F1(t) F2(t)
    
    @variables FA_s(t)[1:3] C_bs(t)[1:3,1:3] FA_b(t)[1:3] eta(t)[1:3] dCMdx(t)[1:3, 1:3] dCMdu(t)[1:3, 1:3] CMac_b(t)[1:3] MAac_b(t)[1:3] rcg_b(t)[1:3] rac_b(t)[1:3] MAcg_b(t)[1:3] FE1_b(t)[1:3] FE2_b(t)[1:3] FE_b(t)[1:3] mew1(t)[1:3] mew2(t)[1:3] MEcg1_b(t)[1:3] MEcg2_b(t)[1:3] MEcg_b(t)[1:3] g_b(t)[1:3] Fg_b(t)[1:3] F_b(t)[1:3] Mcg_b(t)[1:3] H_phi(t)[1:3,1:3]

    # Scalarizing all the array variables. 

    FA_s = collect(FA_s)
    C_bs = collect(C_bs)
    FA_b = collect(FA_b)
    eta = collect(eta)
    dCMdx = collect(dCMdx)
    dCMdu = collect(dCMdu) 
    CMac_b =  collect(CMac_b)
    MAac_b = collect(MAac_b)
    rcg_b = collect(rcg_b)
    rac_b = collect(rac_b)
    MAcg_b = collect(MAcg_b)
    FE1_b = collect(FE1_b)
    FE2_b = collect(FE2_b)
    FE_b = collect(FE_b)
    mew1 = collect(mew1)
    mew2 = collect(mew2)
    MEcg1_b = collect(MEcg1_b)
    MEcg2_b = collect(MEcg2_b)
    MEcg_b = collect(MEcg_b)
    g_b = collect(g_b)
    Fg_b = collect(Fg_b)
    F_b = collect(F_b)
    Mcg_b = collect(Mcg_b)
    H_phi = collect(H_phi)

    array_vars = vcat(vec(FA_s), vec(C_bs), vec(FA_b), vec(eta), vec(dCMdx), vec(dCMdu), vec(CMac_b), vec(MAac_b), vec(rcg_b), vec(rac_b), vec(MAcg_b), vec(FE1_b), vec(FE2_b), vec(FE_b), vec(mew1), vec(mew2), vec(MEcg1_b), vec(MEcg2_b), vec(MEcg_b), vec(g_b), vec(Fg_b), vec(F_b), vec(Mcg_b), vec(H_phi))

    eqns =[
    # Step 1. Intermediate variables 
    # Airspeed
    Va ~ sqrt(u^2 + v^2 + w^2)

    # α and β
    α ~ atan(w,u)
    β ~ asin(v/Va)

    # dynamic pressure
    Q ~ 0.5*ρ*Va^2

    
    # Step 2. Aerodynamic Force Coefficients
    # CL - wing + body
    #CL_wb ~  n*(α - α_L0)

    CL_wb ~ ifelse(α <= α_switch, n*(α - α_L0), a3*α^3 + a2*α^2 + a1*α + a0)

    # CL thrust
    ϵ ~ depsda*(α - α_L0)
    α_t ~ α - ϵ + uT + 1.3*q*lt/Va
    CL_t ~ 3.1*(St/S) * α_t
    
    # Total CL
    CL ~ CL_wb + CL_t

    # Total CD
    CD ~ 0.13 + 0.07 * (n*α + 0.654)^2

    # Total CY
    CY ~ -1.6*β + 0.24*uR
    
    
    # Step 3. Dimensional Aerodynamic Forces
    # Forces in F_s
    FA_s .~ [-CD * Q * S
                CY * Q * S
                -CL * Q * S] 

    
    # rotate forces to body axis (F_b)  
    vec(C_bs .~ [cos(α)      0.0      -sin(α)
                    0.0             1.0      0.0
                    sin(α)      0.0      cos(α)])
    
    
    FA_b .~ C_bs*FA_s 
    
    # Step 4. Aerodynamic moment coefficients about AC
    # moments in F_b
    eta .~ [ -1.4 * β 
                -0.59 - (3.1 * (St * lt) / (S * c̄)) * (α - ϵ)
            (1 - α * (180 / (15 * π))) * β
    ]
    
    
    vec(dCMdx .~ (c̄ / Va)*        [-11.0              0.0                           5.0
                                        0.0   (-4.03 * (St * lt^2) / (S * c̄^2))        0.0
                                        1.7                 0.0                          -11.5])
    
    
    vec(dCMdu .~ [-0.6                   0.0                 0.22
                    0.0   (-3.1 * (St * lt) / (S * c̄))      0.0
                    0.0                    0.0                -0.63])

    # CM about AC in Fb
    CMac_b .~ eta + dCMdx*wbe_b + dCMdu*[uA
                                        uT
                                        uR]
                                        
    # Step 5. Aerodynamic moment about AC 
    # normalize to aerodynamic moment
    MAac_b .~ CMac_b * Q * S * c̄

    # Step 6. Aerodynamic moment about CG
    rcg_b .~    [Xcg
                Ycg
                Zcg]

    rac_b .~ [Xac
            Yac
            Zac]
    
    MAcg_b .~ MAac_b + cross(FA_b, rcg_b - rac_b)

    # Step 7. Engine force and moment
    # thrust
    F1 ~ uE_1 * m * g
    F2 ~ uE_2 * m * g
    
    # thrust vectors (assuming aligned with x axis)
    FE1_b .~ [F1
                0
                0]

    FE2_b .~ [F2
                0
                0]
    
    FE_b .~ FE1_b + FE2_b
    
    # engine moments
    mew1 .~  [Xcg - Xapt1
            Yapt1 - Ycg
            Zcg - Zapt1]

    mew2 .~ [ Xcg - Xapt2
            Yapt2 - Ycg
            Zcg - Zapt2]
    
    MEcg1_b .~ cross(mew1, FE1_b)
    MEcg2_b .~ cross(mew2, FE2_b)
    
    MEcg_b .~ MEcg1_b + MEcg2_b

    # Step 8. Gravity effects
    g_b .~ [-g * sin(θ)
            g * cos(θ) * sin(ϕ)
            g * cos(θ) * cos(ϕ)]

    Fg_b .~ m * g_b

    # Step 9: State derivatives
    
    # form F_b and calculate u, v, w dot
    F_b .~ Fg_b + FE_b + FA_b
    
    D.(V_b) .~ (1 / m)*F_b - cross(wbe_b, V_b)
    
    # form Mcg_b and calc p, q r dot
    Mcg_b .~ MAcg_b + MEcg_b
    
    D.(wbe_b) .~ invIb*(Mcg_b - cross(wbe_b, Ib*wbe_b))

    #phi, theta, psi dot
    vec(H_phi .~ [1.0         sin(ϕ)*tan(θ)       cos(ϕ)*tan(θ)
                    0.0         cos(ϕ)              -sin(ϕ)
                    0.0         sin(ϕ)/cos(θ)       cos(ϕ)/cos(θ)])

    D.(rot) .~  H_phi*wbe_b        
    ]

    all_vars = vcat(V_b,wbe_b,rot,U, Auxiliary_vars, array_vars)
    
    @named rcam_model = ODESystem(eqns, t, all_vars, ps)

    # SIMULATION 
    #=

    inputs = [uA, uT, uR, uE_1, uE_2]
    outputs = []
    sys, diff_idxs, alge_idxs, input_idxs = ModelingToolkit.io_preprocessing(rcam_model, inputs, outputs)

    # Converting ODESystem to ODEProblem for numerical simulation.

    x0 = Dict(
        u => 85,
        v => 0,
        w => 0,
        p => 0,
        q => 0,
        r => 0,
        ϕ => 0,
        θ => 0.1, # approx 5.73 degrees
        ψ => 0
    )

    u0 = Dict(
        uA => 0,
        uT => -0.1,
        uR => 0,
        uE_1 => 0.08,
        uE_2 => 0.08
    )

    tspan = (0.0, 3*60)
    prob = ODEProblem(sys, x0, tspan, u0, jac = true)
    sol = solve(prob, Tsit5())
    plotvars = [u,
                v,
                w,
                p,
                q,
                r,
                ϕ,
                θ,
                ψ,]

    plot(sol, idxs = plotvars, layout = length(plotvars))
=#

# TRIMMING: CONSTRAINED.


#=u0 = Dict(
    uA => 0,
    uT => -0.1,
    uR => 0,
    uE_1 => 0.08,
    uE_2 => 0.08
    )
=#

    inputs = [uA, uT, uR, uE_1, uE_2]
    outputs = []
    trimmed_states = [u,
                v,
                w,
                p,
                q,
                r,
                ϕ,
                θ,
                ψ]
    trimmed_controls = [
                uA,
                uT,
                uR,
                uE_1,
                uE_2]
    

# Trimming:

    # Defaults for trim
    #=
    inputs = []
    outputs = []
    params = Dict()
    desired_states = Dict()
    hard_eq_cons = Equation[]
    soft_eq_cons = Equation[]
    hard_ineq_cons = Num[]
    soft_ineq_cons = Num[]
    penalty_multipliers = Dict(:desired_states => 1.0, :trim_cons => 1.0, :sys_eqns => 10.0)
    dualize = false
    verbose = false
    abstol = 0.0
    simplify = false
    kwargs = []
    =#

    
    #desired_states = Dict(u => 85, v => 0, ϕ => 0, ψ => 0, uT => -0.2, uE_1 => 0.08)
    desired_states = Dict(u => 85, v => 0, ϕ => 0, ψ => 0, uE_1 => 0.1, uE_2 => 0.1)
    hard_eq_cons =  [
        Va ~ 85
        θ - atan(w,u) ~ 0.0
        #uE_1 ~ uE_2
    ]

    # These represent saturation limits on controllers. 
    hard_ineq_cons = [
        - uA + deg2rad(-25)    
        uA - deg2rad(25)
        -uT + deg2rad(-25)
        uT - deg2rad(10)
        -uR + deg2rad(-30)
        uR - deg2rad(30)
        -uE_1 + deg2rad(0.5) 
        uE_1 - deg2rad(10)
        -uE_2 + deg2rad(0.5) 
        uE_2 - deg2rad(10)
    ]

    penalty_multipliers = Dict(:desired_states => 10.0, :trim_cons => 10.0, :sys_eqns => 10.0)
    #solver = Ipopt.Optimizer()
    
    #println("Before opt")
    #using Optimization
    #using OptimizationOptimJL
    #solver = IPNewton()
    #sol, trim_states = trim(rcam_model; desired_states, inputs, hard_eq_cons, hard_ineq_cons, params = u0, solver)
    sol, trim_states = trim(rcam_model; penalty_multipliers, desired_states, inputs, hard_eq_cons, hard_ineq_cons)
    for elem in trimmed_states
        println("$elem:    ", round(trim_states[elem], digits = 4))
    end
    for elem in trimmed_controls
        println("$elem:    ", round(trim_states[elem], digits = 4))
    end
    
    ##############################################
    # SIMULATING THE TRIM State
    
    inputs = [uA, uT, uR, uE_1, uE_2]
    outputs = []
    sys, diff_idxs, alge_idxs, input_idxs = ModelingToolkit.io_preprocessing(rcam_model, inputs, outputs)

    # Converting ODESystem to ODEProblem for numerical simulation.
    
    x0 = filter(trim_states) do (k,v)
        k ∈ Set(trimmed_states)
    end

    u0 = filter(trim_states) do (k,v)
        k ∈ Set(trimmed_controls)
    end

    #=
    Dict(
        u => trim_states[u],
        v => trim_states[u],
        w => trim_states[u],
        p => trim_states[u],
        q => trim_states[u],
        r => trim_states[u],
        ϕ => trim_states[u],
        θ => trim_states[u], # approx 5.73 degrees
        ψ => trim_states[u]
    )

    u0 = Dict(
        uA => 0,
        uT => -0.1,
        uR => 0,
        uE_1 => 0.08,
        uE_2 => 0.08
    )
    =#
    tspan = (0.0, 3*60)
    prob = ODEProblem(sys, x0, tspan, u0, jac = true)
    sol = solve(prob, Tsit5())
    plotvars = [u,
                v,
                w,
                p,
                q,
                r,
                ϕ,
                θ,
                ψ,]
plot(sol, idxs = plotvars, layout = length(plotvars))
    


    #=


        #0. Check DOFs available for optimization. NOTE: It is not always an issue since sys_equations with derivative terms on the lhs that the user specifies are removed from constraints.
        size(inputs, 1) + size(outputs, 1) ≥ size(hard_eq_cons, 1) || @warn "The trim specifications may result in an over-constrained system. Consider removing a `hard_eq_cons` equation or setting `dualize` to `true`."

        # 1. Get all user-defined states. Logic: For any derivative term that appears in `user_specified_variables`, discard the corresponding reduced system equation that has that specified derivative (or variable for DAE's) if it exists from the optimization problem. 
        # E.g., if the user includes a derivative in the `desired_states` dictionary such as `D(H) => 2.0`, then if `D(H)` appears in the reduced system equations, its `rhs` should NOT be constrained to 0.    
        # This logic should also hold if any of the reduced equations are algebraic i.e., when the lhs is not a derivative term. NOT collect_differential_variables, maybe with vars! (TBD??).
        user_trim_eqn_form = vcat(keys(desired_states) .~ 0.0, hard_eq_cons, soft_eq_cons, hard_ineq_cons .~ 0.0, soft_ineq_cons .~ 0.0)  # Complete with 0.0 to get equation forms.  
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
            v == nothing
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
        objective += sum(max(0.0, row)*penalty_multipliers[:trim_cons] for row in soft_ineq_cons; init = 0)


        #if dualize
            verbose && @warn "Hard constraints and system equations are dualized "
            # 7a. Dualizing hard equality constraints
            objective += sum(abs2(eq.lhs - eq.rhs)*penalty_multipliers[:trim_cons] for eq in hard_eq_cons; init = 0)

            # 7b. Dualizing hard inequality constraints
            objective += sum(max(0, abs2(row))*penalty_multipliers[:trim_cons] for row in hard_ineq_cons; init = 0)

            # 7c. Dualizing free system equations
            objective += sum(abs2(eq.rhs)*penalty_multipliers[:sys_eqns] for eq in free_sys_eqns; init = 0)
            objective = obs_state_substituter(objective)
            @named opt_sys = OptimizationSystem(objective, collect(keys(optim_vars_states0)), collect(keys(params_dict)))
            prob = OptimizationProblem(opt_sys, optim_vars_states0, params_dict; grad = true, hess = true, sense = Optimization.MinSense)
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

        constraints = obs_state_substituter.(constraints)

        # Allowing some user-specified tolerances for convergence
        lcons = lcons .- abstol
        ucons = ucons .+ abstol

        # Note that for the purposes of obs_state_substituter, the io variables still pass through as parameters(sys)

        # Formulating the OptimizationSystem and OptimizationProblem
        @named opt_sys = OptimizationSystem(objective, collect(keys(optim_vars_states0)), collect(keys(params_dict)); constraints = constraints)#, defaults = ModelingToolkit.defaults(sys))
        prob = OptimizationProblem(opt_sys, optim_vars_states0, params_dict; lcons = lcons, ucons = ucons, grad = true, hess = true, sense = Optimization.MinSense)
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
    =#