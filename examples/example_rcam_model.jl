#=
    This provides a Julia implementation of the nonlinear Research Civil Aircraft Model (RCAM) developed by GARTEUR. http://garteur.org/wp-content/reports/FM/FM_AG-08_TP-088-3.pdf.
    This work relies on the derivations and errata presented by Christopher Lum (https://www.youtube.com/watch?v=bFFAL9lI2IQ). 
    In addition, this implementation uses a similar set-up to the open-source Python implementation https://github.com/flight-test-engineering/PSim-RCAM
    [TODO: Potentially put this as an example in ModelingToolkit]
=#


using ModelingToolkit
using OrdinaryDiffEq
using DyadControlSystems
using LinearAlgebra:cross, inv
using Plots
using OptimizationMOI, Ipopt
using DyadControlSystems: observed_state_substituter
using Optimization
using OptimizationOptimJL
using ModelingToolkit:AbstractSystem, io_preprocessing


rcam_constants = Dict(
    # Note: Constant values and comments taken from https://github.com/flight-test-engineering/PSim-RCAM
    :m => 120000, # kg - total mass
    :c̄ => 6.6, # m - mean aerodynamic chord
    :lt => 24.8, # m - tail AC distance to CG
    :S => 260, # m2 - wing area
    :St => 64, # m2 - tail area
    
    # centre of gravity position
    :Xcg => 0.23 * 6.6, # m - x pos of CG in Fm
    :Ycg => 0.0, # m - y pos of CG in Fm
    :Zcg => 0.10 * 6.6, # m - z pos of CG in Fm ERRATA - table 2.4 has 0.0 and table 2.5 has 0.10 c̄
    
    # aerodynamic center position
    :Xac => 0.12 * 6.6, # m - x pos of aerodynamic center in Fm
    :Yac => 0.0, # m - y pos of aerodynamic center in Fm
    :Zac => 0.0, # m - z pos of aerodynamic center in Fm
    
    # engine point of thrust aplication
    :Xapt1 => 0, # m - x position of engine 1 in Fm
    :Yapt1 => -7.94, # m - y position of engine 1 in Fm
    :Zapt1 => -1.9, # m - z position of engine 1 in Fm
    
    :Xapt2 => 0, # m - x position of engine 2 in Fm
    :Yapt2 => 7.94, # m - y position of engine 2 in Fm
    :Zapt2 => -1.9, # m - z position of engine 2 in Fm
    
    # other constants
    :g => 9.81, # m/s2 - gravity
    :depsda => 0.25, # rad/rad - change in downwash wrt α
    :α_L0 => deg2rad(-11.5), # rad - zero lift AOA
    :n => 5.5, # adm - slope of linear ragion of lift slope
    :a3 => -768.5, # adm - coeff of α^3
    :a2 => 609.2, # adm -  - coeff of α^2
    :a1 => -155.2, # adm -  - coeff of α^1
    :a0 => 15.212, # adm -  - coeff of α^0 ERRATA RCAM has 15.2. Modification suggested by C Lum. 
    :α_switch => deg2rad(14.5), # rad - kink point of lift slope
    :ρ => 1.225 # kg/m3 - air density
)

"RCAM aircraft model

            states,     x = u  # x-velocity [m/s]
                            v  # y-velocity [m/s]
                            w  # z-velocity [m/s]
                            p  # angular velocity about body x-axis [rad/s]
                            q  # angular velocity about body y-axis [rad/s]
                            r  # angular velocity about body z-axis [rad/s]
                            ϕ  # bank/roll angle [rad]
                            θ  # pitch angle [rad]
                            ψ  # yaw angle [rad]

            controls,   u = uA # aileron [rad]
                            uT # tail [rad]
                            uR # rudder [rad]
                            uE_1 # throttle 1 [rad]
                            uE_2 # throttle 2 [rad]
    " 
function RCAM_model(rcam_constants; name)
    @parameters t
    D =  Differential(t)
    
    # States 
    V_b = @variables u(t) v(t) w(t) # Translational velocity variables in body frame
    wbe_b = @variables p(t) q(t) r(t) # Rotational velocity variables in body frame
    rot = @variables ϕ(t) θ(t) ψ(t) # Rotation angle variables 
    
    # Controls
    U = @variables(
        uA(t), [description="aileron [rad]"],
        uT(t), [description="tail [rad]"],
        uR(t), [description="rudder [rad]"],
        uE_1(t), [description="throttle 1 [rad]"],
        uE_2(t), [description="throttle 2 [rad]"],
    )
    
    # Parameters
    ps = @parameters(
        ρ             = rcam_constants[:ρ], [description="kg/m3 - air density"],
        m             = rcam_constants[:m], [description="kg - total mass"],
        c̄          = rcam_constants[:c̄], [description="m - mean aerodynamic chord"],
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
        α_L0      = rcam_constants[:α_L0], [description="rad - zero lift AOA"],
        n             = rcam_constants[:n], [description="adm - slope of linear ragion of lift slope"],
        a3            = rcam_constants[:a3], [description="adm - coeff of α^3"],
        a2            = rcam_constants[:a2], [description="adm -  - coeff of α^2"],
        a1            = rcam_constants[:a1], [description="adm -  - coeff of α^1"],
        a0            = rcam_constants[:a0], [description="adm -  - coeff of α^0"],
        α_switch  = rcam_constants[:α_switch], [description="rad - kink point of lift slope"],
    )
    #defaults = Dict(p => rcam_constants[Symbol(p)] for p in ps)
    
    # Auxiliary Variables to define model. Most of these will be eliminated with `structural_simplify`.
    Auxiliary_vars = @variables Va(t) α(t) β(t) Q(t) CL_wb(t) ϵ(t) α_t(t) CL_t(t) CL(t) CD(t) CY(t) F1(t) F2(t)
    @variables FA_s(t)[1:3] C_bs(t)[1:3,1:3] FA_b(t)[1:3] eta(t)[1:3] dCMdx(t)[1:3, 1:3] dCMdu(t)[1:3, 1:3] CMac_b(t)[1:3] MAac_b(t)[1:3] rcg_b(t)[1:3] rac_b(t)[1:3] MAcg_b(t)[1:3] FE1_b(t)[1:3] FE2_b(t)[1:3] FE_b(t)[1:3] mew1(t)[1:3] mew2(t)[1:3] MEcg1_b(t)[1:3] MEcg2_b(t)[1:3] MEcg_b(t)[1:3] g_b(t)[1:3] Fg_b(t)[1:3] Ib(t)[1:3,1:3] invIb(t)[1:3,1:3] F_b(t)[1:3] Mcg_b(t)[1:3] H_phi(t)[1:3,1:3]
    
    # Scalarizing all the array variables. TODO: Develop a macro to do this. 
    
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
    Ib = collect(Ib)
    invIb = collect(invIb)
    F_b = collect(F_b)
    Mcg_b = collect(Mcg_b)
    H_phi = collect(H_phi)

    array_vars = [FA_s, C_bs, FA_b, eta, dCMdx, dCMdu, CMac_b, MAac_b, rcg_b, rac_b, MAcg_b, FE1_b, FE2_b, FE_b, mew1, mew2, MEcg1_b, MEcg2_b, MEcg_b, g_b, Fg_b, Ib, invIb, F_b, Mcg_b, H_phi]

    #FA_s = Vector{Num}(undef, 3) # NOTE: This approach won't work because you need to specify dependence on t. Seems best approach is to use Symbolic Arrays then scalarize.
 
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
        
    

        
        FA_b .~ C_bs*FA_s # Old code remark: 1: scalarizing an `Equation` scalarizes both sides - actually defined in Symbolics.jl. Only way I could get `eqns isa Vector{Equation}`. 2: Cross-check that it is normal matmult,
        
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
        
        #thrust vectors (assuming aligned with x axis)
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
        # inertia tensor
        
        vec(Ib .~ m * [40.07          0.0         -2.0923
                        0.0            64.0        0.0  
                        -2.0923        0.0         99.92]) # ERRATA on Ixz p. 12 vs p. 91
             
        vec(invIb .~ inv(Ib))
        
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
        
    # Procedure to get `all_vars isa Vector{Num}` for ODESystem. TODO: Any more elegant alternatives. Double broadcasting didn't work
    all_vars = vcat(V_b,wbe_b,rot,U, Auxiliary_vars)
    for arr in array_vars
        all_vars = vcat(all_vars, vec(arr))
    end
    ODESystem(eqns, t, all_vars, ps; name)
end

# Modeling the system mechanistically using the 9-step methodology presented by C. Lum.
@parameters t
@named rcam_1 = RCAM_model(rcam_constants)
@named full_sys = ODESystem([], t, systems = [rcam_1])
inputs = [rcam_1.uA, rcam_1.uT, rcam_1.uR, rcam_1.uE_1, rcam_1.uE_2]
outputs = []
sys, diff_idxs, alge_idxs, input_idxs = ModelingToolkit.io_preprocessing(full_sys, inputs, outputs)

# Converting ODESystem to ODEProblem for numerical simulation.
x0 = Dict(
    rcam_1.u => 85,
    rcam_1.v => 0,
    rcam_1.w => 0,
    rcam_1.p => 0,
    rcam_1.q => 0,
    rcam_1.r => 0,
    rcam_1.ϕ => 0,
    rcam_1.θ => 0.1, # approx 5.73 degrees
    rcam_1.ψ => 0
)

u0 = Dict(
    rcam_1.uA => 0,
    rcam_1.uT => -0.1,
    rcam_1.uR => 0,
    rcam_1.uE_1 => 0.08,
    rcam_1.uE_2 => 0.08
)

tspan = (0.0, 3*60)
prob = ODEProblem(sys, x0, tspan, u0, jac = true)
sol = solve(prob, Tsit5())
plotvars = [rcam_1.u,
            rcam_1.v,
            rcam_1.w,
            rcam_1.p,
            rcam_1.q,
            rcam_1.r,
            rcam_1.ϕ,
            rcam_1.θ,
            rcam_1.ψ,]
plot(sol, vars=plotvars, layout=length(plotvars))


# Getting identical results to C. Lum

##
using ControlSystemsBase
outputs = plotvars
matrices, simplified_sys = ModelingToolkit.linearize(full_sys, inputs, outputs; op = merge(x0, u0))
lsys = ss(matrices...)

# TODO: apply scaling to get all inputs and outputs in the approximate range ± 1, otherwise sigmaplot becomes hard to interpret

sigmaplot(lsys)
