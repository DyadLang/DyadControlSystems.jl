module ControlDemoSystems
using ModelingToolkit
import ControlSystemsBase
using LinearAlgebra: cross, inv
using IfElse
using StaticArrays
using DyadControlSystems: FunctionSystem, rk4, linearize, OperatingPoint, c2d, ss


"""
    rcam()
    
The nonlinear Research Civil Aircraft Model (RCAM) developed by [GARTEUR](http://garteur.org/wp-content/reports/FM/FM_AG-08_TP-088-3.pdf.) and using the derivations and errata presented by [Prof. Christopher Lum](https://www.youtube.com/watch?v=bFFAL9lI2IQ).

The RCAM model has the following 9 states:
- ``u, v, w`` : the translational velocity [m/s] in the body frame in the x, y and z-axes respectively, 
- ``p, q, r`` : the rotational velocity [rad/s] in the body frame about the x (roll/bank), y (pitch) and z (yaw)-axes respectively,  
- ``ϕ, θ, ψ`` : the rotation angles [rad] in the body frame about the x (roll/bank), y (pitch) and z (yaw)-axes respectively,

and the following 5 control inputs:
- ``u_A`` : the aileron angle [rad]
- ``u_T`` : the tail angle [rad]
- ``u_R`` : the rudder angle [rad]
- ``u_{E1}`` : the left engine/throttle position [rad] 
- ``u_{E2}`` : the right engine/throttle position [rad]

The mechanistic model is defined symbolically as an `ODESystem` in [ModelingToolkit](https://github.com/SciML/ModelingToolkit.jl) using the 9-step procedure and parameters detailed in this [lecture](https://www.youtube.com/watch?v=bFFAL9lI2IQ).

For more details on this example, as well as a tutorial on how to find a trim point, see [the documentation on trimming](https://help.juliahub.com/DyadControlSystems/dev/trimming/).

# Example:
```julia
rcam = DyadControlSystems.ControlDemoSystems.rcam();
display(rcam.rcam_model) # Full model
display(rcam.iosys)      # Model simplified using IO processing
rcam.x0 # Trim point (state)
rcam.u0 # Trim point (input)
```
"""
function rcam()

    rcam_constants = Dict(
        :m => 120000,
        :c̄ => 6.6,
        :lt => 24.8,
        :S => 260,
        :St => 64,
        :Xcg => 0.23 * 6.6,
        :Ycg => 0.0,
        :Zcg => 0.10 * 6.6,
        :Xac => 0.12 * 6.6,
        :Yac => 0.0,
        :Zac => 0.0,
        :Xapt1 => 0,
        :Yapt1 => -7.94,
        :Zapt1 => -1.9,
        :Xapt2 => 0,
        :Yapt2 => 7.94,
        :Zapt2 => -1.9,
        :g => 9.81,
        :depsda => 0.25,
        :α_L0 => deg2rad(-11.5),
        :n => 5.5,
        :a3 => -768.5,
        :a2 => 609.2,
        :a1 => -155.2,
        :a0 => 15.212,
        :α_switch => deg2rad(14.5),
        :ρ => 1.225,
    )

    Ib_mat = [
        40.07 0.0 -2.0923
        0.0 64.0 0.0
        -2.0923 0.0 99.92
    ] * rcam_constants[:m]

    Ib = Ib_mat
    invIb = inv(Ib_mat)

    ps = @parameters(
        ρ        = rcam_constants[:ρ],  [description = "kg/m3 - air density"],
        m        = rcam_constants[:m],  [description = "kg - total mass"],
        c̄       = rcam_constants[:c̄],  [description = "m - mean aerodynamic chord"],
        lt       = rcam_constants[:lt],  [description = "m - tail AC distance to CG"],
        S        = rcam_constants[:S],  [description = "m2 - wing area"],
        St       = rcam_constants[:St],  [description = "m2 - tail area"],
        Xcg      = rcam_constants[:Xcg],  [description = "m - x pos of CG in Fm"],
        Ycg      = rcam_constants[:Ycg],  [description = "m - y pos of CG in Fm"],
        Zcg      = rcam_constants[:Zcg],  [description = "m - z pos of CG in Fm"],
        Xac      = rcam_constants[:Xac],  [description = "m - x pos of aerodynamic center in Fm"],
        Yac      = rcam_constants[:Yac],  [description = "m - y pos of aerodynamic center in Fm"],
        Zac      = rcam_constants[:Zac],  [description = "m - z pos of aerodynamic center in Fm"],
        Xapt1    = rcam_constants[:Xapt1],  [description = "m - x position of engine 1 in Fm"],
        Yapt1    = rcam_constants[:Yapt1],  [description = "m - y position of engine 1 in Fm"],
        Zapt1    = rcam_constants[:Zapt1],  [description = "m - z position of engine 1 in Fm"],
        Xapt2    = rcam_constants[:Xapt2],  [description = "m - x position of engine 2 in Fm"],
        Yapt2    = rcam_constants[:Yapt2],  [description = "m - y position of engine 2 in Fm"],
        Zapt2    = rcam_constants[:Zapt2],  [description = "m - z position of engine 2 in Fm"],
        g        = rcam_constants[:g],  [description = "m/s2 - gravity"],
        depsda   = rcam_constants[:depsda],  [description = "rad/rad - change in downwash wrt α"],
        α_L0     = rcam_constants[:α_L0],  [description = "rad - zero lift AOA"],
        n        = rcam_constants[:n],  [description = "adm - slope of linear ragion of lift slope"],
        a3       = rcam_constants[:a3],  [description = "adm - coeff of α^3"],
        a2       = rcam_constants[:a2],  [description = "adm -  - coeff of α^2"],
        a1       = rcam_constants[:a1],  [description = "adm -  - coeff of α^1"],
        a0       = rcam_constants[:a0],  [description = "adm -  - coeff of α^0"],
        α_switch = rcam_constants[:α_switch],  [description = "rad - kink point of lift slope"]
    )


    @parameters t
    D = Differential(t)

    # States 
    V_b = @variables(
        u(t),
        [description = "translational velocity along x-axis [m/s]"],
        v(t),
        [description = "translational velocity along y-axis [m/s]"],
        w(t),
        [description = "translational velocity along z-axis [m/s]"]
    )

    wbe_b = @variables(
        p(t),
        [description = "rotational velocity about x-axis [rad/s]"],
        q(t),
        [description = "rotational velocity about y-axis [rad/s]"],
        r(t),
        [description = "rotational velocity about z-axis [rad/s]"]
    )

    rot = @variables(
        ϕ(t),
        [description = "rotation angle about x-axis/roll or bank angle [rad]"],
        θ(t),
        [description = "rotation angle about y-axis/pitch angle [rad]"],
        ψ(t),
        [description = "rotation angle about z-axis/yaw angle [rad]"]
    )

    # Controls
    U = @variables(
        uA(t),
        [description = "aileron [rad]", input=true],
        uT(t),
        [description = "tail [rad]", input=true],
        uR(t),
        [description = "rudder [rad]", input=true],
        uE_1(t),
        [description = "throttle 1 [rad]", input=true],
        uE_2(t),
        [description = "throttle 2 [rad]", input=true],
    )

    # Auxiliary Variables to define model.

    Auxiliary_vars =
        @variables Va(t) α(t) β(t) Q(t) CL_wb(t) ϵ(t) α_t(t) CL_t(t) CL(t) CD(t) CY(t) F1(t) F2(
            t,
        )

    @variables FA_s(t)[1:3] C_bs(t)[1:3, 1:3] FA_b(t)[1:3] eta(t)[1:3] dCMdx(t)[1:3, 1:3] dCMdu(t)[1:3,1:3] CMac_b(t)[1:3] MAac_b(t)[1:3] rcg_b(t)[1:3] rac_b(t)[1:3] MAcg_b(t)[1:3] FE1_b(t)[1:3] FE2_b(
        t,
    )[1:3] FE_b(t)[1:3] mew1(t)[1:3] mew2(t)[1:3] MEcg1_b(t)[1:3] MEcg2_b(t)[1:3] MEcg_b(t)[1:3] g_b(
        t,
    )[1:3] Fg_b(t)[1:3] F_b(t)[1:3] Mcg_b(t)[1:3] H_phi(t)[1:3, 1:3]

    # Scalarizing all the array variables. 

    FA_s    = collect(FA_s)
    C_bs    = collect(C_bs)
    FA_b    = collect(FA_b)
    eta     = collect(eta)
    dCMdx   = collect(dCMdx)
    dCMdu   = collect(dCMdu)
    CMac_b  = collect(CMac_b)
    MAac_b  = collect(MAac_b)
    rcg_b   = collect(rcg_b)
    rac_b   = collect(rac_b)
    MAcg_b  = collect(MAcg_b)
    FE1_b   = collect(FE1_b)
    FE2_b   = collect(FE2_b)
    FE_b    = collect(FE_b)
    mew1    = collect(mew1)
    mew2    = collect(mew2)
    MEcg1_b = collect(MEcg1_b)
    MEcg2_b = collect(MEcg2_b)
    MEcg_b  = collect(MEcg_b)
    g_b     = collect(g_b)
    Fg_b    = collect(Fg_b)
    F_b     = collect(F_b)
    Mcg_b   = collect(Mcg_b)
    H_phi   = collect(H_phi)

    array_vars = vcat(
        vec(FA_s),
        vec(C_bs),
        vec(FA_b),
        vec(eta),
        vec(dCMdx),
        vec(dCMdu),
        vec(CMac_b),
        vec(MAac_b),
        vec(rcg_b),
        vec(rac_b),
        vec(MAcg_b),
        vec(FE1_b),
        vec(FE2_b),
        vec(FE_b),
        vec(mew1),
        vec(mew2),
        vec(MEcg1_b),
        vec(MEcg2_b),
        vec(MEcg_b),
        vec(g_b),
        vec(Fg_b),
        vec(F_b),
        vec(Mcg_b),
        vec(H_phi),
    )

    eqns = [
        # Step 1. Intermediate variables 
        # Airspeed
        Va ~ sqrt(u^2 + v^2 + w^2)

        # α and β
        α ~ atan(w, u)
        β ~ asin(v / Va)

        # dynamic pressure
        Q ~ 0.5 * ρ * Va^2


        # Step 2. Aerodynamic Force Coefficients
        # CL - wing + body
        #CL_wb ~  n*(α - α_L0)

        CL_wb ~ ifelse(
            α <= α_switch,
            n * (α - α_L0),
            a3 * α^3 + a2 * α^2 + a1 * α + a0,
        )

        # CL thrust
        ϵ ~ depsda * (α - α_L0)
        α_t ~ α - ϵ + uT + 1.3 * q * lt / Va
        CL_t ~ 3.1 * (St / S) * α_t

        # Total CL
        CL ~ CL_wb + CL_t

        # Total CD
        CD ~ 0.13 + 0.07 * (n * α + 0.654)^2

        # Total CY
        CY ~ -1.6 * β + 0.24 * uR


        # Step 3. Dimensional Aerodynamic Forces
        # Forces in F_s
        FA_s .~ [
            -CD * Q * S
            CY * Q * S
            -CL * Q * S
        ]


        # rotate forces to body axis (F_b)  
        vec(C_bs .~ [
            cos(α) 0.0 -sin(α)
            0.0 1.0 0.0
            sin(α) 0.0 cos(α)
        ])
        FA_b .~ C_bs * FA_s

        # Step 4. Aerodynamic moment coefficients about AC
        # moments in F_b
        eta .~ [
            -1.4 * β
            -0.59 - (3.1 * (St * lt) / (S * c̄)) * (α - ϵ)
            (1 - α * (180 / (15 * π))) * β
        ]
        vec(
            dCMdx .~
                (c̄ / Va) * [
                    -11.0 0.0 5.0
                    0.0 (-4.03*(St*lt^2)/(S*c̄^2)) 0.0
                    1.7 0.0 -11.5
                ],
        )
        vec(dCMdu .~ [
            -0.6 0.0 0.22
            0.0 (-3.1*(St*lt)/(S*c̄)) 0.0
            0.0 0.0 -0.63
        ])

        # CM about AC in Fb
        CMac_b .~ eta + dCMdx * wbe_b + dCMdu * [
            uA
            uT
            uR
        ]

        # Step 5. Aerodynamic moment about AC 
        # normalize to aerodynamic moment
        MAac_b .~ CMac_b * Q * S * c̄

        # Step 6. Aerodynamic moment about CG
        rcg_b .~ [
            Xcg
            Ycg
            Zcg
        ]
        rac_b .~ [
            Xac
            Yac
            Zac
        ]
        MAcg_b .~ MAac_b + cross(FA_b, rcg_b - rac_b)

        # Step 7. Engine force and moment
        # thrust
        F1 ~ uE_1 * m * g
        F2 ~ uE_2 * m * g

        # thrust vectors (assuming aligned with x axis)
        FE1_b .~ [
            F1
            0
            0
        ]
        FE2_b .~ [
            F2
            0
            0
        ]
        FE_b .~ FE1_b + FE2_b

        # engine moments
        mew1 .~ [
            Xcg - Xapt1
            Yapt1 - Ycg
            Zcg - Zapt1
        ]
        mew2 .~ [
            Xcg - Xapt2
            Yapt2 - Ycg
            Zcg - Zapt2
        ]
        MEcg1_b .~ cross(mew1, FE1_b)
        MEcg2_b .~ cross(mew2, FE2_b)
        MEcg_b .~ MEcg1_b + MEcg2_b

        # Step 8. Gravity effects
        g_b .~ [
            -g * sin(θ)
            g * cos(θ) * sin(ϕ)
            g * cos(θ) * cos(ϕ)
        ]
        Fg_b .~ m * g_b

        # Step 9: State derivatives

        # form F_b and calculate u, v, w dot
        F_b .~ Fg_b + FE_b + FA_b
        D.(V_b) .~ (1 / m) * F_b - cross(wbe_b, V_b)

        # form Mcg_b and calc p, q r dot
        Mcg_b .~ MAcg_b + MEcg_b
        D.(wbe_b) .~ invIb * (Mcg_b - cross(wbe_b, Ib * wbe_b))

        #phi, theta, psi dot
        vec(
            H_phi .~ [
                1.0 sin(ϕ)*tan(θ) cos(ϕ)*tan(θ)
                0.0 cos(ϕ) -sin(ϕ)
                0.0 sin(ϕ)/cos(θ) cos(ϕ)/cos(θ)
            ],
        )
        D.(rot) .~ H_phi * wbe_b
    ]

    all_vars = vcat(V_b, wbe_b, rot, U, Auxiliary_vars, array_vars)
    @named rcam_model = ODESystem(eqns, t, all_vars, ps)

    inputs = [uA, uT, uR, uE_1, uE_2]
    outputs = []
    iosys, diff_idxs, alge_idxs, input_idxs = ModelingToolkit.io_preprocessing(rcam_model, inputs, outputs)

    x0 = Dict(  u => 84.9905,
                p => 0.0,
                r => 0.0,
                ϕ => 0.0,
                w => 1.27132,
                θ => 0.0149573,
                v => 0.0,
                ψ => 0.0,
                q => 0.0)

    u0 = Dict(  uR => 0.0,
                uE_2 => 0.0820834,
                uE_1 => 0.0820834,
                uT => -0.178008,
                uA => 0.0)

    (; rcam_model, inputs, iosys, x0, u0, p=ps, rcam_constants)
end

# TODO: add quadtank et. al

#=
The CSTR dynamics are derived from do-mpc with the following license (GNU Lesser General Public License v3.0)
https://github.com/do-mpc/do-mpc/blob/master/LICENSE.txt
=#

const p_cstr = (;
    K0_ab   = 1.287e12, # K0 [h^-1]
    K0_bc   = 1.287e12, # K0 [h^-1]
    K0_ad   = 9.043e9, # K0 [l/mol.h]
    R_gas   = 8.31446e-3, # Universal gas constant
    E_A_ab  = 9758.3, #* R_gas,# [kj/mol]
    E_A_bc  = 9758.3, #* R_gas,# [kj/mol]
    E_A_ad  = 8560.0, #* R_gas,# [kj/mol]
    Hᵣ_ab   = 4.2, # [kj/mol A]
    Hᵣ_bc   = -11.0, # [kj/mol B] Exothermic
    Hᵣ_ad   = -41.85, # [kj/mol A] Exothermic
    Rou     = 0.9342, # Density [kg/l]
    Cp      = 3.01, # Specific Heat capacity [kj/Kg.K]
    Cpₖ     = 2.0, # Coolant heat capacity [kj/kg.k]
    Aᵣ      = 0.215, # Area of reactor wall [m^2]
    Vᵣ      = 10.0, #0.01, # Volume of reactor [l]
    m_k     = 5.0, # Coolant mass[kg]
    T_in    = 130.0, # Temp of inflow [Celsius]
    K_w     = 4032.0, # [kj/h.m^2.K]
    C_A0    = (5.7+4.5)/2.0*1.0,  # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]
)

function cstr(x, u, p=p_cstr, _=0)
    Cₐ, Cᵦ, Tᵣ, Tₖ = x

    (; K0_ab,K0_bc,K0_ad,R_gas,E_A_ab,E_A_bc,E_A_ad,Hᵣ_ab,Hᵣ_bc,Hᵣ_ad,Rou,Cp,Cpₖ,Aᵣ,Vᵣ,m_k,T_in,K_w,C_A0) = p


    F, Q̇    = u
    K₁      = K0_ab * exp((-E_A_ab)/((Tᵣ+273.15)))
    K₂      = K0_bc * exp((-E_A_bc)/((Tᵣ+273.15)))
    K₃      = K0_ad * exp((-E_A_ad)/((Tᵣ+273.15)))
    TΔ      = Tᵣ-Tₖ
    SA[
        F*(C_A0 - Cₐ)-K₁*Cₐ - K₃*abs2(Cₐ),
        -F*Cᵦ + K₁*Cₐ - K₂*Cᵦ,
        ((K₁*Cₐ*Hᵣ_ab + K₂*Cᵦ*Hᵣ_bc + K₃*abs2(Cₐ)*Hᵣ_ad)/(-Rou*Cp)) + F*(T_in-Tᵣ) + (((K_w*Aᵣ)*(-TΔ))/(Rou*Cp*Vᵣ)),
        (Q̇ + K_w*Aᵣ*(TΔ))/(m_k*Cpₖ)
    ]
end

"""
    sys = cstr()

A model of the continuously stirred tank reactor (CSTR).
This model has 4 states and 2 inputs.

`sys` has the following fields:
- `dynamics`: the dynamics of the system in the form of a [`FunctionSystem`](@ref)
- `sys`: the system in the form of a [`ODESystem`](@ref)
- `Ts`: a suggested sample time for the system
- `x0`: a suggested initial state of the system
- `lb`: a vector of pairs with lower bounds for all variables
- `ub`: a vector of pairs with upper bounds for all variables
- `p`: A NamedTuple with the default parameters of the system

# Example:
```julia
sys = DyadControlSystems.ControlDemoSystems.cstr()
display(sys.sys)
```
"""
function cstr()
    @parameters t
    D = Differential(t)
    @variables begin
        Cₐ(t), [description = "Concentration of reactant A"],
        Cᵦ(t), [description = "Concentration of reactant B"],
        Tᵣ(t), [description = "Temperature in reactor"],
        Tₖ(t), [description = "Temperature in cooling jacket"],
        F(t), [description = "Feed", input=true],
        Q̇(t), [description = "Heat flow", input=true]
    end
    vars = [Cₐ, Cᵦ, Tᵣ, Tₖ, F, Q̇]
    eqs = cstr([Cₐ, Cᵦ, Tᵣ, Tₖ], [F, Q̇], p_cstr, t)
    sys = ODESystem(D.([Cₐ, Cᵦ, Tᵣ, Tₖ]) .~ eqs, t; name=:cstr)
    Ts  = 0.005 # sample time
    x0  = [0.8, 0.5, 134.14, 130] # Initial state
    x_names = [:Cₐ, :Cᵦ, :Tᵣ, :Tₖ]
    u_names = [:F, :Q̇]
    lb = vars .=> SA[0.1, 0.1, 50, 50, 5, -8500]
    ub = vars .=> SA[2, 2, 142, 140, 100, 0.0]
    measurement = (x,u,p,t) -> x # We can measure the full state
    dynamics = FunctionSystem(cstr, measurement; x=x_names, u=u_names, y=x_names, input_integrators=1:2)
    (; dynamics, sys, Ts, x0, lb, ub, p=p_cstr)
end



using ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Mechanical.Rotational
using ModelingToolkitStandardLibrary.Blocks

"""
    dcmotor()

Return a system with a DC motor where the speed is controlled by a PI controller.
Interesting signals include
- `speed_sensor.w`
- `pi_controller.ctr_output`
- `source.V` (the motor voltage input)

Analysis points `:r, :u` and `:y` are defined for the reference as well as the input and output of the system.
"""
function dcmotor(;
        ref       = Blocks.Step(height = 1, start_time = 0, name=:ref),
        load_step = Blocks.Step(height = -0.3, start_time = 3, name=:load_step),
        k = 1.1,
        Ti = 0.035,
        J = 0.02, #[description = "[kg.m²] inertia"]
        k_emf = 0.5, # description = "[N.m/A] motor constant"]
    )
    @parameters t

    R = 0.5 # description = "[Ohm] armature resistance"]
    L = 4.5e-3 # description = "[H] armature inductance"]
    f = 0.01 #[description = "[N.m.s/rad] friction factor"]
    @named ground = Ground()
    @named source = Voltage()
    @named pi_controller = Blocks.LimPI(; k, T = Ti, u_max = 10, Ta = 0.035)
    @named feedback = Blocks.Feedback()
    @named R1 = Resistor(R = R)
    @named L1 = Inductor(L = L)
    @named emf = EMF(k = k_emf)
    @named fixed = Fixed()
    @named load = Torque(use_support = false)
    @named inertia = Inertia(J = J, phi = 0.0, w = 0.0)
    @named friction = Damper(d = f)
    @named speed_sensor = SpeedSensor()
    @named angle_sensor = AngleSensor()

    connections = [connect(fixed.flange, emf.support, friction.flange_b)
                connect(emf.flange, friction.flange_a, inertia.flange_a)
                connect(inertia.flange_b, load.flange)
                connect(inertia.flange_b, speed_sensor.flange, angle_sensor.flange)
                connect(speed_sensor.w, :y, feedback.input2)
                connect(feedback.output, :e, pi_controller.err_input)
                connect(pi_controller.ctr_output, :u, source.V)
                connect(source.p, R1.p)
                connect(R1.n, L1.p)
                connect(L1.n, emf.p)
                connect(emf.n, source.n, ground.g)]

    systems = [
        ground,
        pi_controller,
        feedback,
        source,
        R1,
        L1,
        emf,
        fixed,
        load,
        inertia,
        friction,
        speed_sensor,
        angle_sensor,
    ]

    if ref !== nothing
        push!(connections, connect(ref.output, :r, feedback.input1))
        push!(systems, ref)
    end
    if load_step !== nothing
        push!(connections, connect(load_step.output, load.tau))
        push!(systems, load_step)
    end

    ODESystem(connections, t; systems, name=:dcmotor)
end

"""
Nonlinear quadtank model taken from https://help.juliahub.com/DyadControlSystems/stable/examples/quadtank.html
"""
const kc = 0.5
function quadtank(h, u, p = nothing, t = nothing)
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a1, a3, a2, a4 = 0.03, 0.03, 0.03, 0.03
    γ1, γ2 = 0.2, 0.2

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0

    xd = SA[
        -a1 / A1 * ssqrt(2g * h[1]) + a3 / A1 * ssqrt(2g * h[3]) + γ1 * k1 / A1 * u[1]
        -a2 / A2 * ssqrt(2g * h[2]) + a4 / A2 * ssqrt(2g * h[4]) + γ2 * k2 / A2 * u[2]
        -a3 / A3 * ssqrt(2g * h[3]) + (1 - γ2) * k2 / A3 * u[2]
        -a4 / A4 * ssqrt(2g * h[4]) + (1 - γ1) * k1 / A4 * u[1]
    ]
end

function quadtank()
    nu = 2 # number of control inputs
    nx = 4 # number of states
    ny = 2 # number of measured outputs
    Ts = 2 # sample time
    #w = exp10.(-3:0.1:log10(pi / Ts)) # A frequency-vector for plotting

    discrete_dynamics0 = rk4(quadtank, Ts, supersample = 2) # Discretize the nonlinear continuous-time dynamics
    state_names = :h^4
    measurement = (x, u, p, t) -> kc * x[1:2]
    discrete_dynamics = FunctionSystem(discrete_dynamics0, measurement, Ts, x = state_names, u = :u^2, y = :h^2)
    xr0 = [10, 10, 6, 6]; # desired reference state

    xr, ur = [9.820803494468082, 9.820803494468084, 6.285133601386427, 6.285133581845583], [0.26026743939475916, 0.2602687896028877]


    x0 = [2, 1, 8, 3] # Initial tank levels

    Cc, Dc = linearize(measurement, xr, ur, 0, 0)
    op = OperatingPoint(xr, ur, Cc * xr)

    disc(x) = c2d(ss(x), Ts) # Discretize the linear model using zero-order-hold

    Ac, Bc = linearize(quadtank, xr, ur, 0, 0)
    Cc, Dc = linearize(measurement, xr, ur, 0, 0)
    Gc = ss(Ac, Bc, Cc, Dc)

    G = disc(Gc)
    (; G, x0, xr0, xr, op, nu, nx, ny, Ts, discrete_dynamics)
end



# ==============================================================================
## Furuta pendulum
# ==============================================================================


"""
    furuta(x, u, p, t)

Dynamics of the Furuta pendulum

Let the length of the pendulum be `l`, the mass of the weight `M`, the mass of the pendulum `m`, its moment of inertia `J` and the moment of inertia for the arm `Jp`. The length of the arm is `r`. The angle of the pendulum, `θ`, is defined to be zero when in upright position and positive when the pendulum is moving clockwise. The angle of the arm, `φ` is positive when the arm is moving in counter clockwise direction. Further, the central vertical axis is connected to a DC motor which adds a torque proportional to the control signal `u`.

# State variables
```math
θ, θ̇, ϕ, ϕ̇ = x
```

# Parameters

`p = M, l, r, J, Jp, m, g` where
- `M` is the mass of the tip of the pendulum
- `l` is the length of the pendulum
- `r` is the length of the arm
- `J` is the moment of inertia of the pendulum
- `Jp` is the moment of inertia of the arm
- `m` is the mass of the pendulum
- `g` is the gravitational constant

Example:
```julia
using DyadControlSystems, OrdinaryDiffEq, Plots
import DyadControlSystems.ControlDemoSystems as demo

normalize_angles(x::Number) = mod(x + pi, 2pi) - pi
normalize_angles(x::AbstractVector) = SA[normalize_angles(x[1]), x[2], normalize_angles(x[3]), x[4]]

function controller(x, p)
    θ, θ̇, ϕ, ϕ̇ = x
    ks = 100
    th = 2
    if abs(θ) < 0.2 # Use stabilizing controller
        -L*normalize_angles(x)
    else
        E = demo.furuta_energy(x, p)
        [clamp(ks*E*sign(θ̇*cos(θ)), -th, th)]
    end
end

function furuta_cl(x, p, t)
    u = controller(x, p)
    demo.furuta(x, u, p, t)
end

l = 0.413
M = 0.01
Jp = 0.0009
r = 0.235
J = 0.05
m = 0.02
g = 9.82
p = (M, l, r, J, Jp, m, g)

Afu, Bfu = DyadControlSystems.linearize(demo.furuta, zeros(4), [0], p, 0)
Q1 = Diagonal([1, 1, 1e-6, 1])
Q2 = 100
L = lqr(Continuous, Afu, Bfu, Q1, Q2)

x0 = SA[0.99pi, 0.0, 0, 0.0]
tspan = (0.0, 10.0)
prob = ODEProblem(furuta_cl, x0, tspan, p)
sol = solve(prob, Tsit5())
f1 = plot(sol, layout=4, lab=["\$θ\$" "\$\\dot θ\$" "\$ϕ\$" "\$\\dot ϕ\$"], plot_title="Furuta pendulum swing-up")

time = range(tspan..., length=300)
u = map(time) do t
    controller(sol(t), p)
end
U = reduce(hcat, u)'
f2 = plot(time, U, lab="\$u\$")
plot(f1, f2)
```
"""
function furuta(x, u, p, t)
    θ, θ̇, ϕ, ϕ̇ = x
    M, l, r, J, Jp, m, g = p
    τ = only(u)
    α = Jp+M*l^2
    β = J+M*r^2+m*r^2
    γ = M*r*l
    ϵ = l*g*(M+m/2)
    sθ = sin(θ)
    cθ = cos(θ)
    C = 1/(α*β+α^2*(sθ)^2-γ^2*cθ^2)
    scθ = sθ*cθ
    θ̈ = C*((α*β+α^2*(sθ)^2)*ϕ̇^2*scθ-γ^2*θ̇^2*scθ+2*α*γ*θ̇*ϕ̇*sθ*cθ^2-γ*cθ*τ+(α*β+α^2*(sθ)^2)*ϵ/α*sθ)
    ϕ̈ = C*(-γ*α*ϕ̇^2*sθ*(cos(ϕ))^2-γ*ϵ*scθ+γ*α*θ̇^2*sθ-2*α^2*θ̇*ϕ̇*scθ+α*τ)
    SA[
        θ̇
        θ̈ - 0.1*θ̇
        ϕ̇
        ϕ̈ - 0.1*ϕ̇
    ]
end

"""
    E = furuta_energy(x, p)

Energy of simplified dynamic model, normalized with `Mgl`. Defined to be zero at upward equilibrium position (origin).
A reasonable swing-up controller is given by
```
u(x) = clamp(ks*E*sign(θ̇*cos(θ)), -th, th)
```
for some gain `ks` (like 100) and some threashold `th`. A suitable criterion for swithcing to a stabilizing controller is when `abs(θ) < 0.2`.
"""
function furuta_energy(x, p)
    θ, θ̇, ϕ, ϕ̇ = x
    M, l, r, J, Jp, m, g = p
    ω = sqrt(M*g*l/Jp)
    (θ̇/ω)^2/2 + cos(θ) - 1
end





end
