### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 05423b92-31bb-11ed-0514-b3fc3d671cb4
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	using ModelingToolkit, OrdinaryDiffEq, Symbolics, IfElse
	using Symbolics:scalarize
	using LinearAlgebra:cross, inv
	using Plots
end

# ╔═╡ b9e5a4e4-8aad-4e46-bf96-63421182a6a1
md"# Introduction"

# ╔═╡ 2ccddff5-fbbf-4a5b-99d1-bdf32cff8cb0
md"This provides a Julia implementation of the nonlinear Research Civil Aircraft Model (RCAM) developed by [GARTEUR](http://garteur.org/wp-content/reports/FM/FM_AG-08_TP-088-3.pdf.). This work uses the derivations and errata presented by [Prof. Christopher Lum](https://www.youtube.com/watch?v=bFFAL9lI2IQ). This implementation uses a similar set-up to the [open-source Python implementation](https://github.com/flight-test-engineering/PSim-RCAM)"

# ╔═╡ f87fa4e7-7148-472d-930e-4bdb41032261
md"# The RCAM Aircraft model"

# ╔═╡ 2f0d9223-d05c-46cf-9ec8-75bb03c3b8b1
md" [TO ADD MORE] The RCAM model is a non-linear dynamic system with:
- 9 states:
- 5 control inputs" 

# ╔═╡ e485674c-1003-4007-9dd8-dcfd6d078e9a
md"## States:
- u  :  translational velocity in x-direction
- v :  translational velocity in y-direction 
- w :  translational velocity in z-direction
- p : Rotationaal velocity about x-axis 
- q : Rotational velocity about y-axis
- r : Rotational velocity about z-axis
- ϕ : Rotational angle about x-axis ?? [Use pitch, yaw??] 
- θ : Rotational angle about x-axis
- ψ :  Rotational angle about x-axis"

# ╔═╡ 2e0d22cb-836b-4be9-b799-1b0b4d492cb2
begin
    @parameters t
    D =  Differential(t)
    
    # States 
    V_b = @variables u(t) v(t) w(t) # Translational velocity variables in body frame
    wbe_b = @variables p(t) q(t) r(t) # Rotational velocity variables in body frame
    rot = @variables ϕ(t) θ(t) ψ(t) # Rotation angle variables 
	states = reduce(vcat, [V_b, wbe_b,rot])
end

# ╔═╡ 87848c3a-4ef7-4b3c-bef1-12d4a87527df


# ╔═╡ 28363749-b08d-4425-9a0f-10f19704ead1
md" ## Control Inputs:
- uA : Aileron angle
- uT : Tail angle
- uR : Rudder angle
- uE_1 : Engine 1 thrust angle
- uE_2 : Engine 2 thrust angle"

# ╔═╡ 8c4608fa-3f25-4102-8fcf-28d45f0ad858
begin
	# Controls
    U = @variables uA(t) uT(t) uR(t) uE_1(t) uE_2(t) # Control variables
	U = reduce(vcat, U)
end

# ╔═╡ b9578760-972b-4cf7-834a-049b39b738e3
md"## 1. Defining the fixed parameters"

# ╔═╡ a8c950ff-2733-485b-adf0-e4fad35f7019
rcam_constants = Dict(
    # Note: Constant values and comments taken from https://github.com/flight-test-engineering/PSim-RCAM
    :m => 120000, # kg - total mass
    :cbar => 6.6, # m - mean aerodynamic chord
    :lt => 24.8, # m - tail AC distance to CG
    :S => 260, # m2 - wing area
    :St => 64, # m2 - tail area
    
    # centre of gravity position
    :Xcg => 0.23 * 6.6, # m - x pos of CG in Fm
    :Ycg => 0.0, # m - y pos of CG in Fm
    :Zcg => 0.10 * 6.6, # m - z pos of CG in Fm ERRATA - table 2.4 has 0.0 and table 2.5 has 0.10 cbar
    
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
    :depsda => 0.25, # rad/rad - change in downwash wrt Alpha
    :Alpha_L0 => deg2rad(-11.5), # rad - zero lift AOA
    :n => 5.5, # adm - slope of linear ragion of lift slope
    :a3 => -768.5, # adm - coeff of Alpha^3
    :a2 => 609.2, # adm -  - coeff of Alpha^2
    :a1 => -155.2, # adm -  - coeff of Alpha^1
    :a0 => 15.212, # adm -  - coeff of Alpha^0 ERRATA RCAM has 15.2. Modification suggested by C Lum. 
    :Alpha_switch => deg2rad(14.5), # rad - kink point of lift slope
    :ρ => 1.225 # kg/m3 - air density
)

# ╔═╡ d679f975-0877-466c-b429-9d4c98c4737b
begin    
		    ps = @parameters(
		        ρ             = rcam_constants[:ρ], [description="kg/m3 - air density"],
		        m             = rcam_constants[:m], [description="kg - total mass"],
		        cbar          = rcam_constants[:cbar], [description="m - mean aerodynamic chord"],
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
		        depsda        = rcam_constants[:depsda], [description="rad/rad - change in downwash wrt Alpha"],
		        Alpha_L0      = rcam_constants[:Alpha_L0], [description="rad - zero lift AOA"],
		        n             = rcam_constants[:n], [description="adm - slope of linear ragion of lift slope"],
		        a3            = rcam_constants[:a3], [description="adm - coeff of Alpha^3"],
		        a2            = rcam_constants[:a2], [description="adm -  - coeff of Alpha^2"],
		        a1            = rcam_constants[:a1], [description="adm -  - coeff of Alpha^1"],
		        a0            = rcam_constants[:a0], [description="adm -  - coeff of Alpha^0"],
		        Alpha_switch  = rcam_constants[:Alpha_switch], [description="rad - kink point of lift slope"],
		    )
end

# ╔═╡ 4693fbf8-9f27-4c00-b013-c8d9766cf7a6
md"##  2. Developing the mechanistic RCAM model symbolically"

# ╔═╡ ca1d001c-bb86-449c-a671-3d145a0017c3
# function RCAM_model(rcam_constants; name)
begin    
		Auxiliary_vars = @variables Va(t) Alpha(t) beta(t) Q(t) CL_wb(t) epsilon(t) Alpha_t(t) CL_t(t) CL(t) CD(t) CY(t) F1(t) F2(t)
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
		
		        # Alpha and beta
		        Alpha ~ atan(w,u)
		        beta ~ asin(v/Va)
		
		        # dynamic pressure
		        Q ~ 0.5*ρ*Va^2
		
		        
		        # Step 2. Aerodynamic Force Coefficients
		        # CL - wing + body
		        CL_wb ~ ifelse(Alpha <= Alpha_switch, n*(Alpha - Alpha_L0), a3*Alpha^3 + a2*Alpha^2 + a1*Alpha + a0)
		    
		        # CL thrust
		        epsilon ~ depsda*(Alpha - Alpha_L0)
		        Alpha_t ~ Alpha - epsilon + uT + 1.3*q*lt/Va
		        CL_t ~ 3.1*(St/S) * Alpha_t
		        
		        # Total CL
		        CL ~ CL_wb + CL_t
		    
		        # Total CD
		        CD ~ 0.13 + 0.07 * (n*Alpha + 0.654)^2
		    
		        # Total CY
		        CY ~ -1.6*beta + 0.24*uR
		        
		        
		        # Step 3. Dimensional Aerodynamic Forces
		        # Forces in F_s
		        FA_s .~ [-CD * Q * S
		                 CY * Q * S
		                 -CL * Q * S] 
		
		        
		        # rotate forces to body axis (F_b)  
		        vec(C_bs .~ [cos(Alpha)      0.0      -sin(Alpha)
		                     0.0             1.0      0.0
		                     sin(Alpha)      0.0      cos(Alpha)])
		        
		    
		
		        
		        FA_b .~ C_bs*FA_s # Old code remark: 1: scalarizing an `Equation` scalarizes both sides - actually defined in Symbolics.jl. Only way I could get `eqns isa Vector{Equation}`. 2: Cross-check that it is normal matmult,
		        
		        # Step 4. Aerodynamic moment coefficients about AC
		        # moments in F_b
		        eta .~ [ -1.4 * beta 
		                 -0.59 - (3.1 * (St * lt) / (S * cbar)) * (Alpha - epsilon)
		                (1 - Alpha * (180 / (15 * π))) * beta
		        ]
		        
		        
		        vec(dCMdx .~ (cbar / Va)*        [-11.0              0.0                           5.0
		                                          0.0   (-4.03 * (St * lt^2) / (S * cbar^2))        0.0
		                                          1.7                 0.0                          -11.5])
		        
		        
		        vec(dCMdu .~ [-0.6                   0.0                 0.22
		                      0.0   (-3.1 * (St * lt) / (S * cbar))      0.0
		                      0.0                    0.0                -0.63])
		
		        # CM about AC in Fb
		        CMac_b .~ eta + dCMdx*wbe_b + dCMdu*[uA
		                                            uT
		                                            uR]
		                                            
		        # Step 5. Aerodynamic moment about AC 
		        # normalize to aerodynamic moment
		        MAac_b .~ CMac_b * Q * S * cbar
		
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
end

# ╔═╡ 707dc3b8-b256-4bcf-b641-9670db507e6b
begin
	global all_vars = vcat(V_b,wbe_b,rot,U, Auxiliary_vars)
    for arr in array_vars
        vars = vcat(all_vars, vec(arr))
    end
end

# ╔═╡ 21f5e6a8-f60e-4ca2-8c0c-7c47734843d6
rcam_model = ODESystem(eqns, t, all_vars, ps; name = :rcam_model)

# ╔═╡ 11ede6b2-7edc-4c9f-98d5-396760e3cf3d
md" ## 3. Simplifying the symbolic RCAM model"

# ╔═╡ 744f2b54-8a87-468e-855c-92dbd0f2b9f4
begin
	inputs = [uA, uT, uR, uE_1, uE_2]
	outputs = []
	sys, diff_idxs, alge_idxs, input_idxs = ModelingToolkit.io_preprocessing(rcam_model, inputs, outputs)
end

# ╔═╡ 6d6d1072-8e04-478a-9b81-71162a539897
md" ## 4. Numerical Simulation of RCAM model"

# ╔═╡ ac68785c-1c92-4621-9a9e-3f5506ef51e7
begin
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
end

# ╔═╡ e75cdc65-010a-4ebe-a24f-6d68734d346b
plot(sol, idxs = (u))

# ╔═╡ 042a6c18-5622-46f6-bc12-89865b1d5f63
plot(sol, idxs = (v))

# ╔═╡ 61af4767-ef71-4a80-9515-413eefa49ee8
plot(sol, idxs = (w))

# ╔═╡ 5d6c9e93-139b-42f1-8fd1-6520ff0d31a9
plot(sol, idxs = (p))

# ╔═╡ 8dcf94c3-b0bd-40c2-95b1-be259e690e2b
plot(sol, idxs = (q))

# ╔═╡ 54b6c8a6-25c9-44a0-9e0b-55b16fdaadbf
plot(sol, idxs = (r))

# ╔═╡ a3e6ae0a-ae3a-4a3c-adb4-b614e21f4bfd
plot(sol, idxs = (ϕ))

# ╔═╡ 3e1bab5e-fd5a-4ab9-aa6f-03f15f651670
plot(sol, idxs = (θ))

# ╔═╡ f2d0daba-a181-44e0-afa9-d17829905af6
plot(sol, idxs = (ψ))

# ╔═╡ e4326349-7c42-402c-ab13-eec0e1d2d1dc
md" # 5. Trimming RCAM model
##  - Straight-and-Level Flight
This involves finding the states and controls that give a certain desirable trajectory.
- Steady-state implies that `x_dot` = 0.
- Let is also be desired that the speed `V_A` is a constant at 85 m/s.
- Flight path angle gamma is 0 rad
- No side slip v = 0.
- Wings should be level. `\phi` = 0.
- Heading angle is pointing north. '\psi` = 0."

# ╔═╡ de556b54-0957-4bba-a415-c62792de813c


# ╔═╡ 1dfa0916-9213-4aa5-b062-fee01d354eb5


# ╔═╡ 09746714-3b0b-4540-82dd-614a3ab5eb48
md" # 6. Linearization of RCAM model [TBD]"

# ╔═╡ f6471d81-0ad8-4fd8-a274-8a9d93f6e947
md" # 7. Linear MPC of RCAM model [TBD]" 

# ╔═╡ c749db7b-59d1-4719-8504-6f8edf571866
md" # 8. Surrogatization of Linear MPC of RCAM model [TBD]" 

# ╔═╡ Cell order:
# ╠═05423b92-31bb-11ed-0514-b3fc3d671cb4
# ╟─b9e5a4e4-8aad-4e46-bf96-63421182a6a1
# ╠═2ccddff5-fbbf-4a5b-99d1-bdf32cff8cb0
# ╠═f87fa4e7-7148-472d-930e-4bdb41032261
# ╠═2f0d9223-d05c-46cf-9ec8-75bb03c3b8b1
# ╠═e485674c-1003-4007-9dd8-dcfd6d078e9a
# ╠═2e0d22cb-836b-4be9-b799-1b0b4d492cb2
# ╠═87848c3a-4ef7-4b3c-bef1-12d4a87527df
# ╠═28363749-b08d-4425-9a0f-10f19704ead1
# ╠═8c4608fa-3f25-4102-8fcf-28d45f0ad858
# ╟─b9578760-972b-4cf7-834a-049b39b738e3
# ╟─a8c950ff-2733-485b-adf0-e4fad35f7019
# ╟─d679f975-0877-466c-b429-9d4c98c4737b
# ╠═4693fbf8-9f27-4c00-b013-c8d9766cf7a6
# ╠═ca1d001c-bb86-449c-a671-3d145a0017c3
# ╠═707dc3b8-b256-4bcf-b641-9670db507e6b
# ╠═21f5e6a8-f60e-4ca2-8c0c-7c47734843d6
# ╠═11ede6b2-7edc-4c9f-98d5-396760e3cf3d
# ╠═744f2b54-8a87-468e-855c-92dbd0f2b9f4
# ╟─6d6d1072-8e04-478a-9b81-71162a539897
# ╠═ac68785c-1c92-4621-9a9e-3f5506ef51e7
# ╠═e75cdc65-010a-4ebe-a24f-6d68734d346b
# ╠═042a6c18-5622-46f6-bc12-89865b1d5f63
# ╠═61af4767-ef71-4a80-9515-413eefa49ee8
# ╠═5d6c9e93-139b-42f1-8fd1-6520ff0d31a9
# ╠═8dcf94c3-b0bd-40c2-95b1-be259e690e2b
# ╠═54b6c8a6-25c9-44a0-9e0b-55b16fdaadbf
# ╠═a3e6ae0a-ae3a-4a3c-adb4-b614e21f4bfd
# ╠═3e1bab5e-fd5a-4ab9-aa6f-03f15f651670
# ╠═f2d0daba-a181-44e0-afa9-d17829905af6
# ╠═e4326349-7c42-402c-ab13-eec0e1d2d1dc
# ╠═de556b54-0957-4bba-a415-c62792de813c
# ╠═1dfa0916-9213-4aa5-b062-fee01d354eb5
# ╠═09746714-3b0b-4540-82dd-614a3ab5eb48
# ╠═f6471d81-0ad8-4fd8-a274-8a9d93f6e947
# ╠═c749db7b-59d1-4719-8504-6f8edf571866
