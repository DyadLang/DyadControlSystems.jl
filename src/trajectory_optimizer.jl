# Multi segement trajectory generator
# Ref: Richter, Charles, Adam Bry, and Nicholas Roy. "Polynomial trajectory planning for aggressive quadrotor flight in dense indoor environments."
using SparseArrays, OSQP
import Polynomials
"""
    px,py,pz,pyaw = optimal_trajectory_gen(T,x,xwp,vel)

 Trajectory generating function. Default order 10 of generated polynomial
## Output: 
-`prx`   : Coefficients of polynomial for x-direction of optimized trajectory 
-`pry`   : Coefficients of polynomial for y-direction of optimized trajectory 
-`prz`   : Coefficients of polynomial for z-direction of optimized trajectory 
-`pryaw` : Coefficients of polynomial for body axis orientation (yaw) of optimized trajectory 
 ## Input: 
-`T`     : Time segments between waypoints for optimization of trajectory
-`x`     : Initial coordinates for generated polynomial [x,y,z,yaw]
-`xwp`   : Waypoint array for generated polynomial [x,y,z,yaw]
-`vel`   : Initial velocity for generated polynomial
"""   

function optimal_trajectory_gen(Ta,x,xwp,vel;Order = 10,cost_r = 5, cost_yaw = 3,eps_abs = 1e-6)
    mseg = length(Ta)
    if cost_r < 0 || cost_r > 10
        throw(ArgumentError("Penalty on the position derivative must be between 1 and 10"))
    end
    
    if cost_yaw < 0 || cost_yaw > 10
        throw(ArgumentError("Penalty on the position derivative must be between 1 and 10"))
    end

    cr_r = zeros(11)
    cr_r[cost_r] = 1.0 # penalty on the rth derivative for position optimization

    cr_yaw = zeros(11)
    cr_yaw[cost_yaw] = 1.0 # penalty on the rth derivative for yaw optimization

    # Waypoint x
    posx = [
        x[1]     # Initial position
        xwp[:,1]
           ]
    # Waypoint y
    posy = [      
        x[2]     
        xwp[:,2] 
           ]
    # Waypoint z 
    posz = [   
        x[3]       
        xwp[:,3]      
           ]
    # Waypoint yaw
    posyaw = [
        x[4]
        xwp[:,4]
           ]

    # initial derivatives
    der_init = [    #initial derivatives
        vel         #initial vel
        0           #initial acc
        0           #initial jerk
        0           #initial snap
               ]
    # terminal derivatives   
    der_term = [    #terminal derivatives
        0           #terminal vel
        0           #terminal acc
        0           #terminal jerk
        0           #terminal snap
               ]

    # Hessian Computation
    Qm = zeros(Order+1,Order+1,mseg)
    Qr = zeros(Order+1,Order+1,Order+1)
    Qyaw = zeros(Order+1,Order+1,mseg)
    
    # waypoint position constraints
    Apos = zeros(2,Order+1,mseg)
    
    # terminal derivative constraints
    Ader_ini = zeros(1,mseg*(Order+1),6);
    Ader_ter = zeros(1,mseg*(Order+1),6);

    # derivative continuity constraints
    Ader = zeros(2,Order+1,mseg,6)
    Ader_end = zeros(mseg,mseg*(Order+1),6)
    Ader_beg = zeros(mseg,mseg*(Order+1),6)
    Aderj = zeros(mseg-1,mseg*(Order+1),6)
    bvelj = vel.*ones(mseg-1,1);    # vel  continuity
    baccj = zeros(mseg-1,1);        #acc  continuity
    bjrkj = zeros(mseg-1,1);        #jerk continuity
    bsnpj = zeros(mseg-1,1);        #snap continuity
    bcrkj = zeros(mseg-1,1);        #crackle continuity
    bpopj = zeros(mseg-1,1);        #pop continuity
    
    
     for ms = 1:mseg
        T = Ta[ms]
        for r = 0 : Order
            for i = 0 : Order
                for l = 0 : Order
                    if (i >= r) && (l >= r)
                        Mul = 1;
                        for m = 0 : r-1
                            Mul = Mul*(i-m)*(l-m)
                        end
                        Qr[i+1,l+1,r+1] = Mul*T^float(i+l-2*r+1)/(i+l-2*r+1)
                    end
                end
            end
        end
        for i = 0 : Order
            Qm[:,:,ms] = Qm[:,:,ms] + cr_r[i+1]*Qr[:,:,i+1]
            Qyaw[:,:,ms] = Qyaw[:,:,ms] + cr_yaw[i+1]*Qr[:,:,i+1]
        end 
    end
        Qj = Qm[:,:,1]
        Qjy= Qyaw[:,:,1]
        for msi = 2:mseg
            Qj = cat(Qj,Qm[:,:,msi], dims=(1,2))
            Qjy = cat(Qjy,Qyaw[:,:,msi], dims=(1,2))
        end
    for ms = 1:mseg
        T = Ta[ms]
        # Constraint computation
        A0m = zeros(7,Order+1)
        ATm = zeros(7,Order+1)
        for r = 0 : 6
            for n = 0 : Order
                if n >= r
                    Mul = 1
                    for m = 0 : r-1
                        Mul = Mul*(n-m)
                    end
                    ATm[r+1,n+1] = Mul*T^(n-r);
                    if n == r
                        A0m[r+1,n+1] = Mul;
                    end
                end
            end
            if r == 0
                Apos[1,:,ms] = A0m[r+1,:]
                Apos[2,:,ms] = ATm[r+1,:]
            else
                Ader[1,:,ms,r] = A0m[r+1,:]
                Ader[2,:,ms,r] = ATm[r+1,:]
            end
        end
    end

    Aposj = [Apos[1,:,1]';Apos[2,:,1]']
    T = eltype(Apos)
    bposjx= T[]
    bposjy= T[]
    bposjz= T[]
    bposjyaw= T[]
    for ms = 1:mseg
        if ms > 1
            Aposj = cat(Aposj,[Apos[1,:,ms]';Apos[2,:,ms]'], dims=(1,2))
        end
        bposjx = [bposjx; posx[ms]; posx[ms+1]]
        bposjy = [bposjy; posy[ms]; posy[ms+1]]
        bposjz = [bposjz; posz[ms]; posz[ms+1]]
        bposjyaw = [bposjyaw; posyaw[ms]; posyaw[ms+1]]
        for i = 1 : 6
            pos=(ms-1)*(Order+1)+1
            Ader_beg[ms,pos:pos+Order,i] = Ader[1,:,ms,i]
            Ader_end[ms,pos:pos+Order,i] = Ader[2,:,ms,i]
        end
    end

    for i = 1 : 6
        Aderj[:,:,i] = Ader_end[1:mseg-1,:,i]-Ader_beg[2:mseg,:,i]
        Ader_ini[1,:,i] = [Ader[1,:,1,i]' zeros(1,(Order+1)*(mseg-1))]
        Ader_ter[1,:,i] = [zeros(1,(Order+1)*(mseg-1)) Ader[2,:,mseg,i]'];
    end
    Abnd = T[]
    bbnd = T[]
    for i = 1 : 4
        Abnd= cat(Abnd,Ader_ini[1,:,i]',Ader_ter[1,:,i]',dims = 1)
        bbnd = [bbnd;
                der_init[i];
                der_term[i];
               ]
    end
    Abndyaw = zeros(4,length(Ader_ini[1,:,1]))
    bbndyaw = zeros(4,1)
    Abndyaw[1,:] = Ader_ini[1,:,1] 
    Abndyaw[2,:] = Ader_ter[1,:,1]
    Abndyaw[3,:] = Ader_ini[1,:,2] 
    Abndyaw[4,:] = Ader_ter[1,:,2]

    bbndyaw[1,1] = der_init[1]
    bbndyaw[2,1] = der_term[1]
    bbndyaw[3,1] = der_init[2]
    bbndyaw[4,1] = der_term[2]

    Avelj = Aderj[:,:,1]
    Aaccj = Aderj[:,:,2]
    Ajrkj = Aderj[:,:,3]
    Asnpj = Aderj[:,:,4]
    Acrkj = Aderj[:,:,5]
    Apopj = Aderj[:,:,6]

    Atj =  [Aposj;  Abnd; Avelj; Aaccj; Ajrkj; Asnpj; Acrkj; Apopj]
    btjx = [bposjx; bbnd; bvelj; baccj; bjrkj; bsnpj; bcrkj; bpopj]
    btjy = [bposjy; bbnd; bvelj; baccj; bjrkj; bsnpj; bcrkj; bpopj]
    btjz = [bposjz; bbnd; bvelj; baccj; bjrkj; bsnpj; bcrkj; bpopj]

    Atjyaw = [Aposj; Abndyaw; Avelj; Aaccj; Ajrkj; Asnpj];
    btjyaw = [bposjyaw; bbndyaw; bvelj; baccj; bjrkj; bsnpj];


    velj = zeros(mseg,(Order+1)*mseg)
    accj = zeros(mseg,(Order+1)*mseg)

    for ms = 1 : mseg
        velj[ms, (Order+1)*(ms-1)+1 : (Order+1)*ms] =  Ader[1,:,ms,1]
        accj[ms, (Order+1)*(ms-1)+1 : (Order+1)*ms] =  Ader[1,:,ms,2]
    end
    Qtot = cat(Qj,Qj,Qj, dims=(1,2))
    Atot = cat(Atj,Atj,Atj, dims=(1,2))
    ptot = zeros(33,1)
    btot = vec([btjx; btjy; btjz])

    Qt_r   = Qtot[:,:,1]
    P_r    = sparse(Qt_r)
    A      = sparse(Atot)
    l      = vec(btot)
    u      = vec(btot)

    Qt_yaw = Qjy[:,:,1]
    P_yaw  = sparse(Qt_yaw)    
    A_yaw  = sparse(Atjyaw)    
    ly     = convert(Vector{Float64},vec(btjyaw))
    uy     = convert(Vector{Float64},vec(btjyaw))

    # Create OSQP object
    prob_r   = OSQP.Model()
    prob_yaw = OSQP.Model()

    # Setup workspace and change alpha parameter
    OSQP.setup!(prob_r; P=P_r, A=A, l=l, u=u,eps_abs = eps_abs)
    OSQP.setup!(prob_yaw; P=P_yaw, A=A_yaw, l=ly, u=uy)

    # Solve problem
    results_r   = OSQP.solve!(prob_r)
    results_yaw = OSQP.solve!(prob_yaw)


    ptot = results_r.x
    pyaw = results_yaw.x

    px = ptot[1 : mseg*(Order+1)]
    py = ptot[mseg*(Order+1)+1 : 2*mseg*(Order+1) ]
    pz = ptot[2*mseg*(Order+1)+1 : end]
    
    prx = reshape(px,Order+1,mseg)
    pry = reshape(py,Order+1,mseg)
    prz = reshape(pz,Order+1,mseg)
    pryaw = reshape(pyaw,Order+1,mseg)
    
    return prx, pry, prz, pryaw
end

"""

    time_polynomial_trajectory(t, prx, pry, prz, pyaw, Ta)
    
Function to synthesize polynomial trajectories given optimal polynomial coefficients
# Arguments:
-`t`: Current time
-`prx`: Optimal polynomial coefficients for x direction motion
-`pry`: Optimal polynomial coefficients for y direction motion
-`prz`: Optimal polynomial coefficients for z direction motion
-`pryaw`: Optimal polynomial coefficients for yaw direction
-`Ta`: Input array of time values to take between two waypoints
"""
function time_polynomial_trajectory(t, prx, pry, prz, pyaw, Ta)
    if length(Ta) == 1
        Ta = [Ta]
    end
    timestamps = zeros(length(Ta) + 1)
    for i = 1:length(Ta)
        timestamps[i+1] = sum(Ta[1:i])
    end
    for i = 1:(length(timestamps)-1)
        if timestamps[i] <= t <= timestamps[i+1]
            xp = Polynomials.Polynomial(prx[:, i])
            yp = Polynomials.Polynomial(pry[:, i])
            zp = Polynomials.Polynomial(prz[:, i])
            xd = [xp(t - timestamps[i]); yp(t - timestamps[i]); zp(t - timestamps[i])]

            xpoly = [xp; yp; zp]
            xder1 = Polynomials.derivative.(xpoly)
            xder2 = Polynomials.derivative.(xder1)
            xder3 = Polynomials.derivative.(xder2)
            xder4 = Polynomials.derivative.(xder3)
            xdd = [
                xder1[1](t - timestamps[i])
                xder1[2](t - timestamps[i])
                xder1[3](t - timestamps[i])
            ]
            xddd = [
                xder2[1](t - timestamps[i])
                xder2[2](t - timestamps[i])
                xder2[3](t - timestamps[i])
            ]
            xd3d = [
                xder3[1](t - timestamps[i])
                xder3[2](t - timestamps[i])
                xder3[3](t - timestamps[i])
            ]
            xd4d = [
                xder4[1](t - timestamps[i])
                xder4[2](t - timestamps[i])
                xder4[3](t - timestamps[i])
            ]

            yawp = Polynomials.Polynomial(pyaw[:, i])
            b1d = [cos(yawp(t - timestamps[i])); sin(yawp(t - timestamps[i])); 0]
            dyawp = Polynomials.derivative(yawp)
            ddyawp = Polynomials.derivative(dyawp)
            b1d_1dot = [
                -sin(yawp(t - timestamps[i])) * dyawp(t - timestamps[i])
                cos(yawp(t - timestamps[i])) * dyawp(t - timestamps[i])
                0
            ]
            b1d_2dot = [
                -(cos(yawp(t - timestamps[i])) * (dyawp(t - timestamps[i])^2)) -
                (sin(yawp(t - timestamps[i])) * ddyawp(t - timestamps[i]))
                -(sin(yawp(t - timestamps[i])) * (dyawp(t - timestamps[i])^2)) +
                (cos(yawp(t - timestamps[i])) * ddyawp(t - timestamps[i]))
                0
            ]
            xdes = [xd; xdd; xddd; xd3d; xd4d]
            b1des = [b1d; b1d_1dot; b1d_2dot]
            return xdes, b1des
        end
    end
end
