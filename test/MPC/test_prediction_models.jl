using DyadControlSystems, LinearAlgebra, DyadControlSystems.MPC
nu = 2 # number of controls
nx = 4 # number of states
ny = 2 # number of outputs
Ts = 0.5 # sample time
N = 50

x0 = [2, 1, 8, 3]       # Initial state
xr = [10, 10, 4.9, 4.9] # reference state
ur = [0.26, 0.26]

kc = 0.5
k1, k2, g = 1.6, 1.6, 9.81
A1 = A3 = A2 = A4 = 4.9
a1, a3, a2, a4= 0.03, 0.03, 0.03, 0.03
h01, h02, h03, h04 = xr
T1, T2 = (A1/a1)sqrt(2*h01/g), (A2/a2)sqrt(2*h02/g)
T3, T4 = (A3/a3)sqrt(2*h03/g), (A4/a4)sqrt(2*h04/g)
c1, c2 = (T1*k1*kc/A1), (T2*k2*kc/A2)
γ1, γ2 = 0.3, 0.3

# Define the process dynamics
Ac = [-1/T1     0 A3/(A1*T3)          0
0     -1/T2          0 A4/(A2*T4)
0         0      -1/T3          0
0         0          0      -1/T4]
Bc = [γ1*k1/A1     0
0                γ2*k2/A2
0                (1-γ2)k2/A3
(1-γ1)k1/A4 0              ]

Cc = kc*[I(2) 0*I(2)] # Measure the first two tank levels
# Cc = kc*I(nx)
Dc = 0
Gc = ss(Ac,Bc,Cc,Dc)

op = OperatingPoint(xr, ur, Cc*xr)

disc = (x) -> c2d(ss(x), Ts)
G = disc(Gc)
Ad,Bd,Cd,Dd = ssdata(G)

# Control limits
umin = 0 * ones(nu)
umax = 1 * ones(nu)

# State limits (state constraints are soft by default)
xmin = 0 * ones(nx)
xmax = Float64[12, 12, 8, 8]

@testset "MPCContstraints" begin
    @info "Testing MPCContstraints"

    v = 1:nx

    constraints = MPCConstraints(; umin, umax, xmin, xmax)
    Cv, Dv, vmin, vmax, soft_indices = MPC.setup_output_constraints(nx, nu, constraints, op, v)

    @test Cv == [I(nx); zeros(nu,nx)]
    @test Dv == [zeros(nx,nu); I(nu)]
    @test soft_indices == 1:nx

    @test vmin == [constraints.xmin; constraints.umin] .- [op.x; op.u]
    @test vmax == [constraints.xmax; constraints.umax] .- [op.x; op.u]


    # One u constraint is inactivated, check that matrices are correct
    constraints = MPCConstraints(; umin = [0, -Inf], umax = [1, Inf], xmin, xmax)
    Cv, Dv, vmin, vmax, soft_indices = MPC.setup_output_constraints(nx, nu, constraints, op, v)

    @test Cv == [I(nx); zeros(nu-1,nx)]
    @test Dv == [zeros(nx,nu); [1 0]]

    # One x constraint is inactivated, check that matrices are correct
    constraints = MPCConstraints(; umin, umax, xmin=[0,0,0], xmax=[12,12,8])
    Cv, Dv, vmin, vmax, soft_indices = MPC.setup_output_constraints(nx, nu, constraints, op, 1:3)

    @test Cv == [I(nx-1) zeros(nx-1); zeros(nu,nx)]
    @test Dv == [zeros(nx-1,nu); I(nu)]

end




@testset "LinearMPCModel" begin
    @info "Testing LinearMPCModel"
    
    
    constraints = MPCConstraints(; umin, umax, xmin, xmax)
    R1 = 1e-5*I(nx)
    R2 = I(ny)
    kf = KalmanFilter(Ad, Bd, Cd, Dd, R1, R2)


    pm = LinearMPCModel(G, kf; constraints, op, strictly_proper=false, x0)
    @test pm.A == pm.sys.A
    @test pm.nx == nx
    @test pm.nu == nu
    @test pm.ny == ny
    @test pm.nz == nx
    @test pm.Ts == Ts
    @test pm.nxw2 == 0



    z = Cd
    pm = LinearMPCModel(G, kf; constraints, op, strictly_proper=false, z, x0)
    @test pm.A == pm.sys.A
    @test pm.nx == nx
    @test pm.nu == nu
    @test pm.ny == ny
    @test pm.nz == ny
    @test pm.Ts == Ts
    @test pm.nxw2 == 0
    @test pm.Cz == Cd

    z = [1]
    pm = LinearMPCModel(G, kf; constraints, op, strictly_proper=false, z, x0)
    @test pm.A == pm.sys.A
    @test pm.nx == nx
    @test pm.nu == nu
    @test pm.ny == ny
    @test pm.nz == 1
    @test pm.Ts == Ts
    @test pm.nxw2 == 0
    @test pm.Cz == [1 0 0 0]

end