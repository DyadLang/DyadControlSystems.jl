using DyadControlSystems
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit

##
using ModelingToolkit
using ModelingToolkit: defaults, unknowns, value

@variables t x(t)=0 y(t)=0 u(t)=0 r(t)=0 [input=true]
@parameters kp=1
D = Differential(t)

eqs = [
    u ~ kp * (r - y)
    D(x) ~ -x + u
    y ~ x
]

@named sys = ODESystem(eqs, t)

# f, xout, p = ModelingToolkit.generate_control_function(sys, expression=Val{false})
f = DyadControlSystems.build_controlled_dynamics(sys, r)
x = zeros(f.nx)
u = [1]
p = [1]
xd = f(x,u,p,1)
varmap = Dict(
    sys.x => 1,
)

@test xd[] == 1
@test_broken xd == ModelingToolkit.varmap_to_vars(varmap, xout) # https://github.com/JuliaComputing/DyadControlSystems.jl/issues/57

##


function plant(; name)
    @variables  x(t)=1
    @variables u(t)=0 [input=true] y(t)=0 [output=true]
    D = Differential(t)
    eqs = [
        D(x) ~ -x + u
        y ~ x
    ]
    ODESystem(eqs, t; name=name)
end

function filt_(; name)
    @variables x(t)=0 y(t)=0 [output=true]
    @variables u(t)=0 [input=true] 
    D = Differential(t)
    eqs = [
        D(x) ~ -2*x + u
        y ~ x
    ]
    ODESystem(eqs, t, name=name)
end

function controller(kp; name)
    @variables y(t)=0 r(t)=0 [input=true] u(t)=0
    @parameters kp=kp
    eqs = [
        u ~ kp * (r - y)
    ]
    ODESystem(eqs, t; name=name)
end



@named f = filt_()
@named c = controller(1)
@named p = plant()

connections = [
    f.y ~ c.r # filtered reference to controller reference
    c.u ~ p.u # controller output to plant input
    p.y ~ c.y # feedback
]

@named cl = ODESystem(connections, t, systems=[f,c,p])

fsys = DyadControlSystems.build_controlled_dynamics(cl, f.u, p.y)
# f,xout,p = ModelingToolkit.generate_control_function(cl, expression=Val{false})
xout = fsys.x
x = zeros(2)
u = [1]
p_ = [1]

xd = fsys(x,u,p_,1)
varmap = Dict(
    f.x => 1,
    p.x => 0
)
@test xd == ModelingToolkit.varmap_to_vars(varmap, xout)

