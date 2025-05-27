import ModelingToolkit: get_iv, isoperator, time_varying_as_func, value

using ModelingToolkit: ODESystem, observed

function build_controlled_dynamics(sys::FunctionSystem, u; kwargs...)
    u == sys.u || throw(ArgumentError("The inputs of a FunctionSystem can not be altered."))
    sys
end


"""
    f, xout, pout = build_controlled_dynamics(sys, u; kwargs...)
    f, obsf, xout, pout = build_controlled_dynamics(sys, u, y; z=nothing, w=nothing, kwargs...)

Build a function on the form (x,u,p,t) -> ẋ where 
- `x` are the differential states
- `u` are control inputs
- `p` are parameters
- `t` is time
- `kwargs` are sent to [`ModelingToolkit.build_function`](@ref)

- `f` is a tuple of functions, one out of palce and one in place `(x,u,p,t) -> ẋ` and `(ẋ,x,u,p,t) -> nothing`
- `xout` contains the order of the states included in the dynamics
- `pout` contains the order of the parameters included in the dynamics

If in addition to `u`, outputs `y` are also specified, an additional observed function tuple is returned. 

# Example
This example forms a feedback system and builds a function of the dynamics from the reference `r` to the output `y`.
```
using ModelingToolkit, Test

@variables t x(t)=0 y(t)=0 u(t)=0 r(t)=0
@parameters kp=1
D = Differential(t)

eqs = [
    u ~ kp * (r - y) # P controller
    D(x) ~ -x + u    # Plant dynamics
    y ~ x            # Output equation
]

@named sys = ODESystem(eqs, t)

funsys = DyadControlSystems.build_controlled_dynamics(sys, r, y; checkbounds=true)
x  = zeros(funsys.nx) # sys.x
u  = [1] # r
p  = [1] # kp
xd = funsys(x,u,p,1)
varmap = Dict(
    funsys.x[] => 1,
    kp => 1,
)

@test xd == ModelingToolkit.varmap_to_vars(varmap, funsys.x)
```
"""
function build_controlled_dynamics(sys, u, y=nothing; w=nothing, z=nothing, input_integrators=0:-1, kwargs...)

    # sys = structural_simplify(sys)

    u isa Vector || (u = [u])
    t = ModelingToolkit.get_iv(sys)
    if w === nothing
        n_inputs = length(u)
        inputs = u
    else
        n_inputs = length(u) + length(w)
        inputs = [w; u]
    end

    f_, x, p, ssys = try
        ModelingToolkit.generate_control_function(sys, inputs; force_SA = true, split=false)
    catch e
        if e isa ErrorException && contains(e.msg, "Some specified")
            # wrapping system
            sys = ODESystem(Equation[], t, systems=[sys], name=:wrapper_sys)
            ModelingToolkit.generate_control_function(sys, inputs; force_SA = true)
            # ModelingToolkit.generate_control_function(sys, ModelingToolkit.renamespace.(:outer_sys, inputs))
        else
            rethrow()
        end
    end
    f = f_[1] # oop
    is_alg_eq = eq -> isequal(eq.lhs, 0)
    na = count(is_alg_eq, equations(ssys))
    if na > 0
        nx = length(unknowns(ssys))
        ndiff = nx-na
        M = cat(I(ndiff), zeros(na, na), dims = (1, 2))
        isinplace = false
        f = ODEFunction{isinplace, SciMLBase.FullSpecialize}(f, mass_matrix = M)
    end

    n_outputs = (y === nothing ? 0 : length(y)) + (z === nothing ? 0 : length(z))
    if n_outputs > 0
        y === nothing && throw(ArgumentError("Cannot have performance outputs without measured outputs."))
        outs = vcat(y) # this makes sure outs is a vector
        if z === nothing
            z = 1:length(x)
            # TODO: handle z as indices vs. symbols in these two branches
        else
            outs = [z; outs]
        end
        eqs = [eq for eq in equations(ssys)]
        obs = map(outs) do y
            for eq in eqs
                if ModelingToolkit.isoperator(eq.lhs, Symbolics.Operator) && isequal(y, arguments(eq.lhs)[1]) || isequal(y, eq.lhs)
                    return y
                end
            end
            for eq in observed(ssys)
                if isequal(y, eq.lhs)
                    return eq.rhs
                end
            end
        end
        any(isnothing, obs) && error("The following outputs were not found: $(y[isnothing.(obs)])")
        obsf = build_function(obs, x, u, p, t; expression=Val{false}, force_SA=true, kwargs...)
        return FunctionSystem(f, obsf[1], Continuous(), x, u, y, w, z, p, input_integrators, (; simplified_system = ssys))
    else
        return FunctionSystem(f, (x,u,p,t) -> x, Continuous(), x, u, x, w, z, p, input_integrators, (; simplified_system = ssys))
    end
end


"""
    FunctionSystem(sys::ODESystem, u, y; z=nothing, w=nothing, kwargs...)

Generate code for the dynamics and output of `sys` by calling [`build_controlled_dynamics`](@ref). See [`build_controlled_dynamics`](@ref) for more details.
"""
function FunctionSystem(sys::ODESystem, args...; kwargs...)
    build_controlled_dynamics(sys, args...; kwargs...)
end