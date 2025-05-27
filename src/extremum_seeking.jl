using ModelingToolkit
@parameters t
D = Differential(t)
using ModelingToolkitStandardLibrary.Blocks: RealInput, RealOutput

"""
    PIESC(; k, tau, a, w, wh = 10w, name)

PI-ESC: Proportional-integral extremum seeking controller. 

PI-ESC may provide faster convergence to the optimum compared to [`ESC`](@ref), but may be harder to tune.

Assume a system to be controlled on the form
```math
\\dot x = f(x) + g(x)u \\\\
y = h(x)
```
where ``h(x)`` is a cost function to be minimized, the value of which can be observed (the function may in practice be unknown). The PI-ESC finds the steady-state control signal ``u^*`` that minimizes ``y`` by applying a dither signal ``a \\sin(ωt)`` to estimate the derivative of the cost function w.r.t. the input.

Note: the tuning guidelines for [`ESC`](@ref) and [`PIESC`](@ref) are very different.

# Arguments:
- `k`: Proportional gain
- `tau`: Integral time constant
- `a`: Dither amplitude
- `w`: Dither frequency
- `wh`: High-pass filter frequency, typically 10x larger than `w`.

Ref: "Proportional-integral extremum-seeking control" M. Guay

# Extended help
- The region of attraction grows with increasing ration `a/k`.
- Convexity of the cost function ``y = h(x)`` is required, this is in contrast to standard ESC.
- ``g`` is full rank ``∀ x``.
- The `PIESC` can handle dither frequencies faster than the process dynamics, in contrast to the guidlines for the standard [`ESC`](@ref).
"""
function PIESC(; k, tau, a, w, wh=10*w, name)
    @parameters k=k tau=tau a=a w=w wh=wh
    @variables v(t)=0 uh(t)=0 u(t) y(t)
    @named input = RealInput()
    @named output = RealOutput()
    eqs = [
        input.u ~ y
        output.u ~ u
        D(v)  ~ -wh*v + y
        D(uh) ~ -1/tau * (-wh^2*v + wh*y)*sin(w*t)
        u     ~ -k/a   * (-wh^2*v + wh*y)*sin(w*t) + uh + a*sin(w*t)
    ]
    ODESystem(eqs, t; name, systems=[input, output])
end

"""
    ESC(; k, tau, a, b=a, w, wh = w/3, name)

ESC: Extremum seeking controller. 

Assume a system to be controlled on the form
```math
\\dot x = f(x) + g(x)u \\\\
y = h(x)
```
where ``h(x)`` is a cost function to be minimized, the value of which can be observed (the function may in practice be unknown). The ESC finds the steady-state control signal ``u^*`` that minimizes ``y`` by applying a dither signal ``b \\sin(ωt)`` to estimate the derivative of the cost function w.r.t. the input.

Note: the tuning guidelines for [`ESC`](@ref) and [`PIESC`](@ref) are very different.

# Arguments:
- `k`: Proportional gain / learning rate. If `k` is positive, the controller minimizes `h` and if `k` is negative `h` is maximized.
- `a`: Dither demodulation amplitude
- `b`: Dither modulation amplitude (defaults to the same as `a`)
- `w`: Dither frequency
- `wh`: High-pass filter frequency, defaults to `w/3`.

# Extended help
- A general guideline is that convergence properties are improved by keeping tuning parameters small, but performance is better for larger parameters.
- `w` should in general be chosen slower than the dominant dynamics of the controlled process. 
- `a` should be chosen large enough to provide a good gradient estimate and ability to escape potential local minima.
- `b` may be selected independently of `a`, some sources recommend choosing `b < a`.
- If several `ESC` controllers are used to estimate different parameters, use different dither frequencies for each.
"""
function ESC(; k, a, b=a, w, wh=w/3, name)
    @parameters k=k a=a b=b w=w wh=wh
    @variables v(t)=0 uh(t)=0 u(t) y(t)
    @named input = RealInput()
    @named output = RealOutput()
    eqs = [
        input.u ~ y
        output.u ~ u
        D(v)  ~ -wh*v + y
        D(uh) ~ -k*a * (-wh^2*v + wh*y)*sin(w*t)
        u     ~ uh + b*sin(w*t)
    ]
    ODESystem(eqs, t; name, systems=[input, output])
end
