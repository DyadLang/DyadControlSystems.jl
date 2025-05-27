# Economic MPC for residential HVAC

This example will demonstrate the use of Model-Predictive Control (MPC) with an economic objective, i.e., we are solving a control problem where the cost function is economically motivated. The example we will consider is the control of indoor temperature in a residential building in southern Sweden, corresponding roughly to a single-family house. 

The house in our example is equipped with a geothermal heat pump with a coefficient of performance (COP) of 4.9, i.e., for every kWh of electricity that is consumed by the heat pump, it delivers 4.9kWh of heat energy. The compressor in the heat pump can be run on 4 different power levels, each higher level consumes 1kW more than the previous, for a max power consumption of 4kW and thus a maximum heating capacity of 4.9*4 = 19.6kW. 

The MPC controller will adjust the power level hourly, and it will make use of the following inputs
- The current indoor temperature.
- The outdoor temperature, available as a 24h forecast.
- The spot price of electricity, available 24h into the future, obtained from the Nord Pool spot market.

In practice, weather forecasts are approximative, and the spot price for electricity is not always known 24h into the future, but for the purposes of this demonstration, these simplifying assumptions are made.[^1]

[^1]: Each day at 13:00, the spot price per hour for the following day is determined on the Nord Pool electricity market, the electricity price is thus available between 11 and 35hrs in advance at all times.

We will make use of a very simple dynamical model that relates the indoor temperature to the power level of the heat pump, and the outdoor temperature. We use a week of temperature data obtained from [SMHI](https://www.smhi.se/) and a week of electricity prices from the Nord Pool spot market for our simulation. 

```@example HVAC
using DyadControlSystems, Plots
using DyadControlSystems.MPC
using ModelingToolkit
using ControlSystemsMTK
using StaticArrays
using LinearAlgebra, Dates, Statistics
gr(fmt=:png) # hide

Ts  = 1   # Sample time hrs
COP = 4.9 # COP of the heat pump
heat_pump_power = 1  # kW per stage

electricity_price_kWh = [ # One week of hourly spot prices (SEK/100) at the Nord Pool energy market in area SE4
    66.31, 53.18, 49.77, 49.75, 55.27, 63.03, 95.82, 111.66, 113.93, 111.02, 109.19, 107.38, 106.05, 102.12, 101.26, 100.94, 106.01, 107.78, 106.79, 98.46, 58.21, 37.47, 32.67, 30.94, 27.8, 26.27, 25.72, 25.47, 25.66, 26.16, 26.91, 30.86, 36.25, 37.47, 36.41, 36.29, 34.13, 33.47, 32.49, 32.31, 33.92, 39.72, 67.16, 39.71, 34.61, 33.02, 32.45, 32.35, 35.8, 37.13, 38.85, 39.3, 39.84, 44.36, 55.24, 68.03, 110.68, 102.85, 107.8, 102.92, 93.51, 60.91, 60.46, 94.18, 109.55, 140.4, 166.13, 169.59, 126.65, 121.84, 106.61, 70.83, 73.01, 82.79, 120.53, 126.65, 129.78, 151.13, 181.87, 206.71, 201.97, 177.29, 164.2, 146.86, 149.8, 146.48, 145.59, 146.37, 169.94, 189.03, 205.37, 214.17, 188.61, 173.26, 165.87, 129.65, 127.18, 110.64, 33.17, 25.95, 88.79, 152.61, 175.33, 196.8, 200.76, 163.51, 140.7, 134.02, 125.95, 128.63, 129.09, 143.74, 166.64, 186.83, 209.22, 209.12, 188.73, 170.8, 127.23, 66.42, 66.94, 49.26, 67.04, 88.75, 133.68, 152.9, 181.15, 206.18, 206.18, 169.46, 140.41, 131.3, 128.08, 126.48, 127.61, 132.84, 146.53, 180.37, 203.59, 197.39, 176.04, 156.69, 153.66, 123.99, 117.55, 114.74, 114.73, 117.56, 140.68, 146.7, 178.54, 200.72, 196.93, 153.05, 137.88, 129.49, 124.86, 122.97, 127.35, 139.18, 165.18, 180.04, 200.09, 145.23, 117.59, 114.65, 25.98, 29.96, 
]

external_temperature = [ # One week of hourly temperatures in Hörby, Sweden https://www.smhi.se/data/meteorologi/ladda-ner-meteorologiska-observationer/#param=airtemperatureInstant,stations=core,stationid=53530
2.7, 2.2, 2, 1.5, 1, 1.4, 1.9, 1.6, 1.5, 1.3, 1.5, 1, 1.9, 3.8, 4.9, 2, 2.6, 2, 1.8, 1.2, 0.8, 0.9, 0.6, 0.6, 0.2, 0.1, 0, -0.4, -0.9, -1.5, -1.6, -1.2, -1.1, -1.3, -1.3, -0.6, -0.6, -0.4, -0.6, -0.6, -0.7, -0.9, -1.1, -1, -1.3, -2.1, -2.6, -3.1, -3.4, -3.6, -3.8, -4.1, -4.7, -6.4, -6.5, -4.7, -2.4, -0.9, 0.4, 1.8, 2.6, 3.4, 3.8, 3.9, 3.1, 0.5, -1, -1.7, -1.5, -1.7, -3, -3.9, -4.3, -3.9, -4.3, -4.8, -4.8, -3.7, -2.9, -2.1, -1.3, 0.4, 2.3, 4.5, 5.7, 4.7, 5.5, 5.5, 5.2, 3.8, 3.1, 2.7, 2.4, 0.8, -0.3, -0.6, -0.6, -0.6, -1.4, -1.3, -0.7, -0.8, -1.1, 0, 2, 2, 3, 4.3, 5.8, 6.4, 7.8, 7.4, 6.5, 4.5, 3.3, 2.2, 1.3, 0.4, -0.8, -0.6, -0.8, -1.1, -1.6, -2.3, -2, -2.4, -3, -2.5, -0.9, 1.7, 3.4, 5.4, 7.7, 8.4, 8.8, 8.3, 7.4, 4.8, 1.7, 1.1, -0.4, -1, -1.5, -2.2, -2.3, -2.5, -3.2, -3.4, -3.2, -3.5, -3.7, -1.9, 0.9, 2.6, 4.6, 6.2, 7.5, 8.3, 8.4, 8.1, 6.8, 4.9, 3.3, 1.7, 0.6, 0, -1.2, -1.3, 
]

timevec = 0:Ts:length(electricity_price_kWh)-1
datevec = DateTime(Date("2023-02-24")):Dates.Hour(1):DateTime(Date("2023-03-02"), Time(23))

plot(datevec, [electricity_price_kWh/100 external_temperature], lab=["Electricity price SEK" "Temp °C"], title="A week in the end of February")
```
The figure above indicates that the price of electricity usually peaks during the morning and the evening, and is lowest during the night.

For the purposes of this tutorial, we will use a very simple dynamical model. The building is dived into 4 thermal masses, corresponding to walls, floor, circulating water and indoor air, and the time constants ``τ`` associated with heat transfer between them is approximated. While this model is extremely simple, it does capture the essential property of a large thermal inertia smoothing out any variations in external temperature, and also the fact that it takes a while for the added power from the heat pump to affect the indoor temperature due to the thermal inertia of a floor-heating system embedded in concrete.

```@example HVAC
@parameters t

@variables power(t)  = 0
@variables Text(t)   = 10
@variables Twall(t)  = 10
@variables Tint(t)   = 20
@variables Tfloor(t) = 20
@variables Twater(t) = 20

@parameters τwall_ext = 2.0*24
@parameters τwall_int = 0.75*24
@parameters τint_floor = 0.15*24
@parameters τwater_floor = 0.25*24
@parameters input_gain = COP*0.034

D = Differential(t)

eqs = [
    D(Twall) ~ (Text - Twall)/τwall_ext + (Tint - Twall)/τwall_int,
    D(Tint) ~ (Twall - Tint)/τwall_int + (Tfloor - Tint)/τint_floor,
    D(Tfloor) ~ (Tint - Tfloor)/τint_floor + (Twater - Tfloor)/τwater_floor,
    D(Twater) ~ (Tfloor - Twater)/τwater_floor + input_gain*power,
]

@named sys = ODESystem(eqs, t)
```

We now discretize the model using a sampling time of 1 hour.
```@example HVAC
G = named_ss(sys, [power, Text], [Tint]) # Create a state space model
Gd = c2d(G, Ts)  # Discrete-time model

stepres = step(G, 7*24)
plot(stepres, title="One week step response", layout=(2,1), sp=[1 2], xlabel = "Time [hr]")
```

We are now ready to setup the MPC controller. We start with some constants
```@example HVAC
nu = 1                   # number of control inputs
nx = G.nx                # number of states
N  = round(Int, 24/Ts)   # MPC optimization horizon
x0 = fill(18.0, nx)      # Initial state
r  = [20, NaN, NaN, NaN] # Reference
nothing # hide
```

We then define the dynamics and measurement functions
```@example HVAC
dynamics, measurement = let A = to_static(Gd.A), B = to_static(Gd.B), C = to_static(Gd.C), D = to_static(Gd.D), timevec=timevec, external_temperature=external_temperature
    function dynamics(x, u, p, t)
        i = findfirst(==(t), timevec)
        u_tot = SA[u[1]; external_temperature[i]]
        return A * x + B * u_tot
    end
    function measurement(x, u, p, t)
        return C * x
    end
    dynamics, measurement
end

discrete_dynamics = FunctionSystem(dynamics, measurement, Gd.Ts, u=Gd.u[1:1], y=Gd.y[1:1], x = G.x)
nothing # hide
```

We also define cost functions for the MPC controller as well as some constraints. We assume a quadratic cost related to temperature deviations form the set point, increasing the penalty ``Q_1`` will cause the MPC controller to control the temperature more tightly. In effect, this allow us to set some form of "comfort zone", but the concept of a comfort zone could for course be formalized and enforced more rigorously than using a quadratic cost.

The cost related to the input power is the monetary cost of electricity, here measured in SEK.
```@example HVAC
Q1 = 40 # temperature deviation cost
p = (; Q1,)

running_cost = StageCost() do si, p, t
    (; Q1) = p
    e = si.r[1] - (Gd.C*si.x)[]
    u = si.u
    i = findfirst(==(t), timevec)
    e^2*Q1 +
    u[1] * heat_pump_power * electricity_price_kWh[i] * Ts
end

terminal_cost = TerminalCost() do ti, p, t
    e = ti.r[1] - (Gd.C*ti.x)[]
    e^2*10p.Q1
end

objective = Objective(running_cost, terminal_cost)

x = zeros(nx, N+1)
u = 2ones(nu, N)
x, u = MPC.rollout(discrete_dynamics, x0, u, p, 0)
oi = ObjectiveInput(x, u, r)
nothing # hide
```

Here, we include no constraints on the state vector, but including a lower bound on, either the indoor temperature or the maximum budget per day etc. could be considered. We also assume that we have access to the full state for feedback, an unrealistic assumption in general, but with both internal and external thermometers available, not too unrealistic for this assumption to be okay for the purposes of this tutorial. 

```@example HVAC
bounds_constraint = BoundsConstraint(umin = [0], umax = [4], xmin = fill(-Inf, nx), xmax = fill(Inf, nx))
observer = StateFeedback(discrete_dynamics, x0)
nothing # hide
```

We now specify the optimization solver to use in the MPC controller. Since the input is an integer variable, corresponding to one of 4 possible power levels of the compressor, we solve this problem using an MINLP solver, Juniper.jl. We use Ipopt as the internal NLP solver.

```@example HVAC
inner_solver = ()->MPC.IpoptSolver(;
        verbose = false,
        tol = 1e-4,
        acceptable_tol = 1e-3,
        max_iter = 200,
        max_cpu_time = 10.0,
        max_wall_time = 10.0,
        constr_viol_tol = 1e-4,
        acceptable_constr_viol_tol = 1e-3,
        acceptable_iter = 2,
)

using Juniper, OptimizationMOI
const MOI = OptimizationMOI.MOI
solver = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer,
    "nl_solver" => inner_solver, 
    "allow_almost_solved"=>true,
    "allow_almost_solved_integral"=>true,
    "atol"=>1e-5,
    "mip_gap" => 1e-3,
    "solution_limit" => 3,
    "log_levels"=>[],
    "time_limit" => 30,
)
nothing # hide
```
We now package everything into a [`GenericMPCProblem`](@ref) and solve it. We indicate that we have integer control variables using the argument `int_u` and specify that the integer horizon is `Nint = 3`, beyond this horizon, the integer constraint is relaxed for faster solve times.[^2]

[^2]: For the intuition behind this relaxation, see [MPC with binary or integer variables (MINLP)](@ref).
```@example HVAC
prob = GenericMPCProblem(
    discrete_dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
    verbose = false,
    int_u = [true],
    Nint = 3,
);

# Run MPC controller
using Logging: with_logger, NullLogger
history = with_logger(NullLogger()) do # To silence the solver output
    MPC.solve(prob; x0, T = length(timevec)-N-1)
end
nothing # hide
```

For comparison, we solve the same problem also without the economic cost-function term, i.e., with a purely comfort-based cost. 
```@example HVAC
## Non-econominc MPC
running_cost2 = StageCost() do si, p, t
    (; Q1) = p
    e = si.r[1] - (Gd.C*si.x)[]
    u = si.u
    i = findfirst(==(t), timevec)
    e^2*Q1
end

objective2 = Objective(running_cost2, terminal_cost)
prob2 = GenericMPCProblem(
    discrete_dynamics;
    N,
    observer,
    objective = objective2,
    constraints = [bounds_constraint],
    p,
    objective_input = oi,
    solver,
    xr = r,
    presolve = true,
    verbose = false,
    int_u = [true],
    Nint = 3,
);

history2 = with_logger(NullLogger()) do # To silence the solver output
    MPC.solve(prob2; x0, T = length(timevec)-N-1, verbose=isinteractive())
end
nothing # hide
```

To visualize the result, we plot both solutions together. 
```@example HVAC
xticks = 0:24:7*24
X,E,R,U,Y,UE = reduce(hcat, history)
X2,E2,R2,U2,Y2,UE2 = reduce(hcat, history2)

power = U[:]
total_cost = sum(electricity_price_kWh[1:length(power)] .* power .* heat_pump_power) / 100
power2 = U2[:]
total_cost2 = sum(electricity_price_kWh[1:length(power2)] .* power2 .* heat_pump_power) / 100

temp = Y
temp2 = Y2

err = mean(abs2, temp .- r[1])
err2 = mean(abs2, temp2 .- r[1])

using Printf
@printf("Total savings with price-aware predictive control: %2.1f%%\n", 100(total_cost2-total_cost)/total_cost2)
@printf("RMS temp deviation predictive control: %2.1f°C\n", err2)
@printf("RMS temp deviation price-aware predictive control: %2.1f°C\n", err)

plot(timevec[1:length(Y)], [Y[:] U[:]]; seriestype=:steppre, lab=[@sprintf("Internal temperature (economic), RMS: %2.1f°C", err) @sprintf("Heatpump power level (economic), cost: %2.0f SEK", total_cost)], xticks, layout=(2, 1), linewidth=2, size=(800, 800), legendfontsize=7)
plot!(timevec[1:length(Y)], [Y2[:] U2[:]]; seriestype=:steppre, lab=[@sprintf("Internal temperature (comfort), RMS: %2.1f°C", err2) @sprintf("Heatpump power level (comfort), cost: %2.0f SEK", total_cost2)], xticks, plot_title="MINLP MPC for temperature control")
hline!(r[1:1], lab="Setpoint", sp=1, color=1, linestyle=:dash)
plot!(timevec, external_temperature, lab="External temperature", sp=1, ylabel = "Temperature [°C]", xlabel = "Time [h]", margin=5Plots.mm)
plot!(timevec, electricity_price_kWh ./ 100, lab="Electricity price [SEK]", sp=2, seriestype=:steppre, xlabel = "Time [h]")
```

The figure indicates that the economic MPC controller uses more power at night when the electricity is cheap. It usually uses a bit of power during the typical lunch-time dip in prices as well. The comfort controller is oblivious to the price of electricity and is thus choosing to control the temperature much tighter, but does so to a significantly higher price.


# Concluding remarks

In this example, we have made use of an MPC controller with an economically motivated cost function to minimize the price of heating a residential home using a geothermal heat pump, while keeping the indoor temperature at a comfortable level. The MPC controller ran with a prediction horizon of 24hrs and made use of (in this case perfect) forecasts of the spot price of electricity and the outdoor temperature. 

While the economically motivated cost function appeals to the end user, it's also beneficial from a system perspective. The reason for the typically much cheaper electricity price at night is due to the significantly lower demand for electricity at night. By allocating the night hours for heating residential buildings, the peak load on the power grid that otherwise appears in the morning and afternoon is reduced. 

## Simplifying assumptions
In order to simplify the exposition of the MPC problem with an economic objective, this example made numerous simplifying and unrealistic assumptions:
- The model used is very simplistic and a perfect model is assumed in the simulation.
- The price is assumed known 24hrs in advance at all times. In practice, the price is known between 11 and 35 hrs in advance.
- The outside temperature is assumed to be known exactly 24 hrs in advance. In practice, forecasts are inaccurate.
- No external disturbances such as solar irradiation are modeled or simulated. 
- The cost of running the compressor in the heat pump at full capacity is not modeled, neither is the cost associated with frequent starts and stops of the compressor.
- In practice, heat pumps must control not only the temperature of the indoor air, but also consider things like the temperature of the feed water as well as the temperature of the cold circuit to avoid overloading the bore hole etc.
- The efficiency of the bore hole may decrease if too much energy is extracted during a short time period, this effect was not modeled.
- The terminal cost used was naive. This can be improved by using longer-term forecasts or seasonal averages.
- This example did not account for *occupancy*. At times when the building is unoccupied, the temperature can be allowed to deviate much more from the setpoint, thus simultaneously increasing the effective "storage capacity" of the building and reducing the amount of energy required to meet the comfort gaols.
- In practice, one may make use of longer forecasts of both weather and spot prices. This is likely beneficial due to the very long thermal time constants of many buildings. Planned power plant maintenance and unusually cold weather drives up electricity prices, both of which can be forecasted with reasonable accuracy. Thermally inert buildings may leverage the inertia to reduce the power consumption during forecasted price peaks.

Before an economically motivated controller is deployed in practice, it is important to validate the model and the controller in a realistic setting. 
