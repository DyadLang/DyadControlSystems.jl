


## MPC 

using DyadControlSystems, Plots
using DyadControlSystems.MPC
using StaticArrays
using LinearAlgebra
using Dates


Ts = 1 # Sample time hrs
COP = 4.9 # COP of the heat pump
heat_pump_power = 1 # kW per stage
power_to_temp = 2.5 # degC/kWh




electricity_price_kWh = [
    66.31, 53.18, 49.77, 49.75, 55.27, 63.03, 95.82, 111.66, 113.93, 111.02, 109.19, 107.38, 106.05, 102.12, 101.26, 100.94, 106.01, 107.78, 106.79, 98.46, 58.21, 37.47, 32.67, 30.94, 27.8, 26.27, 25.72, 25.47, 25.66, 26.16, 26.91, 30.86, 36.25, 37.47, 36.41, 36.29, 34.13, 33.47, 32.49, 32.31, 33.92, 39.72, 67.16, 39.71, 34.61, 33.02, 32.45, 32.35, 35.8, 37.13, 38.85, 39.3, 39.84, 44.36, 55.24, 68.03, 110.68, 102.85, 107.8, 102.92, 93.51, 60.91, 60.46, 94.18, 109.55, 140.4, 166.13, 169.59, 126.65, 121.84, 106.61, 70.83, 73.01, 82.79, 120.53, 126.65, 129.78, 151.13, 181.87, 206.71, 201.97, 177.29, 164.2, 146.86, 149.8, 146.48, 145.59, 146.37, 169.94, 189.03, 205.37, 214.17, 188.61, 173.26, 165.87, 129.65, 127.18, 110.64, 33.17, 25.95, 88.79, 152.61, 175.33, 196.8, 200.76, 163.51, 140.7, 134.02, 125.95, 128.63, 129.09, 143.74, 166.64, 186.83, 209.22, 209.12, 188.73, 170.8, 127.23, 66.42, 66.94, 49.26, 67.04, 88.75, 133.68, 152.9, 181.15, 206.18, 206.18, 169.46, 140.41, 131.3, 128.08, 126.48, 127.61, 132.84, 146.53, 180.37, 203.59, 197.39, 176.04, 156.69, 153.66, 123.99, 117.55, 114.74, 114.73, 117.56, 140.68, 146.7, 178.54, 200.72, 196.93, 153.05, 137.88, 129.49, 124.86, 122.97, 127.35, 139.18, 165.18, 180.04, 200.09, 145.23, 117.59, 114.65, 25.98, 29.96, 
]

external_temperature = [ # https://www.smhi.se/data/meteorologi/ladda-ner-meteorologiska-observationer/#param=airtemperatureInstant,stations=core,stationid=53530
2.7, 2.2, 2, 1.5, 1, 1.4, 1.9, 1.6, 1.5, 1.3, 1.5, 1, 1.9, 3.8, 4.9, 2, 2.6, 2, 1.8, 1.2, 0.8, 0.9, 0.6, 0.6, 0.2, 0.1, 0, -0.4, -0.9, -1.5, -1.6, -1.2, -1.1, -1.3, -1.3, -0.6, -0.6, -0.4, -0.6, -0.6, -0.7, -0.9, -1.1, -1, -1.3, -2.1, -2.6, -3.1, -3.4, -3.6, -3.8, -4.1, -4.7, -6.4, -6.5, -4.7, -2.4, -0.9, 0.4, 1.8, 2.6, 3.4, 3.8, 3.9, 3.1, 0.5, -1, -1.7, -1.5, -1.7, -3, -3.9, -4.3, -3.9, -4.3, -4.8, -4.8, -3.7, -2.9, -2.1, -1.3, 0.4, 2.3, 4.5, 5.7, 4.7, 5.5, 5.5, 5.2, 3.8, 3.1, 2.7, 2.4, 0.8, -0.3, -0.6, -0.6, -0.6, -1.4, -1.3, -0.7, -0.8, -1.1, 0, 2, 2, 3, 4.3, 5.8, 6.4, 7.8, 7.4, 6.5, 4.5, 3.3, 2.2, 1.3, 0.4, -0.8, -0.6, -0.8, -1.1, -1.6, -2.3, -2, -2.4, -3, -2.5, -0.9, 1.7, 3.4, 5.4, 7.7, 8.4, 8.8, 8.3, 7.4, 4.8, 1.7, 1.1, -0.4, -1, -1.5, -2.2, -2.3, -2.5, -3.2, -3.4, -3.2, -3.5, -3.7, -1.9, 0.9, 2.6, 4.6, 6.2, 7.5, 8.3, 8.4, 8.1, 6.8, 4.9, 3.3, 1.7, 0.6, 0, -1.2, -1.3, 
]

timevec = 0:Ts:length(electricity_price_kWh)-1
datevec = DateTime(Date("2023-02-24")):Dates.Hour(1):DateTime(Date("2023-03-02"), Time(23))


plot(datevec, [electricity_price_kWh/100 external_temperature], lab=["Electricity price SEK" "Temp °C"])


# Gyu = named_ss(ss(tf(sqrt(power_to_temp*COP*heat_pump_power), [24, 1])^2), :heatpump, u=:power_level, y=:Tin)
# Gyd = named_ss(ss(tf(1, [4*24, 1])), :ambient, u=:Text, y=:Text)

# G = [Gyu Gyd]


using ModelingToolkit
using ControlSystems
using ControlSystemsMTK
@parameters t

@variables power(t) = 0
@variables Text(t) = 10
@variables Twall(t) = 10
@variables Tint(t) = 20
@variables Tfloor(t) = 20
@variables Twater(t) = 20
D = Differential(t)

@parameters τwall_ext = 2.5*24
@parameters τwall_int = 0.75*24
@parameters τint_floor = 0.15*24
@parameters τwater_floor = 0.25*24
@parameters gain = 10/50

eqs = [
    D(Twall) ~ (Text - Twall)/τwall_ext + (Tint - Twall)/τwall_int,
    D(Tint) ~ (Twall - Tint)/τwall_int + (Tfloor - Tint)/τint_floor,
    D(Tfloor) ~ (Tint - Tfloor)/τint_floor + (Twater - Tfloor)/τwater_floor,
    D(Twater) ~ (Tfloor - Twater)/τwater_floor + gain*power,
]

@named sys = ODESystem(eqs, t)

# lsys_sym, _ = ModelingToolkit.linearize_symbolic(sys, [power, Text], [Tint])
# lsys_sym.A

G = named_ss(sys, [power, Text], [Tint])



Gd = c2d(G, Ts)

res = step(G, 7*24)

plot(res)


nu = 1               # number of control inputs
nx = G.nx               # number of states
N  = round(Int, 24/Ts) # MPC optimization horizon
x0 = fill(18.0, nx)    # Initial state
r  = [20, NaN, NaN, NaN]


# discrete_dynamics = FunctionSystem(Gd)


dynamics, measurement = let A = to_static(Gd.A), B = to_static(Gd.B), C = to_static(Gd.C), D = to_static(Gd.D), timevec=timevec
    function dynamics(x, u, p, t)
        # @show t
        i = findfirst(==(t), timevec)
        u_tot = SA[u[1]; external_temperature[i]]
        return A * x + B * u_tot
    end
    function measurement(x, u, p, t)
        return C * x
    end
    dynamics, measurement
end

discrete_dynamics = FunctionSystem(dynamics, measurement, Gd.Ts, u=Gyu.u, y=Gyu.y, x = G.x)

Q1 = 40 # output cost
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


bounds_constraint = BoundsConstraint(umin = [0], umax = [4], xmin = fill(-Inf, nx), xmax = fill(Inf, nx))
observer = StateFeedback(discrete_dynamics, x0)

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
solver = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer, "nl_solver" => inner_solver, 
"allow_almost_solved"=>true,
"allow_almost_solved_integral"=>true,
"atol"=>1e-5,
"mip_gap" => 1e-3,
"solution_limit" => 3,
"log_levels"=>[],
"time_limit" => 10,
"feasibility_pump" => true,
)
##
prob = GenericMPCProblem(
    discrete_dynamics;
    N,
    observer,
    objective,
    constraints = [bounds_constraint],
    p,
    objective_input = oi,
    solver,# =inner_solver(),
    xr = r,
    presolve = true,
    verbose = true,
    int_u = [true],
    Nint = 3,
    # hessian_method = :forwarddiff,
);

# Run MPC controller
@time history = MPC.solve(prob; x0, T = length(timevec)-N-1, verbose=true)




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
    solver,# =inner_solver(),
    xr = r,
    presolve = true,
    verbose = true,
    int_u = [true],
    Nint = 3,
    # hessian_method = :forwarddiff,
);

@time history2 = MPC.solve(prob2; x0, T = length(timevec)-N-1, verbose=true)


xticks = 0:24:7*24
X,E,R,U,Y,UE = reduce(hcat, history)
X2,E2,R2,U2,Y2,UE2 = reduce(hcat, history2)

power = U[:]
total_cost = sum(electricity_price_kWh[1:length(power)] .* power .* heat_pump_power) / 100
power2 = U2[:]
total_cost2 = sum(electricity_price_kWh[1:length(power2)] .* power2 .* heat_pump_power) / 100

mean(power) 
mean(power2)


temp = Y
temp2 = Y2

err = mean(abs2, temp .- r[1])
err2 = mean(abs2, temp2 .- r[1])

using Printf
@printf("Total savings with price-aware predictive control: %2.1f%%\n", 100(total_cost2-total_cost)/total_cost2)
@printf("RMS temp deviation predictive control: %2.1f°C\n", err2)
@printf("RMS temp deviation price-aware predictive control: %2.1f°C\n", err)


plot(timevec[1:length(Y)], [Y[:] U[:]]; seriestype=:steppre, lab=[@sprintf("Internal temperature (economic), RMS: %2.1f°C", err) @sprintf("Heatpump power level (economic), cost: %2.0f SEK", total_cost)], xticks, layout=(2, 1), linewidth=2, legendfontsize=7)
plot!(timevec[1:length(Y)], [Y2[:] U2[:]]; seriestype=:steppre, lab=[@sprintf("Internal temperature (comfort), RMS: %2.1f°C", err2) @sprintf("Heatpump power level (comfort), cost: %2.0f SEK", total_cost2)], xticks, plot_title="MINLP MPC for temperature control")
hline!(r[1:1], lab="Setpoint", sp=1, color=1, linestyle=:dash)
plot!(timevec, external_temperature, lab="External temperature", sp=1, seriestype=:steppre, ylabel = "Temperature [°C]", xlabel = "Time [h]", margin=5Plots.mm)
plot!(timevec, electricity_price_kWh ./ 100, lab="Electricity price [SEK]", sp=2, seriestype=:steppre, xlabel = "Time [h]")
display(current())



# ## More realistic model
# using ModelingToolkit
# using ControlSystems
# using ControlSystemsMTK
# @parameters t

# @variables power(t) = 0
# @variables Text(t) = 10
# @variables Twall(t) = 10
# @variables Tint(t) = 20
# @variables Tfloor(t) = 20
# @variables Twater(t) = 20
# D = Differential(t)

# @parameters τwall_ext = 2.5*24
# @parameters τwall_int = 0.75*24
# @parameters τint_floor = 0.15*24
# @parameters τwater_floor = 0.25*24
# @parameters gain = 10/50

# eqs = [
#     D(Twall) ~ (Text - Twall)/τwall_ext + (Tint - Twall)/τwall_int,
#     D(Tint) ~ (Twall - Tint)/τwall_int + (Tfloor - Tint)/τint_floor,
#     D(Tfloor) ~ (Tint - Tfloor)/τint_floor + (Twater - Tfloor)/τwater_floor,
#     D(Twater) ~ (Tfloor - Twater)/τwater_floor + gain*power,
# ]

# @named sys = ODESystem(eqs, t)

# # lsys_sym, _ = ModelingToolkit.linearize_symbolic(sys, [power, Text], [Tint])
# # lsys_sym.A

# lsys = named_ss(sys, [power, Text], [Tint])
# # dampreport(lsys)
# bodeplot(lsys, hz=true)

# plot(step(lsys*Diagonal([1,10]), 40*24))
# plot(step(G*Diagonal([1,10]), 40*24))
