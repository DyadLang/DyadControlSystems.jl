using DyadControlSystems
using ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Mechanical.Rotational
using ModelingToolkitStandardLibrary.Blocks
using Plots
using Test
t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits
connect = ModelingToolkit.connect

@mtkmodel DCMotor begin
    @parameters begin
        R = 0.5, [description = "Armature resistance"] # Ohm
        L = 4.5e-3, [description = "Armature inductance"] # H
        k = 0.5, [description = "Motor constant"] # N.m/A
        J = 0.02, [description = "Inertia"] # kg.m²
        f = 0.01, [description = "Friction factor"] # N.m.s/rad
        tau_L_step = -0.3, [description = "Amplitude of the load torque step"] # N.m
    end
    @components begin
        ground = Ground()
        source = Voltage()
        ref = Blocks.Step(height = 1, start_time = 0)
        pi_controller = Blocks.LimPI(k = 1.1, T = 0.035, u_max = 10, Ta = 0.035)
        feedback = Blocks.Feedback()
        R1 = Resistor(R = R)
        L1 = Inductor(L = L, i=0.0)
        emf = EMF(k = k)
        fixed = Fixed()
        load = Torque()
        load_step = Blocks.Step(height = tau_L_step, start_time = 3)
        inertia = Inertia(J = J, phi=0.0, w=0.0)
        friction = Damper(d = f)
        speed_sensor = SpeedSensor()
    end
    @equations begin
        connect(fixed.flange, emf.support, friction.flange_b)
        connect(emf.flange, friction.flange_a, inertia.flange_a)
        connect(inertia.flange_b, load.flange)
        connect(inertia.flange_b, speed_sensor.flange)
        connect(load_step.output, load.tau)
        connect(ref.output, :r, feedback.input1)
        connect(speed_sensor.w, :y, feedback.input2)
        connect(feedback.output, pi_controller.err_input)
        connect(pi_controller.ctr_output, :u, source.V)
        connect(source.p, R1.p)
        connect(R1.n, L1.p)
        connect(L1.n, emf.p)
        connect(emf.n, source.n, ground.g)
    end
end

@named model = DCMotor()

inputs = [model.u]
outputs = [model.y]

cm = complete(model)
op = Dict()
using DyadControlSystems
import DyadControlSystems as JSC

# Verify that the model linearizes
lsys = named_ss(model, inputs, outputs; op, loop_openings=[:y])

spec = JSC.ClosedLoopAnalysisSpec(;
    name = :DCMotorAnalysis,
    model,
    measurement = "y",
    control_input = "u",
    duration = 0.25,
)


asol = JSC.run_analysis(spec)

default(size=(1200, 1200), titlefontsize=10)
Splot = JSC.artifacts(asol, :all)



## What happens if there is no controller in the model when we run the analysis?

@mtkmodel DCMotorWithoutController begin
    @parameters begin
        R = 0.5, [description = "Armature resistance"] # Ohm
        L = 4.5e-3, [description = "Armature inductance"] # H
        k = 0.5, [description = "Motor constant"] # N.m/A
        J = 0.02, [description = "Inertia"] # kg.m²
        f = 0.01, [description = "Friction factor"] # N.m.s/rad
        tau_L_step = -0.3, [description = "Amplitude of the load torque step"] # N.m
    end
    @components begin
        ground = Ground()
        source = Voltage()
        ref = Blocks.Step(height = 1, start_time = 0)
        R1 = Resistor(R = R)
        L1 = Inductor(L = L, i=0.0)
        emf = EMF(k = k)
        fixed = Fixed()
        load = Torque()
        load_step = Blocks.Step(height = tau_L_step, start_time = 3)
        inertia = Inertia(J = J, phi=0.0, w=0.0)
        friction = Damper(d = f)
        speed_sensor = SpeedSensor()
        sink = RealInput()
    end
    @equations begin
        connect(fixed.flange, emf.support, friction.flange_b)
        connect(emf.flange, friction.flange_a, inertia.flange_a)
        connect(inertia.flange_b, load.flange)
        connect(inertia.flange_b, speed_sensor.flange)
        connect(load_step.output, load.tau)
        connect(ref.output, :u, source.V)
        connect(speed_sensor.w, :y, sink)
        connect(source.p, R1.p)
        connect(R1.n, L1.p)
        connect(L1.n, emf.p)
        connect(emf.n, source.n, ground.g)
    end
end

@named model = DCMotorWithoutController()

# Verify that the model linearizes
lsys = named_ss(model, inputs, outputs; op, loop_openings=[:y])


spec = JSC.ClosedLoopAnalysisSpec(;
    name = :DCMotorWithoutControllerAnalysis,
    model,
    measurement = "y",
    control_input = "u",
    duration = 0.25,
)


asol = JSC.run_analysis(spec)

Splot = JSC.artifacts(asol, :all)

