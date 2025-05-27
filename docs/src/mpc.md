# Model-Predictive Control (MPC)
## Introduction

Model-Predictive Control refers to the process of using a prediction model to simulate the future response of the controlled system, and an optimizer to optimize the future control signal trajectory to optimize a cost function over the simulation, subject to constraints. When an optimal trajectory is found, the first control input from this trajectory is applied to the controlled system, a new measurement is obtained, the estimate of the current state is updated and the process is repeated. An MPC controller is thus a model-based feedback controller capable of incorporating constraints, making it a generally applicable advanced control method.

The prediction model used by the optimizer may be either linear or nonlinear, and the same applies to the state observer. The following documentation will demonstrate how to setup, tune and simulate MPC controllers.

The workflow for designing an MPC controller is roughly split up into
1. Specifying the dynamics. This can be done in several ways:
    - Using [`LinearMPCModel`](@ref) for standard linear MPC.
    - Using [`RobustMPCModel`](@ref) for linear MPC with robust loop shaping, including integral action etc.
    - Using a [`FunctionSystem`](@ref) that accepts a nonlinear discrete-time dynamics function with signature `(x,u,p,t) -> x⁺`, or a continuous-time function with signature `(x,u,p,t) -> ẋ`. Learn more under [Discretization](@ref).
2. Defining a state observer.
3. Specifying an MPC solver.
4. Defining an MPC problem containing things like
    - Prediction horizon
    - State and control constraints
    - The dynamics and the observer

The currently defined problem types are
- [`LQMPCProblem`](@ref): Linear Quadratic MPC problem. The cost is on the form $z^T Q_1 z + u^T Q_2 u$ where $z = C_z x$ are the controlled outputs.
- [`QMPCProblem`](@ref): Quadratic Nonlinear MPC problem. The cost is on the form $x^T Q_1 x + u^T Q_2 u$.
- [`GenericMPCProblem`](@ref): An MPC problem that allows the user to specify arbitrary costs and constraints, sometimes referred to as *economic MPC*. This documentation uses the terms, linear-quadratic, nonlinear-quadratic and generic MPC to refer to different variations of the MPC problem.


See docstrings and examples for further information. The following sections contains details around each step mentioned above. See the [Getting started](@ref getting_started_mpc) section below to navigate the topic.

## [Getting started](@ref getting_started_mpc)

The MPC documentation is divided into the two main sections
- [MPC with quadratic cost functions](@ref)
- [MPC with generic cost and constraints](@ref)
each of which has its own references to help you get started. 

Common references for both sections are provided under [MPC Details](@ref)

Here are some quick links to help you navigate the documentation:
- **Examples**: see examples provided under either [MPC with quadratic cost functions](@ref) or [MPC with generic cost and constraints](@ref).
- **State observers/ state estimation**: see [Observers](@ref).
- **Integral action**: see [Integral action](@ref).
- **Discretization**: to convert continuous-time dynamics to discrete time, see [Discretization](@ref)
- **Constraints**: see [Constraints](@ref) and [MPC with model estimated from data](@ref)
- **Reference handling and reference preview**: see [Getting started: Linear system](@ref)
- **Disturbance modeling**: see [Disturbance modeling and rejection with MPC controllers](@ref), [Mixed-sensitivity $\mathcal{H}_2$ design for MPC controllers](@ref) and [model augmentation in RobustAndOptimalControl.jl](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/#Model-augmentation)
- **Tuning an MPC controller**: see [Integral action and robust tuning](@ref) and [Mixed-sensitivity $\mathcal{H}_2$ design for MPC controllers](@ref)


