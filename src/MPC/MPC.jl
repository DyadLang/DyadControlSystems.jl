module MPC

using LowLevelParticleFilters
using LowLevelParticleFilters: AbstractFilter, state, dynamics, predict!, correct!
using UnPack
using RobustAndOptimalControl
using RobustAndOptimalControl: dare3
using DiffEqBase
using ControlSystems
import DyadControlSystems:
    StateFeedback,
    OperatingPoint,
    OperatingPointWrapper,
    FunctionSystem,
    linearize,
    issystem,
    input_names,
    state_names,
    output_names,
    reset_observer!,
    mpc_observer_predict!,
    mpc_observer_correct!,
    GMF,
    inverse_lqr,
    safe_lqr
    
import DyadControlSystems: rk4

# reexport
export GMF, inverse_lqr, OperatingPoint

export ObjectiveInput, get_xu, ControllerInput, ControllerOutput, ObserverInput
include("signals.jl")
include("generic_constraints.jl")

export MPCConstraints, NonlinearMPCConstraints
include("constraints.jl")

export LinearPredictionModel, LinearMPCModel, RobustMPCModel, MPCConstraints, LinearMPCConstraints
include("LinearPredictionModel2.jl")

export MPCIntegrator
include("integrators.jl")

export sparse, spdiagm, speye, QMPCProblem, LQMPCProblem, rk4, Colloc, solve, optimize!, step!
include("mpc_qp_nonlinear.jl")
include("mpc_qp_linear.jl")

export OSQPSolver
include("solver_osqp_nonlinear.jl")
include("solver_osqp_linear.jl")


export StageConstraint, BoundsConstraint, TrajectoryConstraint, MultipleShooting, CollocationFinE, Trapezoidal, CompositeMPCConstraint, GenericMPCProblem, StageCost, DifferenceCost, DifferenceConstraint, TerminalCost, TerminalStateConstraint, Objective, IpoptSolver, MPCParameters

include("generic_costs.jl")
include("discretization.jl")
include("solver_optimization.jl")

get_reference(p::AbstractMPCProblem) = p.r
get_reference(p::QMPCProblem) = p.xr

end
