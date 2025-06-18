"""
JuliaSim control library.

For help, see https://help.juliahub.com/DyadControlSystems/
"""
module DyadControlSystems
using DyadInterface
import Reexport
using DiffEqBase
using CommonSolve
using RecipesBase
using Convex, Hypatia
import JuMP
using Setfield
using UnPack
import ModelingToolkit
using ThreadTools
using ForwardDiff
Reexport.@reexport using ControlSystems
Reexport.@reexport using RobustAndOptimalControl
using ControlSystemsMTK
using ControlSystemIdentification
import Distributions: MvNormal
using SeeToDee
import Optim
using TimerOutputs
import Colors # To set transparent legends etc.
import ModelingToolkitStandardLibrary.Blocks
connect = ModelingToolkit.connect

using LinearAlgebra
using ChainRules, ForwardDiffChainRules

@ForwardDiff_frule LinearAlgebra.eigvals!(x1::AbstractMatrix{<:ForwardDiff.Dual})


export input_names, output_names, state_names, FunctionSystem, simplified_system
include("app_interface.jl")

export build_controlled_dynamics
include("code_generation.jl")

export frequency_response_analysis
include("linear_analysis.jl")

export hinfsyn_lmi, ispassive_lmi, spr_synthesize
include("hinfsyn.jl")

export mussv, mussv_tv, mussv_DG
include("ssv.jl")

export common_lqr, common_lyap
include("common_lqr.jl")

export StateFeedback, FixedGainObserver
export MvNormal, KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
include("observers.jl")

export inverse_lqr, LMI, Eq29, Eq32, GMF
include("inverse_lqr.jl")

export OperatingPoint
include("MPC/MPC.jl")

export AutoTuningProblem, solve, OptimizedPID
include("autotuning.jl")

export solve
include("autotuning2.jl")

export ESC, PIESC
include("extremum_seeking.jl")

export Stateful, SlidingModeController, SuperTwistingSMC, LinearSlidingModeController
include("smc.jl")

export MaximumSensitivityObjective, MaximumTransferObjective, OvershootObjective, RiseTimeObjective, SettlingTimeObjective, StepTrackingObjective, GainMarginObjective, PhaseMarginObjective, SimulationObjective, StructuredAutoTuningProblem, StructuredAutoTuningResult
include("tuning_objectives.jl")

export trim
include("trimming.jl")

# Contents of this file not part of the public interface
include("model_reducer.jl")

include("pluto_app_launcher.jl")

include("demosystems.jl")

export optimal_trajectory_gen, time_polynomial_trajectory
include("trajectory_optimizer.jl")


export pqr, build_K_function, predicted_cost, poly_approx, safe_lqr
include("pqr.jl")

export analyze_robustness
include("automatic_analysis.jl")
include("dyad/dyad.jl")

end
