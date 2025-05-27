ENV["GKSwstype"] = 322 # workaround for gr segfault on GH actions
using Pkg
# Pkg.add(name="Documenter", rev="master")

using Documenter
using DyadControlSystems
using DyadControlSystems.MPC
using Plots
import Plots.Measures
using ControlSystems, RobustAndOptimalControl, LowLevelParticleFilters, SeeToDee

const buildpath = haskey(ENV, "CI") ? ".." : ""

default(margin=5*Measures.mm)

ENV["JULIA_DEBUG"]=nothing # Enable this for debugging

makedocs(
    sitename = "DyadControlSystems",
    modules = [DyadControlSystems, ControlSystems, RobustAndOptimalControl, LowLevelParticleFilters, SeeToDee],
    remotes = Dict(
        dirname(dirname(pathof(ControlSystems))) => (Remotes.GitHub("JuliaControl", "ControlSystems.jl"), "1"),
        dirname(dirname(pathof(ControlSystemsBase))) => (Remotes.GitHub("JuliaControl", "ControlSystems.jl"), "1"),
        dirname(dirname(pathof(RobustAndOptimalControl))) => (Remotes.GitHub("JuliaControl", "RobustAndOptimalControl.jl"), "0.4"),
        dirname(dirname(pathof(SeeToDee))) => (Remotes.GitHub("baggepinnen", "SeeToDee.jl"), "1"),
        dirname(dirname(pathof(LowLevelParticleFilters))) => (Remotes.GitHub("baggepinnen", "LowLevelParticleFilters.jl"), "3"),
    ),
    warnonly = [:missing_docs, :cross_references],
    doctest = false,
    pagesonly = false,
    draft = false,
    pages = [
        "Home" => "index.md",
        "Getting started with linear control" => "getting_started_linear.md",
        "Topic tutorials" => [
            "MPC tutorials" => [
                "MPC with binary or integer variables (MINLP)" => "examples/minlp_mpc.md",
                "Adaptive MPC" => "examples/adaptive_mpc.md",
                "Robust MPC tuning using Glover-McFarlane" => "examples/robust_mpc_gmf.md",
                "Robust MPC with uncertain parameters" => "examples/robust_mpc.md",
                "Disturbance modeling" => "examples/disturbance_rejection_mpc.md",
                "Mixed-sensitivity modeling" => "examples/mixed_sensitivity_mpc.md",
                "MPC with Neural Surrogate" => "examples/mpc_neural_surrogate.md",
            ],
            "Optimal control tutorials" => [
                "Optimal Control (Trajectory Optimization)" => "examples/optimal_control.md",
                # "Optimal Control with ModelingToolkit models" => "examples/optimal_control_mtk.md",
            ],
            "ModelingToolkit for control" => [
                "Modeling for control using ModelingToolkit" => "examples/mtk_control.md",
                # "Optimal Control with ModelingToolkit models" => "examples/optimal_control_mtk.md",
                "Disturbance modeling with ModelingToolkit" => "examples/mtk_disturbance_modeling.md", # MTKstdlib parameter handling
                "State estimation" => "examples/state_estimation.md",
                # "Control of PDE systems" => "examples/pde_control.md", # SymbolicUtils compat problem
            ],
            "Linear control tutorials" =>[
                "``H_\\infty`` control design" => "examples/hinf_design.md",
                "Passive ``H_2`` control design" => "examples/passive_synthesis.md",
                "MIMO robust stability using μ-analysis" => "examples/mimo_robust_stability.md",
                "Uncertainty-aware LQR" => "examples/uncertainty_aware_LQR.md",
            ],
        ],
        "Workflow examples" => [
            "MPC examples" => [
                "Using a model estimated from data" => "examples/sysid_mpc.md",
                # "Controlling an aircraft" => "examples/rcam_mpc.md", # Polynomials v4 compat -> StackOverFlow due to old version of SymbolicUtils
                "Controlling a Continuously Stirred Tank Reactor (CSTR)" => "examples/cstr_mpc.md",
                "Autonomous lane changes" => "examples/self_driving_car.md",
                "Economic MPC for residential HVAC" => "examples/hvac.md",
                "Redundant actuators" => "examples/redundant_control.md",
                "Control design for quadruple tank" => "examples/quadtank.md",
            ],
            "Linear control examples" => [
                "Control design for quadruple tank" => "examples/quadtank.md",
            ],
        ],
        "MPC" => [
            "MPC Home" => "mpc.md",
            "Quadratic MPC" => "quadratic_mpc.md",
            "Generic MPC" => "generic_mpc.md",
            "MPC details" => "mpc_details.md",
        ],
        "Controller autotuning" => [
            "PID Autotuning" => "autotuning.md",
            "Tuning of structured controllers in ModelingToolkit" => "tuning_objectives.md",
        ],
        "Robust control" => "robust_control.md",
        "Integral action" => "integral_action.md",
        "Linear analysis" => "linear_analysis.md",
        "Simulation of systems with inputs" => "simulation.md",
        # "Trimming" => "trimming.md", # attempt to access ModelingToolkit.MTKParameters{Tuple{Vector{Float64}}, Tuple{}, Tuple{}, Tuple{}, Tuple{}, Nothing, Nothing} at index [26]
        "Model reduction" => "model_reduction.md",
        "System identification" => "sysid.md",
        "Nonlinear control" => [
            "Polynomial-quadratic control (PQR)" => "pqr.md",
            "Extremum seeking control" => "extremum_seeking.md",
            "Sliding-Mode Control" => "smc.md",
        ],
        "Apps" => "gui_apps.md",
        "API documentation" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = haskey(ENV, "CI"),
        edit_link = nothing,
        canonical = "https://JuliaComputing.github.io/DyadControlSystems.jl",
        size_threshold = 10000000,
    ),
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/JuliaComputing/DyadControlSystems.jl.git",
    branch = "gh-pages", # gh-pages is the default branch, just making it explicit
    push_preview = true
)
