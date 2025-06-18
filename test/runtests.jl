# ENV["DEBUG"] = "loading"
using DyadControlSystems
using Test
using SeeToDee
using Plots
using LinearAlgebra
using ControlSystemsBase

if haskey(ENV, "CI")
    ENV["PLOTS_TEST"] = "true"
    ENV["GKSwstype"] = "100" # gr segfault workaround
end

@testset "DyadControlSystems.jl" begin

    @testset "Dyad integration" begin
        @info "Testing Dyad integration"
        @testset "pid analysis" begin
            @info "Testing pid analysis"
            include("dyad/test_pid_analysis.jl")
        end

        @testset "closed_loop_analysis" begin
            @info "Testing closed_loop_analysis"
            include("dyad/test_closed_loop_analysis.jl")
        end

        @testset "closed_loop_sensitivity_analysis" begin
            @info "Testing closed_loop_sensitivity_analysis"
            include("dyad/test_closed_loop_sensitivity_analysis.jl")
        end

        @testset "linear_analysis" begin
            @info "Testing linear_analysis"
            include("dyad/test_linear_analysis.jl")
        end
    end

    @testset "automatic_analysis" begin
        @info "Testing automatic_analysis"
        include("test_analyze_robustness.jl")
    end

    @testset "inverse_lqr" begin
        @info "Testing inverse_lqr"
        include("test_inverse_lqr.jl")
    end
    @testset "hinfsyn" begin
        @info "Testing hinfsyn"
        include("test_hinfsyn.jl")
    end
    @testset "mussv" begin
        @info "Testing mussv"
        include("test_mussv.jl")
    end
    @testset "common_lqr" begin
        @info "Testing common_lqr"
        include("test_common_lqr.jl")
    end
    @testset "app_interface" begin
        @info "Testing app_interface"
        include("test_app_interface.jl")
    end
    @testset "MPC" begin
        @info "Testing MPC"
        @testset "integrators" begin
            @info "Testing integrators"
            # include("MPC/test_integrators.jl")
        end
        @testset "mpc_prediction_models" begin
            @info "Testing prediction_models"
            include("MPC/test_prediction_models.jl")
        end
        @testset "mpc_qp" begin
            @info "Testing mpc_qp"
            include("MPC/test_mpc_qp.jl")
        end
        @testset "Q3" begin
            @info "Testing Q3"
            include("MPC/test_Q3.jl")
        end
        @testset "mpc_linear" begin
            @info "Testing mpc_linear"
            include("MPC/test_mpc_linear.jl")
        end
        @testset "mpc_collocation" begin
            @info "Testing collocationFinE"
            include("MPC/test_collocation.jl")
        end
        @testset "nonlinear_mpc_on_linear_sys" begin
            @info "Testing mpc_linear"
            include("MPC/test_nonlinear_mpc_on_linear_sys.jl")
        end
        @testset "observers" begin
            @info "Testing observers"
            include("MPC/test_observers.jl")
        end
        @testset "quadtank" begin
            @info "Testing quadtank"
            include("MPC/test_quadtank.jl")
        end
        @testset "generic_linear" begin
            @info "Testing generic_linear"
            include("MPC/test_generic_linear.jl")
        end
        @test_skip @testset "codegen" begin
            @info "Testing codegen"
            include("MPC/test_codegen.jl")
        end # broken due to https://github.com/SciML/ModelingToolkit.jl/issues/2000
        @testset "robust_mpc" begin
            @info "Testing robust_mpc"
            include("MPC/test_robust_mpc.jl")
        end
        @testset "input_integration_bounds_constraints" begin
            @info "Testing input_integration_bounds_constraints"
            include("MPC/test_input_integration_bounds_constraints.jl")
        end
        @testset "integer_vars" begin
            @info "Testing integer_vars"
            include("MPC/test_integer_vars.jl")
        end
    end
    @testset "tuning_objectives" begin
        @info "Testing tuning_objectives"
        include("test_tuning_objectives.jl")
    end
    @testset "tuning_objectives" begin
        @info "Testing tuning_objectives"
        include("test_tuning_objectives_evaluation.jl")
    end
    @testset "build_controlled_dynamics" begin
        @info "Testing build_controlled_dynamics"
        include("test_build_controlled_dynamics.jl")
    end
    @testset "fra" begin
        @info "Testing fra"
        include("test_fra.jl")
    end
    @testset "autotuning" begin
        @info "Testing autotuning"
        include("test_autotuning.jl")
    end
    @testset "autotuning2" begin
        @info "Testing autotuning2"
        include("test_autotuning2.jl")
    end
    @testset "extremum" begin
        @info "Testing extremum"
        include("test_extremum.jl")
    end
    @testset "smc" begin
        @info "Testing smc"
        include("test_smc.jl")
    end
    @testset "Autotuning app" begin
        @info "Testing Autotuning app"
        include("../notebooks/autotuning.jl")
    end
    @testset "Model reduction app" begin
        @info "Testing Model reduction app"
        include("../notebooks/model_reduction.jl")
    end
    # @testset "Trimming" begin
    #     @info "Testing Trimming feature"
    #     include("test_trimming.jl")
    # end
    @testset "demosystems" begin
        @info "Testing demosystems"
        include("test_demosystems.jl")
    end
    @testset "Optimal trajectory generation" begin
        @info "Testing trajectory generation"
        include("test_trajectory_optimizer.jl")
    end
    @testset "pqr" begin
        @info "Testing pqr"
        include("test_pqr.jl")
    end
    # @testset "MPC app" begin
    #     @info "Testing MPC app"
    #     include("../notebooks/mpc.jl")
    # end
end
