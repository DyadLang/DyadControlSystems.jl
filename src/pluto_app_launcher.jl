import PlutoVSCodeApp as PA
using RelocatableFolders

# This needs to be declared at the module level, as otherwise
# RelocatableFolders will not be able to store the files on the module
const NOTEBOOKS_PATH = @path joinpath(@__DIR__, "..", "notebooks")

"""
    app_workspace = app_modelreduction(P)

Start a graphical user interface (GUI) for model reduction of linear systems.

`P::StateSpace` is a linear system of high order to be reduced.
`app_workspace` contains the entire workspace of the model-reduction app. The reduced-order system is available as `app_workspace.Pr`
"""
function app_modelreduction(P::AbstractStateSpace; kwargs...)
    app = PA.create_app(
        joinpath(NOTEBOOKS_PATH, "model_reduction.jl"),
        [:P => P];
        kwargs...
    )
    PA.open_app(app) # this will open the app inside a VS Code tab
    PA.wait_for_done(app)
    PA.shutdown(app)
    PA.workspace_module(app)
end

"""
    app_workspace = app_autotuning(P)

Start a graphical user interface (GUI) for autotuning of PID controllers for linear systems.
`app_workspace` contains the entire workspace of the autotuning app.
- The optimization result structure is available as `app_workspace.res`. See [`AutoTuningResult`](@ref).
- The final settings for the autotuning problem solved by the app are aviable as `app_workspace.prob`, see [`AutoTuningProblem`](@ref).
"""
function app_autotuning(P; kwargs...)
    ControlSystemsBase.issiso(P) || throw(ArgumentError("The autotuning app only supports SISO systems."))
    app = PA.create_app(
        joinpath(NOTEBOOKS_PATH, "autotuning.jl"),
        [:P => P];
        kwargs...
    )
    PA.open_app(app) # this will open the app inside a VS Code tab
    PA.wait_for_done(app)
    PA.shutdown(app)
    PA.workspace_module(app)
end
