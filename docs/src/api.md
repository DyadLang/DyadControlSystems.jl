# Exported functions and types
In addition to the documentation for this package, we encourage the reader to explore the documentation for
[ControlSystems.jl](https://juliacontrol.github.io/ControlSystems.jl/stable/) and [RobustAndOptimalControl.jl](https://juliacontrol.github.io/RobustAndOptimalControl.jl/stable/) that contains functionality and types for basic control analysis and design, as well as the documentation of [ModelingToolkit](https://mtk.sciml.ai/dev/) for modeling and simulation.
## Index

```@index
```

# Docstrings
## DyadControlSystems
!!! note
    Docstrings of the MPC submodule are located under [MPC](@ref).

```@autodocs
Modules = [DyadControlSystems]
Pages = [
    "inverse_lqr.jl"
    "observers.jl"
    "ssv.jl"
    "hinfsyn.jl"
    "code_generation.jl"
    "app_interface.jl"
]
Private = false
```

## ControlSystems and RobustAndOptimalControl
```@autodocs
Modules = [ControlSystems, ControlSystemsBase, RobustAndOptimalControl, LowLevelParticleFilters]
Private = false
```