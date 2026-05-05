# BiotSavartBCs.jl

## Agent Routing Note
- This repo's local CLAUDE file is this project-root `CLAUDE.md`.
- Read the global `c:\Users\gweymouth\.claude\CLAUDE.md` first, then this local file.
- Write top-level persistent directives to the global `c:\Users\gweymouth\.claude\CLAUDE.md` file.
- On conversation compaction or resume, reread both the global file and this project-root `CLAUDE.md` before continuing.


## Package Purpose
This package extends the WaterLily.jl package with a boundary condition based on the BiotSavart reconstruction of the velocity from the vorticity. See the README.md   

## Key Types

- `BiotSimulation` — constructor function returning a plain `WaterLily.Simulation` injected with `BiotSavartPoisson` via `pois_ctor`. (Was a wrapper struct; now just a convenience function.)
- `BiotSavartPoisson <: WaterLily.AbstractPoisson` — custom Poisson type carrying the Biot-Savart state. Fields:
  - `ml::MultiLevelPoisson` — wrapped standard pressure solver (all pressure system state lives here)
  - `ω::NTuple` — multi-level vorticity (top level aliases `flow.f`)
  - `tar::NTuple` / `ftar::AbstractVector` — domain boundary target index arrays (per level, and flattened)
  - `x₀::AbstractArray` — accumulator for the pressure solution across outer Newton iterations
  - `fmm::Bool` — toggles Fast Multi-level Method vs tree-sum for the Biot-Savart integral
- `MLArray(u)` — builds a multi-level tuple of arrays at halving resolutions; top level aliases `u`

## Entry Points

- `BiotSimulation(dims, uBC, L; body, ν, fmm=true, nonbiotfaces=(), mem=Array, kwargs...)` — constructs a simulation using Biot-Savart boundary conditions. Drop-in replacement for `WaterLily.Simulation`; returns a plain `Simulation`.
- `sim_step!(sim; remeasure=true, ...)` — inherited from WaterLily; dispatches through `mom_step!` → `WaterLily.mom_project!(::AbstractFlow, ::BiotSavartPoisson, w, t)` for the Biot-Savart projection.
- `biotBC!(u, uBC, ω, tar, ftar, t=0; fmm=true)` — sets velocity on domain boundary faces from the Biot-Savart integral. `uBC` may be a `Function(i,x,t)` or a plain `Tuple`. Can be called standalone.
- `fill_ω!(ω, u)` — computes curl of `u` into the multi-level vorticity array.
- `pflowBC!(u)` — enforces div=0 / curl=0 on ghost cells after projection.

## Source Layout

- `src/BiotSavartBCs.jl` — module entry: includes, exports, `BiotSimulation` constructor
- `src/BiotSavartPoisson.jl` — `BiotSavartPoisson` struct + constructor, `WaterLily.mom_project!` override, `WaterLily.update!` delegation, `apply_grad_p!`
- `src/velocity.jl` — `fill_ω!`, `biotBC!`, `biotBC_r!`, `pflowBC!`, `fix_resid!`
- `src/fmm.jl` — Fast Multi-level Method: `interaction!`, `fmmBC!`, `symmetry`
- `src/tree.jl` — tree-sum alternative: `tree`, `treeBC!`
- `src/util.jl` — `MLArray`, `collect_targets`, `flatten_targets`, `restrict!`, `project!`, `@vecloop`

## Pending

- WaterLily PR: add `WaterLily.update!(::AbstractPoisson) = nothing` fallback to `Flow-refactor` branch, so all `AbstractPoisson` subtypes don't need to define it explicitly.