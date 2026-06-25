module BiotSavartBCs

using WaterLily

include("ka.jl")
include("geom.jl")
include("multilevel.jl")
export MLArray,collect_targets,flatten_targets

include("fmm.jl")
include("tree.jl")
include("velocity.jl")
export fill_ω!,biotBC!,pflowBC!

include("BiotSavartPoisson.jl")
export BiotSavartPoisson

"""
   BiotSimulation((WaterLily.Simulation inputs)...; fmm=true, nonbiotfaces=(), nimages=4, mem=Array)

Constructor for a WaterLily.Simulation that uses Biot-Savart boundary conditions.
Returns a plain `WaterLily.Simulation` with a `BiotSavartPoisson` solver injected via `pois_ctor`.

- `fmm`: Use the Fast Multi-level Method (`true`, default) or tree-sum (`false`).
- `nonbiotfaces`: tuple of face indices to exclude from Biot-Savart BCs (e.g. `(-2,)` for the negative-y face).
- `nimages`: number of periodic image copies per side for each direction in `perdir` (default `4`).
- `mem`: memory backend (`Array`, `CuArray`, etc.).

Pass `perdir` as a keyword argument (forwarded to `WaterLily.Simulation`) to enable periodic BCs.
Periodic faces are automatically excluded from Biot-Savart targets; the image sum accounts for
their vorticity contribution at the remaining (Biot-Savart) faces.

See: `Using Biot-Savart boundary conditions for unbounded external flow on Eulerian meshes,
https://arxiv.org/abs/2404.09034` and `WaterLily.Simulation`.
"""
function BiotSimulation(args...; nonbiotfaces=(), fmm=true, mem=Array, nimages=4, kwargs...)
    Simulation(args...; mem,
        pois_ctor=flow->BiotSavartPoisson(flow; nonbiotfaces, fmm, mem, nimages),
        kwargs...)
end
export BiotSimulation

end # module