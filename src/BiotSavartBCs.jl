module BiotSavartBCs

using WaterLily

include("util.jl")
export MLArray,collect_targets,flatten_targets

include("fmm.jl")

include("tree.jl")

include("velocity.jl")
export fill_ω!,biotBC!,pflowBC!

include("flow.jl")
export biot_mom_step!

"""
   BiotSimulation((WaterLily.Simulation inputs)...; fmm=true)

Constructor for a WaterLily.Simulation that uses the Biot-Savart boundary conditions:
    - fmm: Use the Fast Multi-level Method for the Biot-Savart integral (`fmm=true` default`),
           or use the tree-sum method (`fmm=false`). 

Note that WaterLily.Simulation inputs which set boundary conditions (`exitBC`,`per`) are currently ignored.
See: `Using Biot-Savart boundary conditions for unbounded external flow on Eulerian meshes, https://arxiv.org/abs/2404.09034`
and `WaterLily.Simulation`.

"""
mutable struct BiotSimulation <: AbstractSimulation
    sim  :: Simulation
    ω    :: NTuple
    tar  :: NTuple
    ftar :: AbstractVector
    x₀   :: AbstractArray
    fmm  :: Bool
    nonbiotfaces :: NTuple
    function BiotSimulation(args...; nonbiotfaces=(),  fmm=true, mem = Array, kwargs...)
        # WaterLily simulation
        sim = Simulation(args...; mem, kwargs...)
        # MultiLevel vorticity array (top level points to sim.flow.f)
        ω = MLArray(sim.flow.f) 
        # domain boundary target index arrays
        tar = mem.(collect_targets(ω,nonbiotfaces)); ftar = flatten_targets(tar)
        # holder array for old pressure values
        x₀ = copy(sim.flow.p)
        new(sim,ω,tar,ftar,x₀,fmm)
    end
end
import WaterLily: sim_step! # new type dispatch
function sim_step!(sim::BiotSimulation;remeasure=true,meanflow=nothing,kwargs...)
    remeasure && measure!(sim)
    biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;fmm=sim.fmm,kwargs...)
end
# overload properties
Base.getproperty(f::BiotSimulation, s::Symbol) = s in propertynames(f) ? getfield(f, s) : getfield(f.sim, s)
export BiotSimulation

end # module