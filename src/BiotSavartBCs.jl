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

end
