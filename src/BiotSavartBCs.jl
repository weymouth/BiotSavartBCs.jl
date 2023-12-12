module BiotSavartBCs

using WaterLily
include("vorticity.jl")
export MLArray,fill_ω!

include("velocity.jl")
export u_ω

include("util.jl")
export biotBC!,pflowBC!,fix_resid!

include("flow.jl")
export biot_mom_step!
end
