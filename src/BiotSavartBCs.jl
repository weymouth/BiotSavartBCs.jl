module BiotSavartBCs

using WaterLily
include("vorticity.jl")
export MLArray,fill_ω!
include("velocity.jl")
include("util.jl")
include("flow.jl")
end
