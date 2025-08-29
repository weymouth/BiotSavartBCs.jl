using WaterLily,StaticArrays,BiotSavartBCs
function sym_square(N;Re=5e2,mem=Array,U=1,T=Float32,thk=2,L=T(N/2))
    body = AutoBody() do xyz,t
        x,y,z = xyz - SA[L,0,0]
        √(x^2+(y-min(y,L-thk))^2+(z-min(z,L-thk))^2)-thk
    end
    BiotSimulation((2N,N,N), (U,0,0),L;ν=U*2L/Re,body,mem,T,nonbiotfaces=(-2,-3))
end
using Plots
function vorticity_slice(flow;kwargs...)
    @inside flow.σ[I] = WaterLily.curl(3,I,flow.u)
    ω = @view flow.σ[inside(flow.σ)]
    flood(WaterLily.squeeze(ω[:,:,2]);kwargs...)
end

using CUDA
import BiotSavartBCs: interaction,symmetry,image
sim_slip_walls = sym_square(96,mem=CuArray);
# @inline symmetry(ω,T,args...) = interaction(ω,T,args...) # this is the default with no images
sim_step!(sim_slip_walls,2,remeasure=false)
vorticity_slice(sim_slip_walls.flow,clims=(-0.5,0.5))

sim_sym_walls = sym_square(96,mem=CuArray); # no difference!
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₂,sgn₂ = image(T,size(ω),-2)  # image target and sign in y
    T₃,sgn₃ = image(T,size(ω),-3)  # image target and sign in z
    T₂₃,_   = image(T₃,size(ω),-2) # image of image!
    # Add up the four contributions
    return interaction(ω,T,args...)+sgn₃*interaction(ω,T₃,args...)+
     sgn₂*(interaction(ω,T₂,args...)+sgn₃*interaction(ω,T₂₃,args...))
end
sim_step!(sim_sym_walls,2,remeasure=false)
vorticity_slice(sim_sym_walls.flow,clims=(-0.5,0.5))