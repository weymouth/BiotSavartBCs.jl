using WaterLily,BiotSavartBCs,CUDA,StaticArrays

function porous(l;g=l/4,α=0.9,θ=π/20,Re=15e3,T=Float32,mem=Array)
    # mapping
    g,α,θ,thk = T(g),T(α),T(θ),T(1+0.5√2)                       # fix the variable types
    s,c = sincos(θ); R = g*√((1-α)/π); cen = SA{T}[1.5l,2l/3,0] # set geometric parameters
    map(xyz,t) = SA[s c 0; -c s 0; 0 0 1]*(xyz-cen)             # shift and rotate

    # signed distance functions
    plate((x,y,z),t) = hypot(x,y-clamp(y,-l+thk,l-thk))-thk # plate in x,y (infinite in z)
    pores((x,y,z),t) = hypot(abs(y)%g-g/2,z%g-g/2)-R        # y,z circles in mod-g coordinates
    body = AutoBody(plate,map) - AutoBody(pores,map)        # perforate the plate and position it!

    # Simulation with Biot-Savart BCs (but free-slip in z)
    BiotSimulation((5l,2l,l),(1,0,0),2l;ν=2l/Re,body,T,mem,nonbiotfaces=(-3,3))
end

import BiotSavartBCs: interaction,image,symmetry
@inline function symmetry(ω,T,args...) # add (only two) symmetry images in z
    T₁,sgn₁ = image(T,size(ω),-3)
    T₂,sgn₂ = image(T,size(ω),3)
    return interaction(ω,T,args...)+sgn₁*interaction(ω,T₁,args...)+sgn₂*interaction(ω,T₂,args...)
end

# 90% solid
using JLD2,GLMakie
L,α = 96,90
sim = porous(L,mem=CuArray,α=α/100);
mean = MeanFlow(sim.flow);
try
    load!(sim,fname="porous3d_$(α)_$(L)_30.jld2")
    load!(sim,fname="porous3d_$(α)_$(L)_60.jld2")
    load!(mean,fname="porous3d_$(α)_$(L)_mean30_60.jld2")
catch
    viz!(sim,duration=30,video="porous3d_$(α)_$(L).mp4",remeasure=false)
    save!("porous3d_90_$L_30.jld2",sim)
    while sim_time(sim)<60
        sim_step!(sim,sim_time(sim)+0.1,remeasure=false)
        WaterLily.update!(mean,sim.flow)
        @show sim_time(sim)
    end
    save!("porous3d_$(α)_$(L)_60.jld2",sim)
    save!("porous3d_$(α)_$(L)_mean30_60.jld2",mean)
end

fig,ax = viz!(sim)  # move around to a good view
hidespines!(ax)
hidedecorations!(ax)
viz!(sim;fig,ax,colorrange=(0.05,0.85)) # change range to see more detail
save("porous3d_$(α)_$(L)_default.png", current_figure())
viz!(sim;duration=1,step=0.01,fig,ax,video="porous3d_$(α)_$(L)_default.mp4",colorrange=(0.05,0.85))

viz!(sim,mean.P,cut=(0,0,96÷8*3),d=2,clims=(-0.5,0.5),levels=11)
save("porous3d_$(α)_$(L)_meanPo.png", current_figure())
viz!(sim,mean.P,cut=(0,0,96÷2),d=2,clims=(-0.5,0.5),levels=11)
save("porous3d_$(α)_$(L)_meanPc.png", current_figure())
function mean_ω_mag(arr, sim)
    ω = sim.flow.σ
    @inside ω[I] = WaterLily.ω_mag(I,mean.U)
    copyto!(arr, ω[inside(ω)]) # copy to CPU
end
viz!(sim;f=mean_ω_mag,cut=(0,0,96÷8*3),d=2,clims=(-0.5,0.5))
save("porous3d_$(α)_$(L)_meanωo.png", current_figure())
viz!(sim;f=mean_ω_mag,cut=(0,0,96÷2),d=2,clims=(-0.5,0.5))
save("porous3d_$(α)_$(L)_meanωc.png", current_figure())
