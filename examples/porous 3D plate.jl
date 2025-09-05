using WaterLily,BiotSavartBCs,Plots,CUDA,StaticArrays

function porous(l;g=l/4,α=0.9,θ=π/20,Re=15e3,T=Float32,mem=Array)
    g,α,θ,thk = T(g),T(α),T(θ),T(1+0.5√2)                     # fix the variable types
    s,c = sincos(θ); R = g*√((1-α)/π); cen = SA{T}[1.5l,2l/3,0]
    @show R,g,α,(π*R^2)/g^2
    map(xyz,t) = SA[s c 0; -c s 0; 0 0 1]*(xyz-cen)           # shift and rotate
    plate((x,y,z),t) = hypot(x,y-clamp(y,-l+thk,l-thk))-thk   # semi-infinite plate
    pores((x,y,z),t) = hypot(abs(y)%g-g/2,z%g-g/2)-R          # defined mod-g
    body = AutoBody(plate,map) - AutoBody(pores,map)          # perforate the plate!
    BiotSimulation((5l,2l,l),(1,0,0),2l;ν=2l/Re,body,T,mem,   # Simulation with Biot-Savart BCs
        nonbiotfaces=(-3,3)) # but free-slip in z
end

import BiotSavartBCs: interaction,image,symmetry
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₁,sgn₁ = image(T,size(ω),-3)
    T₂,sgn₂ = image(T,size(ω),3)
    return interaction(ω,T,args...)+sgn₁*interaction(ω,T₁,args...)+sgn₂*interaction(ω,T₂,args...)
end

# 90% solid
sim = porous(96,mem=CuArray,α=0.9);sim_step!(sim)
sim_step!(sim,30,remeasure=false)
save_object("porous_90_96_30.jld2",sim)
mean = MeanFlow(sim.flow)
while sim_time(sim)<60
    sim_step!(sim)
    WaterLily.update!(mean,sim.flow)
end
using JLD2
save_object("porous_90_96_60.jld2",sim)
save_object("porous_90_96_mean30_60.jld2",mean)