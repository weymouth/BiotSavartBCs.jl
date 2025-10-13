using WaterLily,BiotSavartBCs,CUDA,StaticArrays,JLD2,TypedTables

function porous(l;g=l/4,α=0.9,θ=π/6,Re=20e3,T=Float32,mem=Array)
    # mapping
    g,α,θ,thk = T(g),T(α),T(θ),T(1+0.5√2)                       # fix the variable types
    s,c = sincos(θ); R = g*√((1-α)/π); cen = SA{T}[1.5l,1.3l,0] # set geometric parameters
    map(xyz,t) = SA[s c 0; -c s 0; 0 0 1]*(xyz-cen)             # shift and rotate

    # signed distance functions
    plate((x,y,z),t) = hypot(x,y-clamp(y,-l+thk,l-thk))-thk # plate in x,y (infinite in z)
    body = AutoBody(plate,map)                              # make & position the plate
    pores((x,y,z),t) = hypot(abs(y)%g-g/2,z%g-g/2)-R        # y,z circles in mod-g coordinates
    α<1 && (body -= AutoBody(pores,map))                    # perforate the plate

    # Simulation with Biot-Savart BCs (but free-slip in z)
    BiotSimulation((5l,3l,l),(1,0,0),2l;ν=2l/Re,body,T,mem,nonbiotfaces=(-3,3)),cen
end

import BiotSavartBCs: interaction,image,symmetry
@inline function symmetry(ω,T,args...) # add (only two) symmetry images in z
    T₁,sgn₁ = image(T,size(ω),-3)
    T₂,sgn₂ = image(T,size(ω),3)
    return interaction(ω,T,args...)+sgn₁*interaction(ω,T₁,args...)+sgn₂*interaction(ω,T₂,args...)
end

# Read or simulate & write
function porous_case(L,α;ramp=20,acc=50)
    sim,x₀ = porous(L,mem=CuArray,α=α/100)
    mean = MeanFlow(sim.flow)
    hist = try
        load!(sim,fname="porous3d_$(α)_$(L)_$acc.jld2")
        load!(mean,fname="porous3d_$(α)_$(L)_mean$(ramp)_$acc.jld2")
        load_object("porous3d_$(α)_$(L)_histT_$acc.jld2")
    catch
        hist = map(0.1:0.1:acc) do t
            sim_step!(sim,t,remeasure=false)                   # update to time t
            force = WaterLily.pressure_force(sim)/L^2          # compute force & moment
            moment = WaterLily.pressure_moment(x₀,sim)/2L^3
            t==ramp && (mean = MeanFlow(sim.flow))             # reset mean
            t>ramp && WaterLily.update!(mean,sim.flow)         # accumulate mean
            @show t
            return (;t,Fx=force[1],Fy=force[2],Mz=moment[3])   # record data
        end |> Table
        save!("porous3d_$(α)_$(L)_$acc.jld2",sim)
        save!("porous3d_$(α)_$(L)_mean$(ramp)_$acc.jld2",mean)
        save_object("porous3d_$(α)_$(L)_histT_$acc.jld2",hist)
        hist
    end
    return sim,mean,hist
end

# # Convergence study
# conv = Table(L=[32,64,96,128],ramp=[30,30,20,12])
# # data = map(conv) do case
# #     L,α,ramp,acc=case.L,90,case.ramp,case.ramp+30
# #     sim,mean,hist = porous_case(L,α;ramp,acc) # run/grab everything
# # end;

# # Plot moment time traces
# using Plots
# plot(xlabel="Time",ylabel="moment");
# data = map(conv) do case
#     L,α,ramp,acc=case.L,90,case.ramp,case.ramp+30
#     hist = load_object("porous3d_$(α)_$(L)_histT_$acc.jld2") # just hist
#     plot!(hist.t,hist.Mz,label=2L)
#     # save time-averages after ramp
#     mean(var) = sum(var[30 .<= hist.t .<=42])/length(var[30 .<= hist.t .<=42])
#     (case...,Fx=mean(hist.Fx),Fy=mean(hist.Fy),Mz=mean(hist.Mz))
# end
# plot!(legend_title="plate resolution")
# savefig("porous3d_90_moment.png")

# Porousity study
using Plots
plot(xlabel="Time",ylabel="moment");
for α in 100 .- [0,4,8,10,12,20]
    @show α
    sim,mean,hist = porous_case(96,α)
    plot!(hist.t,hist.Mz,label="$(100-α)%")
end
plot!(legend_title="pourosity")

# # Visualization
# using GLMakie 
# sim,mean,hist = porous_case(32,96,ramp=1,acc=2); # 4% porous
# fig,ax = viz!(sim,body=true)  # move around to a good view
# hidespines!(ax)
# hidedecorations!(ax)
# viz!(sim;fig,ax,colorrange=(0.05,0.85)) # change range to see more detail
# save("porous3d_$(α)_$(L)_default.png", current_figure())

# viz!(sim,mean.P,cut=(0,0,96÷8*3),d=2,clims=(-0.5,0.5),levels=11)
# save("porous3d_$(α)_$(L)_meanPo.png", current_figure())
# viz!(sim,mean.P,cut=(0,0,96÷2),d=2,clims=(-0.5,0.5),levels=11)
# save("porous3d_$(α)_$(L)_meanPc.png", current_figure())
# function mean_ω_mag(arr, sim)
#     ω = sim.flow.σ
#     @inside ω[I] = WaterLily.ω_mag(I,mean.U)
#     copyto!(arr, ω[inside(ω)]) # copy to CPU
# end
# viz!(sim;f=mean_ω_mag,cut=(0,0,96÷8*3),d=2,clims=(-0.5,0.5))
# save("porous3d_$(α)_$(L)_meanωo.png", current_figure())
# viz!(sim;f=mean_ω_mag,cut=(0,0,96÷2),d=2,clims=(-0.5,0.5))
# save("porous3d_$(α)_$(L)_meanωc.png", current_figure())
