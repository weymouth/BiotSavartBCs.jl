using WaterLily,BiotSavartBCs,CUDA,StaticArrays,JLD2,TypedTables
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1)

function porous(l;g=l/4,α=0.9,thk=1/11,θ=π/6,Re=20e3,U=1,T=Float32,mem=Array)
    # define parameters
    g,α,θ = T(g),T(α),T(θ)                                  # fix the variable types
    s,c = sincos(θ); R = g*√((1-α)/π)                       # plate tangent & pore size

    center,corner = SA{T}[1.3l,1.3l,0],SA{T}[l*thk,l]       # plate center & size
    @show g,R,corner

    # define body
    map(xyz,t) = SA[s c 0; -c s 0; 0 0 1]*(xyz-center)      # shift and rotate
    function plate((x,y,z),t)                               # plate in x,y (infinite in z)
        p = abs.(SA[x,y])-corner
        √sum(abs2,max.(p,0))+min(maximum(p),0)
    end
    body = AutoBody(plate,map)                              # make & position the plate
    pores((x,y,z),t) = hypot(abs(y)%g-g/2,z%g-g/2)-R        # y,z circles in mod-g coordinates
    α<1 && (body -= AutoBody(pores,map))                    # perforate the plate

    # Simulation with Biot-Savart BCs (but free-slip in z)
    Ut(i,x,t::T) where T = i==1 ? convert(T,min(t/(2l),U)) : zero(T) # velocity BC
    BiotSimulation((6l,3l,l),Ut,2l;U,ν=2l*U/Re,body,T,mem,nonbiotfaces=(-3,3)),center
end
# Add images
import BiotSavartBCs: symmetry,droste
@inline symmetry(ω,T,args...) = droste(ω,T,3,10,args...) # Droste (hall of mirrors) in z

# Read or simulate & write
function porous_case(L,α;kwargs...)
    sim,x₀ = porous(L,mem=CuArray,α=1-α/100)
    mean = MeanFlow(sim.flow)
    prefix = "porous3d_$(α)_$(L)_$duration"
    hist = try
        load!(sim,mean,prefix)
    catch
        hist = record_hist_mean!(sim,x₀,mean;kwargs...)
        save(prefix,sim,mean,hist)
        hist
    end
    return sim,mean,hist
end
function record_hist_mean!(sim,x₀,mean;start=floor(Int,sim_time(sim)),ramp=10,dt=0.01,duration=20,Z=size(sim.flow.p,3))
    L = sim.L; A = L*Z
    return map(dt:dt:duration) do t
        sim_step!(sim,start+t,remeasure=false)            # update Simulation
        force = 2WaterLily.pressure_force(sim)/A          # compute force
        moment = 2WaterLily.pressure_moment(x₀,sim)/(L*A) # & moment
        t==ramp && WaterLily.reset!(mean)                 # reset mean
        t >ramp && WaterLily.update!(mean,sim.flow)       # accumulate mean
        @show t
        return (;t,Fx=force[1],Fy=force[2],Mz=moment[3])  # record data
    end |> Table
end
function save(prefix,sim,mean,hist)
    WaterLily.save!(prefix*".jld2",sim)
    WaterLily.save!(prefix*"_mean.jld2",mean)
    save_object(prefix*"_hist.jld2",hist)
end
function load!(sim,mean,prefix)
    WaterLily.load!(sim,fname=prefix*".jld2")
    WaterLily.load!(mean,fname=prefix*"_mean.jld2")
    load_object(prefix*"_hist.jld2")
end

using Plots
L,α = 256,0
sim,mean,hist = porous_case(L,100-α);
# sim,x₀ = porous(L,mem=CuArray,α=α/100);
# mean = MeanFlow(sim.flow);
# hist = load!(sim,mean,"porous3d_slip_$(α)_$(L)_20")
# hist = record_hist_mean!(sim,x₀,mean;ramp=-10,duration=1) # no reset or ramp
# save("porous3d_slip_$(α)_$(L)_21",sim,mean,hist)

for α in [0,4,8,10,12,20]
    @show α
    sim,mean,hist = porous_case(L,100-α)
end

pltx = plot(xlabel="Time",ylabel="Force x");
plty = plot(xlabel="Time",ylabel="Force y");
pltz = plot(xlabel="Time",ylabel="Moment z");
for α in [0,4,8,10,12,20]
    @show α
    hist = load_object("porous3d_$(100-α)_$(L)_20_hist.jld2") # just hist
    α == 10 && (hist.t .-= 21) # shift for visibility
    plot!(pltx,hist.t,hist.Fx,label="$α%")
    plot!(plty,hist.t,hist.Fy,label="$α%")
    plot!(pltz,hist.t,hist.Mz,label="$α%")
end
plot!.((pltx,plty,pltz),legend_title="pourosity");
savefig(pltx,"porous3d_L$(L)_Fx.png")
savefig(plty,"porous3d_L$(L)_Fy.png")
savefig(pltz,"porous3d_L$(L)_Mz.png")

# # Visualization
# using GLMakie
# L,α = 256,0
# sim,mean,hist = porous_case(L,100-α);
# fig,ax = viz!(sim)  # move around to a good view
# hidespines!(ax);hidedecorations!(ax);
# viz!(sim;fig,ax,colorrange=(0.05,0.85)); # change range to see more detail
# GLMakie.save("porous3d_$(α)_$(L)_instω.png", current_figure())

# viz!(sim,mean.P,cut=(0,0,L÷8+1),d=2,clims=(-1,1),levels=11)
# GLMakie.save("porous3d_$(α)_$(L)_meanPo.png", current_figure())
# function mean_ω_mag(arr, sim)
#     ω = sim.flow.σ
#     @inside ω[I] = WaterLily.ω_mag(I,mean.U)
#     copyto!(arr, ω[inside(ω)]) # copy to CPU
# end
# viz!(sim;f=mean_ω_mag,cut=(0,0,L÷8+1),d=2,clims=(-0.5,0.5))
# save("porous3d_$(α)_$(L)_meanωo.png", current_figure())
