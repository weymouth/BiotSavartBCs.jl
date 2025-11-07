using WaterLily,BiotSavartBCs,CUDA,StaticArrays,JLD2,TypedTables
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1)

function porous(l;g=l/4,α=0.9,thk=1/11,θ=π/6,Re=20e3,U=1,Z=l,T=Float32,mem=Array)
    # define parameters
    g,α,θ = T(g),T(α),T(θ)                                  # fix the variable types
    s,c = sincos(θ); R = g*√((1-α)/π)                       # plate tangent & pore size

    center,corner = SA{T}[1.5l,1.3l,0],SA{T}[l*thk,l]       # plate center & size
    @show g,R,corner,Z

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
    BiotSimulation((6l,3l,Z),Ut,2l;U,ν=2l*U/Re,body,T,mem,nonbiotfaces=(-3,3)),center
end
# Add images
import BiotSavartBCs: symmetry,droste
@inline symmetry(ω,T,args...) = droste(ω,T,3,3,args...) # Droste (hall of mirrors) in z

# run, save & load
function record_hist_mean!(sim,x₀,mean;start=floor(Int,sim_time(sim)),ramp=10,dt=0.01,duration=20,Z=size(sim.flow.p,3))
    L = sim.L; A = L*Z
    return map(dt:dt:duration) do t
        sim_step!(sim,start+t,remeasure=false)            # update Simulation
        force = 2WaterLily.pressure_force(sim)/A          # compute force
        moment = 2WaterLily.pressure_moment(x₀,sim)/(L*A) # & moment
        t==ramp && WaterLily.reset!(mean)                 # reset mean
        t >ramp && WaterLily.update!(mean,sim.flow)       # accumulate mean
        @show t; flush(stdout)
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

L,Z = 128,48
using Plots
αs = (10,20)

for (i,α) in enumerate(αs)
    @show α; flush(stdout)
    sim,x₀ = porous(L;α=1-α/100,Z=Z*L÷8,mem=CuArray);
    mean = MeanFlow(sim.flow);
    @time hist = record_hist_mean!(sim,x₀,mean)
    save("porous3d_dz$(Z)_$(100-α)_$(L)_20",sim,mean,hist)
end

pltx = plot(xlabel="Time",ylabel="Force x");
plty = plot(xlabel="Time",ylabel="Force y");
pltz = plot(xlabel="Time",ylabel="Moment z");
for α in αs # plot multiple porousities
    hist = load_object("porous3d_dz$(Z)_$(100-α)_$(L)_20_hist.jld2") # just hist
    plot!(pltx,hist.t,hist.Fx,label="$α%")
    plot!(plty,hist.t,hist.Fy,label="$α%")
    plot!(pltz,hist.t,hist.Mz,label="$α%")
end
plot!.((pltx,plty,pltz),legend_title="pourosity");
savefig(pltx,"porous3d_dz$(Z)_L$(L)_Fx.png")
savefig(plty,"porous3d_dz$(Z)_L$(L)_Fy.png")
savefig(pltz,"porous3d_dz$(Z)_L$(L)_Mz.png")

# using Plots
# L,Z = 256,1
# ave = map((4,8,10)) do α
#     hist = load_object("porous3d_dz$(Z)_$(100-α)_$(L)_20_hist.jld2") # just hist
#     Mz = sum(hist.Mz[hist.t .> 10])/length(hist.Mz[hist.t .> 10])
#     σz = √(sum(abs2,hist.Mz[hist.t .> 10] .- Mz)/length(hist.Mz[hist.t .> 10]))
#     (;α,Mz,σz)
# end |> Table
# plot(ave.α,ave.Mz,ribbon=ave.σz,fillalpha=0.5,legend=false,
#     xlabel="Porousity (%)",ylabel="Mz",title="Thin slab simulation pitching moment")
# savefig("porous3d_dz1_L128_mean_Mz.png")