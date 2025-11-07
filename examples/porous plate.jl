using WaterLily,BiotSavartBCs,CUDA,StaticArrays,JLD2,TypedTables
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1)

function porous(l;g=l/4,α=0.9,thk=1/11,θ=π/6,Re=2e3,U=1,T=Float32,mem=Array)
    # define parameters
    g,α,θ = T(g),T(α),T(θ)                          # fix the variable types
    s,c = sincos(θ); R = g*√((1-α)/π)               # plate tangent & pore size

    center,corner = SA{T}[1.5l,1.3l],SA{T}[l*thk,l] # plate center & size
    @show g,R,corner

    # define body
    map(xy,t) = SA[s c; -c s]*(xy-center)           # shift and rotate
    function plate(xy,t)                            # plate in x,y
        p = abs.(xy)-corner
        √sum(abs2,max.(p,0))+min(maximum(p),0)
    end
    body = AutoBody(plate,map)                      # make & position the plate
    pores((x,y),t) = abs(abs(y)%g-g/2)-R            # y slots in mod-g coordinates
    α<1 && (body -= AutoBody(pores,map))            # perforate the plate

    # Simulation with Biot-Savart BCs (but free-slip in z)
    Ut(i,x,t::T) where T = i==1 ? convert(T,min(t/(2l),U)) : zero(T) # velocity BC
    BiotSimulation((6l,3l),Ut,2l;U,ν=2l*U/Re,body,T,mem),center
end

# Read or simulate & write
function porous_case(L,α;ramp=20,dt=0.05,duration=40)
    sim,x₀ = porous(L,mem=CuArray,α=α/100)
    mean = MeanFlow(sim.flow)
    prefix = "porous2d_$(α)_$(L)_$duration"
    hist = try
        load!(sim,mean,prefix)
    catch
        hist = record_hist_mean!(sim,x₀,mean;ramp,dt,duration)
        save(prefix,sim,mean,hist)
        hist
    end
    return sim,mean,hist
end
function record_hist_mean!(sim,x₀,mean;start=floor(Int,sim_time(sim)),ramp=10,dt=0.01,duration=20)
    L = sim.L
    return map(dt:dt:duration) do t
        sim_step!(sim,start+t,remeasure=false)            # update Simulation
        force = 2WaterLily.pressure_force(sim)/L          # compute force
        moment = 2WaterLily.pressure_moment(x₀,sim)/L^2   # & moment
        t==ramp && WaterLily.reset!(mean)                 # reset mean
        t >ramp && WaterLily.update!(mean,sim.flow)       # accumulate mean
        @show t
        return (;t,Fx=force[1],Fy=force[2],Mz=moment[1])  # record data
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

L = 128
for α in [0,4,8,10,12,20]
    @show α
    sim,mean,hist = porous_case(L,100-α)
end

using Plots
pltx = plot(xlabel="Time",ylabel="Force x");
plty = plot(xlabel="Time",ylabel="Force y");
pltz = plot(xlabel="Time",ylabel="Moment z");
for α in [0,4,8,10,12,20]
    @show α
    hist = load_object("porous2d_$(100-α)_$(L)_40_hist.jld2") # just hist
    plot!(pltx,hist.t,hist.Fx,label="$α%")
    plot!(plty,hist.t,hist.Fy,label="$α%")
    plot!(pltz,hist.t,hist.Mz,label="$α%")
end
plot!.((pltx,plty,pltz),legend_title="pourosity");
savefig(pltx,"porous2d_L$(L)_Fx.png")
savefig(plty,"porous2d_L$(L)_Fy.png")
savefig(pltz,"porous2d_L$(L)_Mz.png")

using GLMakie
α = 10
sim,mean,hist = porous_case(L,100-α);
sim.flow.u⁰ .= mean.U;
function mean_u(arr, sim)
    u = sim.flow.σ
    @inside u[I] = √(2WaterLily.ke(I,sim.flow.u⁰))
    copyto!(arr, u[inside(u)]) # copy to CPU
end
viz!(sim,f=mean_u,levels=11,clims=(0,1))
GLMakie.save("porous2d_$(α)_$(L)_meanU.png", current_figure())

function mean_ω(arr, sim)
    ω = sim.flow.σ
    @inside ω[I] = WaterLily.curl(3,I,sim.flow.u⁰)
    copyto!(arr, ω[inside(ω)]) # copy to CPU
end
viz!(sim,f=mean_ω,clims=(-0.25,0.25))
GLMakie.save("porous2d_$(α)_$(L)_meanω.png", current_figure())
