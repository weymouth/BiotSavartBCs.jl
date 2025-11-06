using WaterLily,BiotSavartBCs,CUDA,StaticArrays,JLD2,TypedTables
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=3)

function porous(l;g=l/4,α=0.9,thk=1/11,θ=π/6,Re=20e3,Z=6,U=1,T=Float32,mem=Array)
    # define parameters
    g,α,θ = T(g),T(α),T(θ)                                  # fix the variable types
    s,c = sincos(θ); R = g*√((1-α)/π)                       # plate tangent & pore size

    center,corner = SA{T}[1.3l,1.3l,0],SA{T}[l*thk,l,Z*l-0.5l] # plate center & size
    @show g,R,corner

    # define body
    map(xyz,t) = SA[s c 0; -c s 0; 0 0 1]*(xyz-center)      # shift and rotate
    function plate(x,t)                               # plate in x,y (infinite in z)
        p = abs.(x)-corner
        √sum(abs2,max.(p,0))+min(maximum(p),0)
    end
    body = AutoBody(plate,map)                              # make & position the plate
    pores((x,y,z),t) = hypot(abs(y)%g-g/2,z%g-g/2)-R        # y,z circles in mod-g coordinates
    α<1 && (body -= AutoBody(pores,map))                    # perforate the plate

    # Simulation with Biot-Savart BCs (but free-slip in z)
    Ut(i,x,t::T) where T = i==1 ? convert(T,min(t/(2l),U)) : zero(T) # velocity BC
    BiotSimulation((6l,3l,Z*l),Ut,2l;U,ν=2l*U/Re,body,T,mem,nonbiotfaces=(-3,)),center
end
# Add images
import BiotSavartBCs: interaction,image,symmetry
@inline function symmetry(ω,T,args...) # add symmetry image in z
    T₁,sgn₁ = image(T,size(ω),-3)
    return interaction(ω,T,args...)+sgn₁*interaction(ω,T₁,args...)
end

# Simulate, save & load
function record_hist_mean!(sim,x₀,mean;start=floor(Int,sim_time(sim)),ramp=10,dt=0.01,duration=20)
    L = sim.L; A = L*size(sim.flow.p,3)
    return map(dt:dt:duration) do t
        sim_step!(sim,start+t,remeasure=false)            # update Simulation
        force = 2WaterLily.pressure_force(sim)/A          # compute force
        moment = 2WaterLily.pressure_moment(x₀,sim)/(L*A) # & moment
        t==ramp && WaterLily.reset!(mean)                 # reset mean
        t >ramp && WaterLily.update!(mean,sim.flow)       # accumulate mean
        @show t
        return (;t,Fx=force[1],Fy=force[2],Mz=moment[3])   # record data
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

L,α,Z = 128,10,8
sim,x₀ = porous(L,α=1-α/100;Z);
mean = MeanFlow(sim.flow);
hist = load!(sim,mean,"porous3d_finite_$(Z)_$(100-α)_$(L)_20")
plot(hist.t,hist.Mz,label="Moment",xlabel="Time",ylabel=nothing)

using Plots
A = sim.L^2*(Z-0.5)/2
2WaterLily.pressure_moment(x₀,mean.P,sim.flow.f,sim.body)/(sim.L*A)
2WaterLily.pressure_force(mean.P,sim.flow.f,sim.body)/A

using WaterLily: @loop,cross,loc,nds
function Mz_z(x₀,p,df,body,dz,t=0)
    Tp = eltype(p);
    df .= zero(Tp)
    @loop df[I,:] .= p[I]*cross(loc(0,I,Tp)-x₀,nds(body,loc(0,I,Tp),t)) over I ∈ inside(p)
    [sum(@view df[:,:,2+(n-1)*dz:1+n*dz,3]) for n in 1:(size(p,3)÷dz)]
end
dz = sim.L ÷ 8

Mz8 = 2Mz_z(x₀,mean.P,sim.flow.f,sim.body,dz)/(sim.L^2*dz)
# Mz6 = 2Mz_z(x₀,mean.P,sim.flow.f,sim.body,dz)/(sim.L^2*dz)
# Mz3 = 2Mz_z(x₀,mean.P,sim.flow.f,sim.body,dz)/(sim.L^2*dz)
# MzInf = 2WaterLily.pressure_moment(x₀,mean.P,sim.flow.f,sim.body)[3]/(sim.L*A)
scatter(Mz8,xlabel="z slice",ylabel="Moment z per slice",label="AR=8");scatter!(Mz6,label="AR=6");scatter!(Mz3,label="AR=3");hline!([MzInf],label="AR=∞")
savefig("porous3d_finite_$(α)_$(L)_Mz_slices.png")