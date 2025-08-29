using WaterLily,StaticArrays,BiotSavartBCs
function jelly(p=5;Re=5e2,mem=Array,U=1,T=Float32)
    # Define simulation size & geometry dimensions
    n = 2^p; R = T(2n/3); h = 4n-2R

    # Motion functions
    ω = 2U/R
    A(t) = 1 .- SA[1,1,0]*cos(ω*t)/10     # radial stretch
    B(t) = SA[0,0,1]*((cos(ω*t)-1)*R/4-h) # axial shift
    C(t) = SA[0,0,1]*sin(ω*t)*R/4         # axial stretch

    # Build jelly from a mapped sphere and plane
    sphere = AutoBody((x,t)->abs(√sum(abs2,x)-R)-1, # sdf
                      (x,t)->A(t).*x+B(t)+C(t))     # map
    plane = AutoBody((x,t)->x[3]-h,(x,t)->x+C(t))
    body =  sphere-plane

    # Return initialized simulation
    BiotSimulation((n,n,4n),(0,0,-U),R;ν=U*R/Re,body,mem,T,nonbiotfaces=(-1,-2))
end

import BiotSavartBCs: interaction,image,symmetry
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₁,sgn₁ = image(T,size(ω),-1)
    T₂,sgn₂ = image(T,size(ω),-2)
    T₁₂,_   = image(T₁,size(ω),-2)
    return interaction(ω,T,args...)+sgn₁*interaction(ω,T₁,args...)+
        sgn₂*(interaction(ω,T₂,args...)+sgn₁*interaction(ω,T₁₂,args...))
end

using GLMakie,CUDA
Makie.inline!(false)
CUDA.allowscalar(false)
function copymirrorto!(mCPU,CPU,GPU)
    n = size(inside(GPU),1)
    copyto!(CPU,GPU[inside(GPU)])
    mCPU[reverse(1:n),reverse(1:n),:].=CPU
    mCPU[reverse(n+1:2n),1:n,:].=mCPU[1:n,1:n,:]
    mCPU[:,reverse(n+1:2n),:].=mCPU[:,1:n,:]
    return mCPU
end
function ω!(mirrored,data,sim)
    @inside sim.flow.σ[I] = WaterLily.ω_mag(I,sim.flow.u)*sim.L/sim.U
    copymirrorto!(mirrored,data,sim.flow.σ)
end
using Meshing
function geom!(mirrored,data,sim)
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    get_body(copymirrorto!(mirrored,data,sim.flow.σ),Val(true))
end
begin
    # Define geometry and motion on GPU
    sim = jelly(mem=CUDA.CuArray);
    sim_step!(sim,sim_time(sim)+0.05);

    # Create CPU buffer array for mirrored viz
    buffer = Array{Float32}(undef,size(inside(sim.flow.σ))) # hold data on CPU
    mirrored = Array{Float32}(undef,(2,2,1).*size(buffer))  # hold mirrored data

    # Set up geometry viz
    geom = geom!(mirrored,buffer,sim) |> Observable;
    fig, _, _ = GLMakie.mesh(geom, alpha=0.1, color=:aqua) #:red

    #Set up flow viz
    ω = ω!(mirrored,buffer,sim) |> Observable;
    volume!(ω, algorithm=:mip, colormap=:algae, colorrange=(0.5,16)) #:amp
    fig
end

# Loop in time
GLMakie.record(fig,"jelly.mp4",1:300) do frame
# foreach(1:100) do frame
    @show frame 
    sim_step!(sim,sim_time(sim)+0.05);
    geom[] = geom!(mirrored,buffer,sim);
    ω[] = ω!(mirrored,buffer,sim);
end