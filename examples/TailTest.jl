using WaterLily, CUDA
using StaticArrays
using LinearAlgebra: ×,⋅
using BiotSavartBCs, WriteVTK

dot2(x) = x ⋅ x
function udTriangle(p, a, b, c)
    ba = b - a; pa = p - a;
    cb = c - b; pb = p - b;
    ac = a - c; pc = p - c;
    nor = ba × ac;

    return √(
        (sign((ba × nor) ⋅ pa) +
        sign((cb × nor) ⋅ pb) +
        sign((ac × nor) ⋅ pc)<2)
        ?
        min( min(
        dot2(ba*clamp((ba ⋅ pa)/dot2(ba),0,1)-pa),
        dot2(cb*clamp((cb ⋅ pb)/dot2(cb),0,1)-pb) ),
        dot2(ac*clamp((ac ⋅ pc)/dot2(ac),0,1)-pc) )
        :
        (nor ⋅ pa)^2/dot2(nor) )
end

function simple_tail(L=64;U=1,Re=200,St=0.6,α₀=π/18,mem=Array,T=Float32,use_biotsavart=false)
    # triangle tail SDF
    S = L/√3 # side half-length
    tail(x,t)=udTriangle(x,SA[0,0,0],SA[L,0,S],SA[L,0,-S])-(1+√3)/2

    # pitch and heave mapping
    ω = π*St*U/S; pivot = L/4*SA[1,6,4]; θ₀ = atan(ω*0.3L/U)-α₀
    @show θ₀
    function flap(x,t)
        θ = θ₀*cos(ω*t); h = 0.3L*sin(ω*t)
        SA[cos(θ) sin(θ) 0; -sin(θ) cos(θ) 0; 0 0 1]*(x - pivot + h*SA[0,1,0])
    end 

    # make simulation
    use_biotsavart && return BiotSimulation((4L,3L,2L),(U,0,0),L;ν=U*L/Re,mem,T,body=AutoBody(tail,flap))
    Simulation((4L,3L,2L),(U,0,0),L;ν=U*L/Re,mem,T,body=AutoBody(tail,flap))
end

using Meshing, GeometryBasics,GLMakie
function geom!(d,sim)
    t=WaterLily.time(sim)
    a = sim.flow.σ
    WaterLily.measure_sdf!(a,sim.body,t)
    copyto!(d,a[inside(a)]) # copy to CPU
    mesh = GeometryBasics.Mesh(d,Meshing.MarchingCubes(),origin=Vec(0,0,0),widths=size(d))
    normal_mesh(mesh)
end

function ω!(d,sim)
    a,dt = sim.flow.σ,sim.L/sim.U
    @inside a[I] = WaterLily.ω_mag(I,sim.flow.u)*dt
    copyto!(d,a[inside(a)]) # copy to CPU
end

update(sim,t; geomonly = false) = while sim_time(sim) < t
    geomonly && (push!(sim.flow.Δt,sim.flow.Δt[end]); continue)
    sim_step!(sim;remeasure=true)
end

L,St,biotsavart=32,0.6,true
begin
    # Define geometry and motion on GPU
    sim = simple_tail(L,St=St,mem=Array,use_biotsavart=biotsavart)
    update(sim,0.001)

    # Create CPU buffer arrays for geometry flow viz 
    a = sim.flow.σ
    d = similar(a,size(inside(a))) |> Array

    # Set up geometry viz
    geom = geom!(d,sim) |> Observable;
    fig, _, _ = GLMakie.mesh(geom)#, alpha=0.1, color=:red)

    #Set up flow viz
    ω = ω!(d,sim) |> Observable;
    volume!(ω, algorithm=:mip, colormap=:algae, colorrange=(5,50))
    fig
end
update(sim,1/√3/St)
# Loop in time
# GLMakie.record(fig,"fish.mp4",1:50) do frame
steps,periods = 80,1
GLMakie.record(fig,"tail_true.mp4",1:(steps*periods)) do frame
# foreach(1:(steps*periods)) do frame
    stop = sim_time(sim)+(2/√3)/(St*steps)
    @show frame
    update(sim,stop)
    geom[] = geom!(d,sim)
    ω[] = ω!(d,sim);
end

function ave_slice!(sim,t,ω_ml,d; biotsavart = false)
    # set up midplane slice
    N = size(d)
    z_slice = N[3]÷2
    u_bar = zeros(N[1],N[2])

    # accumulate u
    T₀ = length(sim.flow.Δt)
    while sim_time(sim) < t
        sim_step!(sim;remeasure=true)
        copyto!(d,sim.flow.u[inside(sim.flow.p),1]) # copy to CPU
        u_bar .+= d[:,:,z_slice] # sum
        @show sim_time(sim)
    end

    # average and return
    u_bar ./= (length(sim.flow.Δt)-T₀)
    return u_bar
end
u_bar = ave_slice!(sim,sim_time(sim)+6/√3/St,ω_ml,d;biotsavart)
using JLD2
save_object("u_bar_true.jld2",u_bar)
using Plots
Plots.contourf(u_bar',clims=(0.4,1.5),levels=10,linewidth=0,aspect_ratio=:equal)
savefig("u_bar_true.png")
1

# # Define geometry and motion on GPU
# sim = simple_tail(32,mem=CuArray,use_biotsavart=true);
   
# # make a writer with some attributes, need to output to CPU array to save file (|> Array)
# vort(a::Simulation) = (@WaterLily.loop sim.flow.f[I,:] .= WaterLily.ω(I,sim.flow.u) over I in inside(sim.flow.p);
#                        a.flow.f |> Array)
# _body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); 
#                                      a.flow.σ |> Array;)
# lamda(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u);
#                         a.flow.σ |> Array;)

# custom_attrib = Dict(
#     "vort" => vort,
#     "Body" => _body,
#     "Lambda" => lamda
# )# this maps what to write to the name in the file
# # make the writer
# writer = vtkWriter("tail_biotsavart"; attrib=custom_attrib)

# # Loop in time
# for t in range(0,10;step=0.02)#1:6
#     while sim_time(sim)<t #sim_step!(sim,t)
#         sim_step!(sim;remeasure=true)
#     end
#     @show t
#     write!(writer,sim);
#     flush(stdout)
# end
# close(writer)
