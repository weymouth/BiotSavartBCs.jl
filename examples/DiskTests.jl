using WaterLily,StaticArrays,CUDA,BiotSavartBCs
using JLD2,Plots

function make_sim_acc(dims; N=128, R=32, a=0.5, U=1, Re=1e3, mem=Array, use_biotsavart=false)
    disk3D(x,t) = (z=x-SA[-R,0,0].-N/2; y=z.-SA[0,clamp(z[2],-R,R),clamp(z[3],-R,R)]; √sum(abs2,y)-1.5)
    disk2D(x,t) = (z=x-SA[-R,0].-N/2; y=z.-SA[0,clamp(z[2],-R,R)]; √sum(abs2,y)-1.5)
    Ut(i,t::T) where T = i==1 ? convert(T,min(a*t/R,U)) : zero(T) # velocity BC
    body = length(dims)==2 ? AutoBody(disk2D) : AutoBody(disk3D)
    use_biotsavart && return BiotSimulation(dims, Ut, R; U, ν=U*R/Re, body, mem)
    Simulation(dims, Ut, R; U, ν=U*R/Re, body, mem)
end
slice(a::AbstractArray{T,3}) where T = Array(dropdims(a,dims=3))
slice(a::AbstractArray{T,2}) where T = Array(a)

# size of the domain
N = 2^6; R = N/3
for Dim ∈ [2,3], use_biotsavart ∈ [true,false]
    sim = make_sim_acc(ntuple(i->N,Dim);N,R,mem=CUDA.CuArray,use_biotsavart);
    Rslice = Dim==2 ? inside(sim.flow.p) : CartesianIndices((2:N+1,2:N+1,N÷2+1:N÷2+1))
    forces = []; σ = []; p = [];
    for t in 1:6
        @time while sim_time(sim)<t
            sim_step!(sim;remeasure=false)
            f = 2WaterLily.pressure_force(sim)/R^(Dim-1)
            push!(forces,[sim_time(sim),f[1]])
        end
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        push!(p,slice(sim.flow.p[Rslice]))
        push!(σ,slice(sim.flow.σ[Rslice]))
    end
    jldsave("disk_$(Dim)D_$(N)D_$(use_biotsavart).jld2"; f=forces, σ=σ, p=p)
end
# make the figures
for Dim ∈ [2,3], use_biotsavart ∈ [true,false]
    BCs = use_biotsavart ? "biot" : "reflect"
    jldopen("disk_$(Dim)D_$(N)D_$(use_biotsavart).jld2") do file
        for t ∈ 1:6
            p = file["p"][t]; σ = file["σ"][t]
            flood(p,clims=(-2,2),cfill=:viridis)
            savefig("Disk_$(Dim)D_"*BCs*"_press_$(t).png")
            flood(σ,clims=(-20,20))
            savefig("Disk_$(Dim)D_"*BCs*"_omega_$(t).png")
            flood(σ,clims=(-20,20))
            contour!(clamp.(p',-2,2),levels=range(-2,2,length=10),color=:black,
                     linewidth=0.5,legend=false)
            savefig("Disk_$(Dim)D_"*BCs*"_press_omega_$(t).png")
        end
    end
end