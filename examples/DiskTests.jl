using WaterLily,StaticArrays,CUDA,BiotSavartBCs
using JLD2,Plots

function make_sim_acc(; N=128, R=32, a=0.5, U=1, Re=1e3, mem=Array, use_biotsavart=false)
    disk(x,t) = (y=x-SA[-R,0,0].-N/2; r=√(y[2]^2+y[3]^2); √(y[1]^2+(r-min(r,R))^2)-1.5)
    Ut(i,t::T) where T = i==1 ? convert(T,a*t/R) : zero(T) # velocity BC
    body = AutoBody(disk)
    use_biotsavart && return BiotSimulation((N,N,N), Ut, R; U,ν=U*R/Re, body, mem)
    Simulation((N,N,N), Ut, R; U,ν=U*R/Re, body, mem)
end

# size of the domain
N = 2^7; R = N/3
for use_biotsavart in [false,true]
    sim = make_sim_acc(mem=CUDA.CuArray;N,R,use_biotsavart);
    forces = []; σ = []; p = [];
    for t in 1:20
        @time while sim_time(sim)<t
            sim_step!(sim;remeasure=false)
            f = 2WaterLily.pressure_force(sim)/R^2
            push!(forces,[sim_time(sim),f[1]])
        end
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        push!(p,sim.flow.p[inside(sim.flow.p)]|>Array)
        push!(σ,sim.flow.σ[inside(sim.flow.p)]|>Array)
    end
    jldsave("disk_$(N)D_$(R)R_$(use_biotsavart).jld2"; f=forces, σ=σ, p=p)
end
# make the figures
for use_biotsavart in [false,true]
    BCs = use_biotsavart ? "biot" : "reflect"
    jldopen("disk_$(N)D_$(R)R_$(use_biotsavart).jld2") do file
        for t ∈ 1:20
            p = file["p"][t]; σ = file["σ"][t]
            flood(p,clims=(-2,2),cfill=:viridis)
            savefig("Disk_"*BCs*"_press_$(t).png")
            flood(σ,clims=(-20,20))
            savefig("Disk_"*BCs*"_omega_$(t).png")
            flood(σ,clims=(-20,20))
            contour!(clamp.(p',-2,2),levels=range(-2,2,length=10),color=:black,
                     linewidth=0.5,legend=false)
            savefig("Disk_"*BCs*"_press_omega_$(t).png")
        end
    end
end
