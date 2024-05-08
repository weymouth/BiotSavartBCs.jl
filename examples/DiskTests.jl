using WaterLily,StaticArrays,CUDA,BiotSavartBCs

function make_sim_acc(; N=128, R=32, a=0.5, U=1, Re=1e3, mem=Array)
    disk(x,t) = (y=x-SA[-R,0,0].-N/2; r=√(y[2]^2+y[3]^2); √(y[1]^2+(r-min(r,R))^2)-1.5)
    Ut(i,t::T) where T = i==1 ? convert(T,a*t/R+(1.0+tanh(31.4*(t/R-1.0/a)))/2.0*(1-a*t/R)) : zero(T) # velocity BC
    Simulation((N,N,N), Ut, R; U,ν=U*R/Re, body=AutoBody(disk), mem)
end

include("TwoD_plots.jl")
using JLD2
CIs = CartesianIndices
N = 2^8; R = N/3
domain = (2:N+1,2:N+1,N÷2+1)
for use_biotsavart in [true]
    sim = make_sim_acc(mem=CUDA.CuArray;N,R);
    ω = use_biotsavart ? ntuple(i->MLArray(sim.flow.σ),3) : nothing
    forces = []; k=0; σ = []; p = [];
    @show sizeof(sim)
    for t in 1:20
        @time while sim_time(sim)<t #sim_step!(sim,t)
            measure!(sim,sum(sim.flow.Δt)) # update the body compute at timeNext
            use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
            f = -2WaterLily.∮nds(sim.flow.p,sim.flow.f,sim.body,sum(sim.flow.Δt[1:end-1]))/R^2
            push!(forces,[sim_time(sim),f[1]])
        end
        WaterLily.@loop sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U over I ∈ CIs(domain)
        push!(p,sim.flow.p[CIs(domain[1:2]),domain[3]]|>Array)
        push!(σ,sim.flow.σ[CIs(domain[1:2]),domain[3]]|>Array)
    end
    jldopen("disk_$(N)D_$(R)R_$(use_biotsavart).jld2", "w") do file
        mygroup = JLD2.Group(file,"case")
        mygroup["forces"] = forces
        mygroup["σ"] = σ
        mygroup["p"] = p
    end
end
for use_biotsavart in [true]
    BCs = use_biotsavart ? "biot" : "reflect"
    jldopen("disk_$(N)D_$(R)R_$(use_biotsavart).jld2","r") do file
        for t ∈ 1:20
            p = file["case"]["p"][t]
            σ = file["case"]["σ"][t]
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
