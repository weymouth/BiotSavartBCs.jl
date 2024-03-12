using WaterLily,StaticArrays,CUDA,BiotSavartBCs

function make_sim(; N=128, R=32, a=0.5, U=1, Re=1e3, mem=Array)
    disk(x,t) = (r = √(x[2]^2+x[3]^2); √(x[1]^2+(r-min(r,R))^2)-1.5)
    s(t) = ifelse(t<U/a,0.5a*t^2,U*(t-0.5U/a)) # displacement
    move(x,t) = x - SA[R*s(t/R)-R,0,0] .- N/2  # move the center
    Simulation((N,N,N), (0,0,0), R; U,ν=U*R/Re, body=AutoBody(disk,move), mem)
end
function make_sim_square(; N=128, R=32, a=0.5, U=1, Re=1e3, mem=Array)
    square(x,t) = (y = x.-SA[0,clamp(x[2],-R,R),clamp(x[3],-R,R)]; √sum(abs2,y)-1.5)
    s(t) = ifelse(t<U/a,0.5a*t^2,U*(t-0.5U/a)) # displacement
    move(x,t) = x - SA[R*s(t/R)-R,0,0] .- N/2  # move the center
    Simulation((N,N,N), (0,0,0), R; U,ν=U*R/Re, body=AutoBody(square,move), mem)
end
function make_sim_acc(; N=128, R=32, a=0.5, U=1, Re=1e3, mem=Array)
    disk(x,t) = (y=x-SA[-R,0,0].-N/2; r=√(y[2]^2+y[3]^2); √(y[1]^2+(r-min(r,R))^2)-1.5)
    Ut(i,t::T) where T = i==1 ? convert(T,a*t/R+(1.0+tanh(31.4*(t/R-1.0/a)))/2.0*(1-a*t/R)) : zero(T) # velocity BC
    Simulation((N,N,N), Ut, R; U,ν=U*R/Re, body=AutoBody(disk), mem)
end

include("TwoD_plots.jl")
include("Diagnostics.jl")
CIs = CartesianIndices
N = 2^8; R = N/3
domain = (2:N+1,2:N+1,N÷2+1)
sim = make_sim_acc(mem=CUDA.CuArray;N,R);
ω = ntuple(i->MLArray(sim.flow.σ),3);
use_biotsavart = true; forces = [];
# global k=0
for t in 1:3
    @time while sim_time(sim)<t #sim_step!(sim,t)
        measure!(sim,sum(sim.flow.Δt)) # update the body compute at timeNext
        use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
        # f = -2WaterLily.∮nds(sim.flow.p,sim.flow.f,sim.body,sum(sim.flow.Δt[1:end-1]))/sim.L^2
        # push!(forces,f[1])
        pres,visc = diagnostics(sim); 
        push!(forces,[sim_time(sim),pres...,visc...])
        # flood(sim.flow.p[CIs(domain[1:2]),domain[3]]|>Array,clims=(-2,2))
        # savefig("press_$(k).png")
        # global k+=1
    end
    BCs = use_biotsavart ? "biot" : "reflect"
    flood(sim.flow.p[CIs(domain[1:2]),domain[3]]|>Array,clims=(-2,2))
    savefig("Disk_"*BCs*"_press_$(t).png")
    WaterLily.@loop sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U over I ∈ CIs(domain)
    flood(sim.flow.σ[CIs(domain[1:2]),domain[3]]|>Array,clims=(-20,20))
    savefig("Disk_"*BCs*"_omega_$(t).png")
end
using JLD2
jldopen("disk_$(N)D_$(R)R_$(use_biotsavart).jld2", "w") do file
    mygroup = JLD2.Group(file,"case")
    mygroup["forces"] = forces
end
biot = jldopen("disk_$(N)D_$(R)R_$(true).jld2","r")
ref = jldopen("disk_$(N)D_$(R)R_$(false).jld2","r")
forces_biot = reduce(vcat,biot["case"]["forces"]')
forces_ref = reduce(vcat,ref["case"]["forces"]')
plot(forces_biot[:,1],forces_biot[:,2]/(π*R^2),label="Biot-Savart")
plot!(forces_ref[:,1],forces_ref[:,2]/(π*R^2),label="Reflection")
xlims!(0,3); ylims!(0,4.0)
xlabel!("Convective time"); ylabel!("Force/πR²")
savefig("Disk_$(N)D_$(R)R_force.png")  