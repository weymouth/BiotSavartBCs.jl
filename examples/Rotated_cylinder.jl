using BiotSavartBCs
using WaterLily
using StaticArrays
using CUDA
using JLD2
include("Diagnostics.jl")
include("TwoD_plots.jl")
circ(D,n,m;Re=200,U=(1/√2,1/√2),mem=Array) = Simulation((n*D,m*D), U, D; 
                                                        body=AutoBody((x,t)->√sum(abs2,x.-2.2*D)-D÷2),
                                                        ν=√sum(abs2,U)*D/Re,mem)
norm(x) = √sum(abs2,x)
use_biotsavart = true
D,n,m = (64,7,7)
sim = circ(D,n,m;mem=CUDA.CuArray)
ω = use_biotsavart ? MLArray(sim.flow.σ) : nothing
t_end=100
u = Float32[]; I = CartesianIndex(D+m*D÷2,m*D÷2); forces = []
use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
@assert !all(sim.pois.n .== 32) "pressure problem"
jldopen("rotated_cylinder.jld2", "w") do file
    mygroup = JLD2.Group(file,"case1")
    mygroup["θ"] = (4,4,true)
    while sim_time(sim)<t_end
        use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
        pres,visc = diagnostics(sim;κ=2/D); 
        push!(forces,[sim_time(sim),pres...,visc...])
        sim_time(sim)%0.1<sim.flow.Δt[end]/sim.L && push!(u,norm(sim.flow.u[I,:]))
        sim_time(sim)%1<sim.flow.Δt[end]/sim.L && @show sim_time(sim)
    end
    mygroup["u"] = u
    mygroup["p"] = sim.flow.p
    mygroup["f"] = forces
    @inside sim.flow.σ[I] = BiotSavartBCs.centered_curl(3,I,sim.flow.u)*sim.L/sim.U
    mygroup["ω"] = sim.flow.σ
    # flow
    flood(sim.flow.σ|>Array,clims=(-10,10))
    savefig("flow_rotated_cylinder.png")
end
