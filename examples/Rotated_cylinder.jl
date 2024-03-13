using BiotSavartBCs
using WaterLily
using StaticArrays
using CUDA
using JLD2
include("Diagnostics.jl")
circ(D,n,m;Re=550,U=(1/√2,1/√2),mem=Array) = Simulation((n*D,m*D), U, D; 
                                                        body=AutoBody((x,t)->√sum(abs2,x.-m*D÷4)-D÷2),
                                                        ν=√sum(abs2,U)*D/Re,mem)
norm(x) = √sum(abs2,x)
use_biotsavart = true
D,n,m = (64,4,4)
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
end

using Plots
include("/home/marinlauber/Workspace/BiotSavartBCs.jl/examples/TwoD_plots.jl")

# velocity
plt = plot(dpi=300);
u = Float32.(u)
plot!(plt,range(0,10,length=100),u[1:100],label="$(n)Dx$(m)D using "*BC;ls)
title!("Domain size study");xlabel!("Convective time");ylabel!("v/U near centerline")
savefig("start_rotated_cylinder.png")
#fft
plt = plot(dpi=300);
u = u[700:end]
u_hat=fft(u)./0.5length(u)
plot!(plt,range(0,2.5,length=50),abs.(u_hat[1:50]),label="$(n)Dx$(m)D using Biot-Savart";ls)
title!("Domain size study");xlabel!("Strouhal");ylabel!("PSD(v) near centerline")
savefig("fft_rotated_cylinder.png")
# force
f = reduce(vcat,forces')
force = .√((f[:,2].+f[:,4]).^2 .+ (f[:,3].+f[:,5]).^2)
plot(f[:,1],force/32,dpi=300); xlims!(0,10); ylims!(0,3.5)
savefig("force_rotated_cylinder.png")
# flow
flood(sim.flow.σ|>Array,clims=(-10,10))
savefig("flow_rotated_cylinder.png")

# include("TwoD_plots.jl")
# t₀,duration,tstep = round(sim_time(sim)), 100., 0.1
# R = inside(sim.flow.p)
# @time @gif for tᵢ in range(t₀,t₀+duration,step=tstep)
#     while sim_time(sim) < tᵢ
#         use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
#     end
#     println("tU/L=",round(sim_time(sim),digits=4),
#             ", Δt=",round(sim.flow.Δt[end],digits=3))
#     @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
#     flood(sim.flow.σ[R]|>Array,clims=(-10,10))
# end
