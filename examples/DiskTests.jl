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


# # read results for post processing
a_star = 0.5; Ca = 1.0/3.0
biot = jldopen("disk_$(N)D_$(R)R_$(true).jld2","r")
# ref = jldopen("disk_$(N)D_$(R)R_$(false).jld2","r")
forces_biot = reduce(vcat,biot["case"]["forces"]')
# forces_ref = reduce(vcat,ref["case"]["forces"]')
plot(forces_biot[:,1],forces_biot[:,2],label="Biot-Savart")
# plot!(forces_ref[:,1],forces_ref[:,2]*(Ca*a_star),label="Reflection")

# tₐ = 2.0
# H(x::AbstractFloat) = ifelse(x < 0, zero(x), ifelse(x > 0, one(x), oftype(x,0.5)))
# Dadiff(t,tₐ=0) = Ca * a_star*(√ν)*(√t - H(t-tₐ)√H(t-tₐ))

# # plot!([0,3],[Ca,Ca],label=:none,ls=:dot,color=:black)
# xlims!(0,3); ylims!(0,4)
# xlabel!("Convective time"); ylabel!("( F/ρU²R² )/ Ca")
# savefig("Disk_$(N)D_force.png")
