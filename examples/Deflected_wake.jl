using BiotSavartBCs
using WaterLily
using StaticArrays
using CUDA
using JLD2
include("TwoD_plots.jl")
include("Diagnostics.jl")

function ellipse(D,n,m,Λ=5.0;A₀=1.0,St=0.6,U=1,Re=100,T=Float32,mem=Array)
    h₀=T(A₀*D/2); ω=T(2π*St*U/D)
    function sdf(x,t)
        √sum(abs2,SA[x[1]/Λ,x[2]])-D÷2/Λ
    end
    function map(x,t)
        x .- SA[n*D÷4,m*D÷2-h₀*sin(ω*t)]
    end
    Simulation((n*D,m*D), (U,0), D; body=AutoBody(sdf,map), ν=U*D/Re, T, mem)
end
# do we use Biot-Savart?
# for use_biotsavart in [true]
#     (n,m) = use_biotsavart ? (6,4) : (32,24)
#     sim = use_biotsavart ? ellipse(64,6,4,5;mem=CUDA.CuArray) : ellipse(64,30,20,5;mem=CUDA.CuArray); 
#     ω = use_biotsavart ? MLArray(sim.flow.σ) : nothing
#     t₀,duration,tstep = round(sim_time(sim)), 100., 0.1
#     R = use_biotsavart ? inside(sim.flow.p) : CartesianIndices((386:769, 514:769)) # show the same part
#     forces = []
#     anim = @animate for tᵢ in range(t₀,t₀+duration,step=tstep)
#         while sim_time(sim) < tᵢ
#             measure!(sim)
#             use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : mom_step!(sim.flow,sim.pois)
#             pres,visc = diagnostics(sim);
#             push!(forces,[sim_time(sim),pres...,visc...])
#         end
#         println("tU/L=",round(sim_time(sim),digits=4),
#         ", Δt=",round(sim.flow.Δt[end],digits=3))
#         @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
#         flood(sim.flow.σ[R],clims=(-10,10))
#     end
#     gif(anim, "deflected_wake_$(n)L_$(m)L_$use_biotsavart.gif")
#     jldopen("deflected_wake_$use_biotsavart.jld2","w") do file
#         file["force"] = forces
#     end
# end
using Plots
plt = plot(dpi=300)
for use_biotsavart ∈ [true false]
    jldopen("deflected_wake_$use_biotsavart.jld2","r") do file
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        forces = reduce(vcat,file["force"]'); St=0.6; L=64
        plot!(plt,forces[:,1]*St,sum(forces[:,[3,5]];dims=2)./L^2,label="Airfoil St:$St "*BC)
        L_mean = sum(forces[end-20000:end,[3,5]])./20000L^2
        plot!([0,maximum(forces[:,1]*St)],[L_mean,L_mean],color=:black,label="Mean force "*BC;ls)
    end
end
xlims!(0,60);
ylims!(-0.5,0.5);
title!("Deflected Wake Lift Forces");xlabel!("Convective time");ylabel!("2Force/ρUR")
savefig("force_deflected_wake.png")

# Lyapunov exponenent
using Interpolations
plt = plot(dpi=300)
file = jldopen("deflected_wake_true.jld2","r")
forces_BS = reduce(vcat,file["force"]'); close(file)
file = jldopen("deflected_wake_false.jld2","r")
forces_R  = reduce(vcat,file["force"]'); close(file)
St=0.6; L=64
BS_interp = linear_interpolation(forces_BS[:,1], reduce(vcat,sum(forces_BS[:,[3,5]];dims=2)./L))
R_interp = linear_interpolation(forces_R[:,1], reduce(vcat,sum(forces_R[:,[3,5]];dims=2)./L))
δF = abs.(BS_interp.(0.01:0.01:5) .- R_interp.(0.01:0.01:5))
plot!(plt,collect(0.01:0.01:5).*St,δF,yaxis=:log,label=:none)
title!("Lyapunov Exponent");xlabel!("Convective time");ylabel!("Lift Force")
savefig("Lyapunov_deflected_wake.png")

# Poincaré map
plt = plot(dpi=300)
for use_biotsavart ∈ [true false]
    jldopen("deflected_wake_$use_biotsavart.jld2","r") do file
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        forces = reduce(vcat,file["force"]'); St=0.6; L=64
        plot!(plt,sum(forces[:,[3,5]];dims=2)./L,sum(forces[:,[2,4]];dims=2)./L,label="Airfoil St:$St "*BC,lw=0.2)
    end
end
xlims!(-15,15);ylims!(-1.5,1.5);
title!("Poincaré map");ylabel!("Drag Force");xlabel!("Lift Force")
savefig("poincare_deflected_wake.png")
