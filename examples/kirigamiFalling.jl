using WaterLily,BiotSavartBCs,CUDA,StaticArrays

# Biot-Savart momentum step with U and acceleration prescribed
import WaterLily: scale_u!,conv_diff!,udf!,BDIM!,CFL
import BiotSavartBCs: biot_project!
function biot_mom_step_fall!(sim::BiotSimulation;udf=nothing,U,kwargs...)
    a=sim.flow; b=sim.pois; ω=(sim.ω,sim.x₀,sim.tar,sim.ftar)
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    conv_diff!(a.f,a.u⁰,a.σ,quick,ν=a.ν)
    udf!(a,udf,t₀; kwargs...)
    BDIM!(a);
    biot_project!(a,b,ω...,U;sim.fmm)
    # corrector u → u¹
    conv_diff!(a.f,a.u,a.σ,quick,ν=a.ν)
    udf!(a,udf,t₁; kwargs...)
    BDIM!(a); scale_u!(a,0.5)
    biot_project!(a,b,ω...,U;sim.fmm,w=0.5)
    push!(a.Δt,CFL(a))
end

import WaterLily: @loop
# falling body acceleration term
fall!(flow,t;acceleration) = for i ∈ 1:ndims(flow.p)
    @loop flow.f[I,i] += acceleration[i] over I ∈ CartesianIndices(flow.p)
end

WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1) # good idea when accelerating from rest
linear(t)=min(t,one(t))
function kirigami(N;H=0,rings=16,U=1,a=1,Re=1e4,mem=Array,T=Float32,Ux=linear,R=T(2N/3),θ₀=0.f0,
                  dims=(3N,3N,3N÷2),ϵ=T(1/2),half_thk=ϵ+1/T(√2),fall=false)
    δR = R/rings; δH = R*H/rings^2; x₀ = max(R*(1-H)/2,δR+half_thk-min(0,R*H))+0.25R
    @inline mapped(f) = AutoBody(f,RigidMap(SA[x₀,dims[2]/2.f0,0],SA{T}[0,0,θ₀]))
    @show SA[x₀,N,0]
    @inline ring(R₀,R₁,x₀,x₁,ϕ) = mapped() do (x,y,z),t
        r,θ = hypot(y,z),atan(z,y)
        δx = x₀+tanh(π*r/δR)*(x₁-x₀)*(1+cos(4θ+ϕ))/2
        hypot(x-δx,r-clamp(r,R₀+half_thk,R₁-half_thk))-half_thk
    end
    body = sum(i -> ring(δR*(i-1), δR*i, δH*(i-1)^2, δH*i^2, π*(i%2)), 1:rings)
    H == 0 && (body = ring(0,R,0,0,0))
    Ut = fall ? (0,0,0) : (i,x,t)->(i==1 ? U*Ux(a*U*t/2R) : zero(t)) # velocity BC
    BiotSimulation(dims,Ut,R;U,ν=U*2R/Re,body,mem,T,ϵ,nonbiotfaces=(-3))
end

import BiotSavartBCs: interaction,symmetry,image
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₃,sgn₃ = image(T,size(ω),-3)  # image target and sign in z
    # Add up the four contributions
    return interaction(ω,T,args...)+sgn₃*interaction(ω,T₃,args...)
end

drag!(sim,times,R=sim.L,x₀=SA[R,0,0];remeasure=false) = map(times) do t
    @show t; flush(stdout)
    sim_step!(sim,t;remeasure)
    Cd,Cl = -8WaterLily.total_force(sim)[1:2]/R^2
    Cm = 8WaterLily.pressure_moment(x₀,sim)[3]/R^3
    (;t,Cd,Cl,Cm)
end |> Table

#helper to rotate a vector
@inline @fastmath rotate(v,θ::T) where T = SA{T}[cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]*v

freefalling!(sim,times,state,Xₘ;R=sim.L,g=state.g,X₀=zero(g),vel=zero(g),acc=zero(g),
            θ=state.θ,ω=state.ω,α=state.α,m=state.m,Iₘ=state.Iₘ,Iₐ=state.Iₐ,
            mₐ=state.mₐ,save=false) = map(times) do t
    while sim_time(sim) < t
        # the step we are doing and the initial angle
        Δt,θ = sim.flow.Δt[end],sim.body.a.b.map.θ[3]
        # compute pressure force and moment in lab frame
        force = -WaterLily.total_force(sim)
        moment = -WaterLily.pressure_moment(Xₘ,sim)[3]
        # transform to body frame
        force,acc = rotate(force+m.*g, -θ),rotate(acc, -θ)
        # update linear motion in body frame, and then back to lab frame
        acc = rotate((force - mₐ.*acc)./(m .+ mₐ), θ).*SA{Float32}[1,1,0]
        vel += Δt*acc; X₀ += Δt*vel
        # update rotation ODE
        α = (moment - α*Iₐ)/(Iₘ + Iₐ)
        ω += Δt*α; θ += Δt*ω # Verlet
        # remeasure the sim
        sim.body = setmap(sim.body;θ=SA{Float32}[0,0,θ],ω=SA{Float32}[0,0,ω]) # update rotational variables
        measure!(sim)
        biot_mom_step_fall!(sim;udf=fall!,acceleration=-acc,U=-vel)
    end
    save && save!(writer,sim)
    println("tU/L=",round(t,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3),
            " X₁=", round(X₀[1]/sim.L,digits=3), " θ=", round(rad2deg(θ),digits=3),
            "° u₁=", round(vel[1]/sim.U,digits=3), " a₁=", round(acc[1]/(sim.U^2/sim.L),digits=3))
    Cd,Cl = -4WaterLily.total_force(sim)[1:2]/R^2
    Cm = 4WaterLily.pressure_moment(Xₘ,sim)[3]/R^3
    (;t,Cd,Cl,Cm,u₁=vel[1],u₂=vel[2],a₁=acc[1],a₂=acc[2],θ,ω,α)
end |> Table

# make a writer with some attributes, need to output to CPU array to save file (|> Array)
using WriteVTK
import WaterLily: @loop,ω,λ₂
vtk_ω(a::AbstractSimulation) = (@loop a.flow.f[I,:] .= ω(I,a.flow.u) over I in inside(a.flow.p); a.flow.f |> Array)
vtk_d(a::AbstractSimulation) = (measure_sdf!(a.flow.σ,a.body,WaterLily.time(a)); a.flow.σ |> Array)
vtk_λ₂(a::AbstractSimulation) = (@inside a.flow.σ[I] = λ₂(I,a.flow.u); a.flow.σ |> Array)

# Dynamic opening
using TypedTables,JLD2,Plots
N = 2^8; times = 0.2:0.2:20.0
θ₀=0.4f0; H=1.0; ρ=10.f0; R=2N/3.f0; U=1.f0 # only values H ∈ [0,1]
sim = kirigami(N;mem=CuArray,H=2,fall=true,θ₀);

# all quantities for 1/2 of the disk, assumes thickness of disk is 3 for mass, ρ is density ratios
# m=3πρR² m11 = 8/3R³, m22=m11/3?, Im = 3πρR⁴/4, Ia = 16/45πR⁵
params = (m=3π*ρ*R^2/2,                              # mass of body
          g=SA{Float32}[-U^2/R,0,0],                 # gravity in lab frame
          mₐ=SA{Float32}[4/3.f0*R^3, 1/3.f0*R^3, 0], # added mass in body frame
          Iₘ=ρ*3.f0*π*R^4/8.0f0,                     # moment of inertia of body
          Iₐ=(8/45.f0)*π*R^5,                        # added moment of inertia
          θ=θ₀,ω=0.f0,α=0.f0) 
Xₘ = sim.body.a.b.map.x₀+sim.body.a.b.map.xₚ # moment point in lab frame

# single run
writer = vtkWriter("kirigami_N$(N)_H$(H)_fall"; attrib=Dict("ω"=>vtk_ω,"λ₂"=>vtk_λ₂,"d"=>vtk_d))
data = freefalling!(sim,times,params,Xₘ;save=true)
close(writer)

# flood(sim.flow.μ₀[2:end-1,2:end-1,2,1])
flood(sim.flow.u[2:end-1,2:end-1,2,1])
scatter!([sim.body.a.b.map.x₀[1]+sim.body.a.b.map.xₚ[1]],[sim.body.a.b.map.x₀[2]+sim.body.a.b.map.xₚ[2]],
          markersize=5,color=:red,label=:none)

begin
    p1=plot(data.t,data.Cd,label="Cd",xlim=extrema(times),ylims=(-1,Inf),lw=2)
    plot!(p1,data.t,data.Cm,label="Cm",ylabel="Cd,Cm",lw=2)
    p2=plot(data.t,data.u₁,label="u₁",xlabel="time",xlim=extrema(times),lw=2)
    plot!(p2,data.t,data.u₂,label="u₂",xlabel="time",lw=2)
    plot!(p2,data.t,data.θ,label="θ",ls=:dash,ylabel="u₁,u₂,θ",lw=2)
    plot(p1,p2,layout=(2,1),size=(600,600))
end

# domain sweep
θ₀ = 0.2f0; H = 1.f0
for dims in ((3N,2N,N),(3N,N,N),(6N,N,N))
    @show dims, θ₀, H
    u₀ = zeros(12); u₀[7] = θ₀ # initial rotation
    sim = kirigami(N;mem=CuArray,H,fall=true,θ₀,dims=dims);
    measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    flood(sim.flow.σ[2:end-1,2:end-1,2],clims=(-1,1)); savefig("kirigami_N$(N)_$(dims[1])x$(dims[2])x$(dims[3])_initial.png")
    data = freefalling!(sim,times,params,Xₘ)
    save_object("kirigami_N$(N)_$(dims[1])x$(dims[2])x$(dims[3])_fall.jld2",data)
    flood(sim.flow.u[2:end-1,2:end-1,2,1])
    scatter!([Xₘ[1]],[Xₘ[2]],markersize=5,color=:red,label=:none)
    savefig("kirigami_N$(N)_$(dims[1])x$(dims[2])x$(dims[3])_final.png")
end

# theta and H sweep
for θ₀ in (0.4f0,0.2f0,0.f0), H in (0.5,1.0,2.0,4.f0)
    @show θ₀,H
    u₀ = zeros(12); u₀[7] = θ₀ # initial rotation
    sim = kirigami(N;mem=CuArray,H,fall=true,θ₀);
    measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    flood(sim.flow.σ[2:end-1,2:end-1,2],clims=(-1,1)); savefig("kirigami_N$(N)_H$(H)_θ$(θ₀)_initial.png")
    data = freefalling!(sim,times,params,Xₘ)
    save_object("kirigami_N$(N)_H$(H)_θ$(θ₀)_fall.jld2",data)
    flood(sim.flow.u[2:end-1,2:end-1,2,1])
    scatter!([Xₘ[1]],[Xₘ[2]],markersize=5,color=:red,label=:none)
    savefig("kirigami_N$(N)_H$(H)_θ$(θ₀)_final.png")
end