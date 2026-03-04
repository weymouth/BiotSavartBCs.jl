# nohup julia --project=. kirigamiFalling.jl &> kirigami_out_010302026.log &
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
                  dims=(3N,3N,3N÷2),ϵ=T(1/2),half_thk=ϵ+1/T(√2),fall=false,dir=1)
    δR = R/rings; δH = R*H/rings^2; x₀ = max(R*(1-H)/2,δR+half_thk-min(0,R*H))+0.25R
    @inline mapped(f) = AutoBody(f,RigidMap(SA[x₀,dims[2]/2.f0,0],SA{T}[0,0,θ₀]))
    @inline ring(R₀,R₁,x₀,x₁,ϕ) = mapped() do (x,y,z),t
        r,θ = hypot(y,z),atan(z,y)
        δx = x₀+tanh(π*r/δR)*(x₁-x₀)*(1+cos(4θ+ϕ))/2
        hypot(x-δx,r-clamp(r,R₀+half_thk,R₁-half_thk))-half_thk
    end
    body = sum(i -> ring(δR*(i-1), δR*i, δH*(i-1)^2, δH*i^2, π*(i%2)), 1:rings)
    H == 0 && (body = ring(0,R,0,0,0))
    Ut = fall ? (0,0,0) : (i,x,t)->(i==dir ? U*Ux(a*U*t/2R) : zero(t)) # velocity BC
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

# measures the added-mass of the body, which is needed to update the acceleration in freefalling!
function compute_paramaters!(N,H,θ₀,ρ;R=2N/3,mem=CuArray,T=Float32)
    # longitudinal added mass
    sim = kirigami(N;R,T,mem,H=H,fall=false,θ₀=0.0,dims=(3N,3N,3N÷2),dir=1)
    sim_step!(sim;remeasure=false)
    m₁₁ = -2WaterLily.pressure_force(sim)[1]/(R+1/2+1/T(√2))^2
    # transverse added-mass
    sim = kirigami(N;R,T,mem,H=H,fall=false,θ₀=0.0,dims=(3N,3N,3N÷2),dir=2)
    sim_step!(sim;remeasure=false)
    m₂₂ = -2WaterLily.pressure_force(sim)[2]/(R+1/2+1/T(√2))^2
    # rotational added-mass
    sim = kirigami(N;R,T,mem,H=H,fall=true,θ₀=0.f0,dims=(3N,3N,3N÷2))
    # angular acceleration is not constant, so we just measure the moment at t=0 and 
    # assume it is all due to added mass (not exact but should be close for small θ₀)
    α=1.0; ω=α*sim.flow.Δt[end]; θ=θ₀+ω*sim.flow.Δt[end];
    sim.body = setmap(sim.body;θ=SA{Float32}[0,0,θ],ω=SA{Float32}[0,0,ω])
    sim_step!(sim;remeasure=true)
    Xₘ = H==0 ? sim.body.map.x₀ : sim.body.a.b.map.x₀
    m₆₆ = 2WaterLily.pressure_moment(Xₘ,sim)[3]/(R+1/2+1/T(√2))^2
    # measure mass and moment of inertia of the body itself
    apply!((x)->-x[1],sim.flow.p)
    m = -2WaterLily.pressure_force(sim)[1]
    I =  2WaterLily.pressure_moment(Xₘ,sim)[3]
    # update params
    return (m=ρ*m,                                # mass of body
            g=SA{Float32}[-U^2/R,0,0],            # gravity in lab frame
            mₐ=SA{Float32}[m₁₁*R^3, m₂₂*R^3, 0],  # added mass in body frame
            Iₘ=ρ*I*R,                             # moment of inertia of body
            Iₐ=m₆₆*R^2,                           # added moment of inertia
            θ=θ₀, ω=0.f0, α=0.f0)
end

#helper to rotate a vector
@inline @fastmath rotate(v,θ::T) where T = SA{T}[cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]*v

freefalling!(sim,times,state,Xₘ;R=sim.L,g=state.g,X₀=zero(g),vel=zero(g),acc=zero(g),
            θ=state.θ,ω=state.ω,α=state.α,m=state.m,Iₘ=state.Iₘ,Iₐ=state.Iₐ,
            mₐ=state.mₐ,save=false) = map(times) do t
    while sim_time(sim) < t
        # the step we are doing and the initial angle
        Δt = sim.flow.Δt[end]
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
    flush(stdout)
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

# free falling
using TypedTables,JLD2,Plots
N = 2^7; times = 0.2:0.2:20.0
θ₀=0.2f0; H=2.0; ρ=10.f0; R=2N/3.f0; U=1.f0 # only values H ∈ [0,1]

# single run
# compute real added mass and added-inertial for the body
params = compute_paramaters!(N,H,θ₀,ρ;R,mem=CuArray,T=Float32)
sim = kirigami(N;mem=CuArray,H=H,fall=true,θ₀);
Xₘ = sim.body.a.b.map.x₀ # moment point in lab frame
# writer = vtkWriter("kirigami_N$(N)_H$(H)_fall"; attrib=Dict("ω"=>vtk_ω,"λ₂"=>vtk_λ₂,"d"=>vtk_d))
data = freefalling!(sim,times,params,Xₘ;save=false)
# close(writer)

# domain sweep
θ₀ = 0.2f0; H = 1.f0
params = compute_paramaters!(N,H,θ₀,ρ;R,mem=CuArray,T=Float32)
for dims in ((3N,3N,3N÷2),(3N,3N,2N),(4N,3N,2N))
    @show dims, θ₀, H
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
for H in (0.25,0.5,1.0,2.0)
    # measure every time H changes
    params = compute_paramaters!(N,H,θ₀,ρ;R,mem=CuArray,T=Float32)
    for θ₀ in (0.4f0,0.2f0,0.f0)
        @show θ₀,H
        # set initial condition right
        params = Base.setindex(params, θ₀, :θ)
        sim = kirigami(N;mem=CuArray,H,fall=true,θ₀,dims=(6N,4N,3N÷2));
        Xₘ = sim.body.a.b.map.x₀ # moment point in lab frame
        measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
        flood(sim.flow.σ[2:end-1,2:end-1,2],clims=(-1,1))
        savefig("kirigami_N$(N)_H$(H)_θ$(θ₀)_initial.png")
        data = freefalling!(sim,times,params,Xₘ)
        save_object("kirigami_N$(N)_H$(H)_θ$(θ₀)_fall.jld2",data)
        flood(sim.flow.u[2:end-1,2:end-1,2,1])
        scatter!([Xₘ[1]],[Xₘ[2]],markersize=5,color=:red,label=:none)
        savefig("kirigami_N$(N)_H$(H)_θ$(θ₀)_final.png")
    end
end