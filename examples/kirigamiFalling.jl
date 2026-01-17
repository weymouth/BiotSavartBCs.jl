using WaterLily,BiotSavartBCs,CUDA,StaticArrays,OrdinaryDiffEq

# Biot-Savart momentum step with U and acceleration prescribed
import WaterLily: scale_u!,conv_diff!,udf!,BDIM!,CFL
import BiotSavartBCs: biot_project!
function biot_mom_step_fall!(a::Flow{N},b,ω...;λ=quick,udf=nothing,fmm=true,U,kwargs...) where N
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    @log "p"
    conv_diff!(a.f,a.u⁰,a.σ,λ,ν=a.ν)
    udf!(a,udf,t₀; kwargs...)
    BDIM!(a);
    biot_project!(a,b,ω...,U;fmm)
    # corrector u → u¹
    @log "c"
    conv_diff!(a.f,a.u,a.σ,λ,ν=a.ν)
    udf!(a,udf,t₁; kwargs...)
    BDIM!(a); scale_u!(a,0.5)
    biot_project!(a,b,ω...,U;fmm,w=0.5)
    push!(a.Δt,CFL(a))
end

import WaterLily: @loop
# falling body acceleration term
fall!(flow,t;acceleration) = for i ∈ 1:ndims(flow.p)
    @loop flow.f[I,i] += acceleration[i] over I ∈ CartesianIndices(flow.p)
end

# ODE function for falling body under gravity
function gravity!(du,u,p,t)
    # unpack the state
    x₁,u₁,a₁,x₂,u₂,a₂,θ,ω,α,F₁,F₂,M₃ = u
    # unpack constant params
    m,m₁₁,m₂₂,Iₘ,Iₐ,g = p
    # rotate gravity into body frame
    g₁,g₂ = g*cos(θ), g*sin(θ)
    # rates (du[3,6,9,10,11,12] are unused)
    du[1] = u₁
    u[3] = du[2] = (F₁ - m₁₁*a₁ + m*g₁)/(m + m₁₁)
    du[4] = u₂
    u[6] = du[5] = (F₂ - m₂₂*a₂ + m*g₂)/(m + m₂₂)
    du[7] = ω
    u[9] = du[8] = (M₃ - Iₐ*α)/(Iₐ + Iₘ)
end

WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1) # good idea when accelerating from rest
linear(t)=min(t,one(t))
function kirigami(N;H=0,rings=16,U=1,a=1,Re=1e4,mem=Array,T=Float32,Ux=linear,R=T(2N/3),θ₀=0.f0,ϵ=T(1/2),half_thk=ϵ+1/T(√2),fall=false)
    δR = R/rings; δH = R*H/rings^2; x₀ = max(R*(1-H)/2,δR+half_thk-min(0,R*H))+0.25R
    @inline mapped(f) = AutoBody(f,RigidMap(SA[x₀,3N/2,0],SA{T}[0,0,θ₀]))
    @show SA[x₀,3N/2,0]
    @inline ring(R₀,R₁,x₀,x₁,ϕ) = mapped() do (x,y,z),t
        r,θ = hypot(y,z),atan(z,y)
        δx = x₀+tanh(π*r/δR)*(x₁-x₀)*(1+cos(4θ+ϕ))/2
        hypot(x-δx,r-clamp(r,R₀+half_thk,R₁-half_thk))-half_thk
    end
    body = sum(i -> ring(δR*(i-1), δR*i, δH*(i-1)^2, δH*i^2, π*(i%2)), 1:rings)
    H == 0 && (body = ring(0,R,0,0,0))
    Ut = fall ? (0,0,0) : (i,x,t)->(i==1 ? U*Ux(a*U*t/2R) : zero(t)) # velocity BC
    BiotSimulation((3N,3N,N),Ut,R;U,ν=U*2R/Re,body,mem,T,ϵ,nonbiotfaces=(-2,-3))
end

import BiotSavartBCs: interaction,symmetry,image
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₃,sgn₃ = image(T,size(ω),-3)  # image target and sign in z
    # Add up the four contributions
    return interaction(ω,T,args...)+sgn₃*interaction(ω,T₃,args...)
end

# helper to rotate forces/moments to body frame
@inline rot(α) = SA{Float32}[cos(α) -sin(α) 0; sin(α) cos(α) 0; 0 0 1]

freefalling!(sim,times,gravity,R=sim.L,x₀=SA[R,0,0];x₁=0.f0,u₁=0.f0,a₁=0.f0,x₂=0.f0,u₂=0.f0,a₂=0.f0,
             θ=sim.body.a.b.map.θ[3],ω=0.f0,α=0.f0,remeasure=false) = map(times) do t
    @show t; flush(stdout)
    while sim_time(sim) < t
        # compute pressure force
        force = -WaterLily.total_force(sim)
        moment = -WaterLily.pressure_moment(sim.body.a.b.map.x₀+sim.body.a.b.map.xₚ,sim)[3]
        # update ODE, first pack current state, solve and extract
        force = rot(sim.body.a.b.map.θ[3])*force # transform to body frame
        SciMLBase.set_u!(gravity,[x₁,u₁,a₁,x₂,u₂,a₂,θ,ω,α,force[1:2]...,moment])
        OrdinaryDiffEq.step!(gravity,sim.flow.Δt[end],true)
        x₁,u₁,a₁,x₂,u₂,a₂,θ,ω,α = gravity.u[1:9]
        # remeasure the sim
        θᵢ = SA{Float32}[0,0,θ]
        @show θᵢ
        ωᵢ = SA{Float32}[0,0,ω]
        sim.sim.body = setmap(sim.body;θ=θᵢ,ω=ωᵢ) # update rotational variables
        measure!(sim)
        acceleration = -rot(θ)*SA[a₁,a₂,0.0f0] # acceleration in lab frame
        velocity = -rot(θ)*SA[u₁,u₂,0.0f0] # velocity in lab frame
        biot_mom_step_fall!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                            fmm=sim.fmm,udf=fall!,acceleration,U=velocity) # change of frame
    end
    # now we have 1/2 a disk
    Cd,Cl = -4WaterLily.total_force(sim)[1:2]/R^2
    Cm = 4WaterLily.pressure_moment(x₀,sim)[3]/R^3
    (;t,Cd,Cl,Cm,u₁,u₂,a₁,a₂,θ,ω,α)
end |> Table

drag!(sim,times,R=sim.L,x₀=SA[R,0,0];remeasure=false) = map(times) do t
    @show t; flush(stdout)
    sim_step!(sim,t;remeasure)
    Cd,Cl = -8WaterLily.total_force(sim)[1:2]/R^2
    Cm = 8WaterLily.pressure_moment(x₀,sim)[3]/R^3
    (;t,Cd,Cl,Cm)
end |> Table

# Dynamic opening
using TypedTables,JLD2,Plots
N = 2^6; times = 0.05:0.05:20
θ₀ = 0.2f0
# H(t,k=30) = (t+1)/2-(t-1)/2*tanh(k*(t-1))
H = 1.0; ρ=10.f0; R=2N/3.f0; U=1.f0 # only values H ∈ [0,1]
sim = kirigami(N;mem=CuArray,H,fall=true,θ₀);
u₀ = zeros(12); u₀[7] = θ₀ # initial rotation
# all quantities for 1/2 of the disk, assumes thickness of disk is 3 for mass, ρ is density ratios
# m=3πρR² m11 = 8/3R³, m22=?, Im = 3πρR⁴/4, Ia = 16/45πR⁵
params = (ρ*3.f0*π*R^2/2.0,4/3.f0R^3,1.f0*R,ρ*3.f0*π*R^4/8.0f0,(8/45.f0)*π*R^5,-U^2/2R)
gravity = init(ODEProblem(gravity!,u₀,extrema(times),params),Tsit5(),abstol=1e-6,reltol=1e-6,save_everystep=false)
data = freefalling!(sim,times,gravity,remeasure=false)
save_object("kirigami_N$(N)_Hdynamic_hist_fall.jld2",data)
begin
    data = load_object("kirigami_N$(N)_Hdynamic_hist_fall.jld2")
    # plot(data.t,data.Cd,label="dynamic",ylabel="Cd",xlabel="time")
    plot(data.t,data.u₁,label="u₁",xlabel="time")
    plot!(data.t,data.u₂,label="u₂",xlabel="time")
    plot!(data.t,data.θ,label="θ")
end
