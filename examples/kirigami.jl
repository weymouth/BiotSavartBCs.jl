using WaterLily,BiotSavartBCs,CUDA,StaticArrays
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1) # good idea when accelerating from rest
linear(t)=min(t,one(t))
function kirigami(N;H=0,rings=16,U=1,a=1,Re=1e4,mem=Array,T=Float32,Ux=linear,R=T(2N/3),ϵ=T(1/2),half_thk=ϵ+1/T(√2),fall=false)
    δR = R/rings; δH = R*H/rings^2; x₀ = max(R*(1-H)/2,δR+half_thk-min(0,R*H))
    @inline mapped(f) = AutoBody(f,(x,t)->x-SA[x₀,0,0])
    @inline ring(R₀,R₁,x₀,x₁,ϕ) = mapped() do (x,y,z),t
        r,θ = hypot(y,z),atan(z,y)
        δx = x₀+tanh(π*r/δR)*(x₁-x₀)*(1+cos(4θ+ϕ))/2
        hypot(x-δx,r-clamp(r,R₀+half_thk,R₁-half_thk))-half_thk
    end
    body = sum(i -> ring(δR*(i-1), δR*i, δH*(i-1)^2, δH*i^2, π*(i%2)), 1:rings)
    H == 0 && (body = ring(0,R,0,0,0))
    Ut = fall ? (0,0,0) : (i,x,t)->(i==1 ? U*Ux(a*U*t/2R) : zero(t)) # velocity BC
    BiotSimulation((3N,N,N),Ut,R;U,ν=U*2R/Re,body,mem,T,ϵ,nonbiotfaces=(-2,-3))
end

import BiotSavartBCs: interaction,symmetry,image
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₂,sgn₂ = image(T,size(ω),-2)  # image target and sign in y
    T₃,sgn₃ = image(T,size(ω),-3)  # image target and sign in z
    T₂₃,_   = image(T₃,size(ω),-2) # image of image!
    # Add up the four contributions
    return interaction(ω,T,args...)+sgn₃*interaction(ω,T₃,args...)+
     sgn₂*(interaction(ω,T₂,args...)+sgn₃*interaction(ω,T₂₃,args...))
end
using TypedTables
drag!(sim,times,R=sim.L,x₀=SA[R,0,0];remeasure=false) = map(times) do t
    @show t; flush(stdout)
    sim_step!(sim,t;remeasure)
    Cd,Cl = -8WaterLily.total_force(sim)[1:2]/R^2
    Cm = 8WaterLily.pressure_moment(x₀,sim)[3]/R^3
    (;t,Cd,Cl,Cm)
end |> Table

# Test Ux ramps
using JLD2
N,H = 2^8,1; times = 0.05:0.05:10
for a in (0.5f0,1,2), (name,Ux) in zip(("linear", "tanh"), (linear, tanh))
    @show name; flush(stdout)
    sim = kirigami(N;H,mem=CuArray,a,Ux);
    data = drag!(sim,times,remeasure=true)
    save_object("kirigami_N$(N)_$(name)$(a)_hist.jld2",data)
    save!("kirigami_N$(N)_$(name)$(a).jld2",sim)
end
using Plots
plot();for (color,a) = zip(palette(:amp,4)[2:end], (0.5,1,2))
    data = load_object("kirigami_N$(N)_linear$(a)_hist.jld2")
    plot!(data.t,data.Cd,label=a;color)
    data = load_object("kirigami_N$(N)_tanh$(a)_hist.jld2")
    plot!(data.t,data.Cd,label=nothing,ls=:dash;color)
end;plot!(legend=:topright,legendtitle="a*",xlabel="time",ylabel="Cd")
savefig("kirigami_Ux_H1.png")
# # Angle sweep
# N = 3*2^7
# times = 0.05:0.05:3
# for α = 0.15:0.05:0.25, H = (-1,1)
#     @show α,H; flush(stdout)
#     data = nothing; sim = nothing; GC.gc(true)
#     @show CUDA.pool_status()
#     sim = kirigami(N;H,α,mem=CUDA.CuArray)
#     data = hist!(sim,times)
#     @show CUDA.pool_status()
#     save_object("kirigami_N$(N)_H$(H)_a$(α)_hist.jld2",data)
#     save!("kirigami_N$(N)_H$(H)_a$(α).jld2",sim)
# end
# using TypedTables,JLD2,Plots
# N = 3*2^7
# plot();for (color,α) = zip(palette(:amp,6)[2:end], 0.05:0.05:0.25)
#     H=1
#     data = load_object("kirigami_N$(N)_H$(H)_a$(α)_hist.jld2")
#     plot!(data.t,data.Cm,label=α;color)
#     H=-1
#     data = load_object("kirigami_N$(N)_H$(H)_a$(α)_hist.jld2")
#     plot!(data.t,data.Cm,label=nothing,ls=:dash;color)
# end;plot!(legend=:topleft,legendtitle="AoA",xlabel="time",ylabel="Cm")
# savefig("kirigami_AoA.png")

# # Rings sweep
# using TypedTables,JLD2
# N,H = 3*2^7,1
# times = 0.05:0.05:3
# for rings = 4:4:20
#     rings == 16 && continue # skip this one, done below
#     @show rings; flush(stdout)
#     sim = kirigami(N;H,rings,mem=CUDA.CuArray)
#     data = Table(times=times,Cd=4drag!(sim,times))
#     save_object("kirigami_N$(N)_H$(H)_rings$(rings)_hist.jld2",data)
#     save!("kirigami_N$(N)_H$(H)_rings$(rings).jld2",sim.flow)
# end
# using Plots
# N,H = 3*2^7,1
# plot(ylims=(0,8)); for (color,rings) = zip(palette(:amp,6)[2:end], 4:4:20)
#     data = load_object("kirigami_N$(N)_H$(H)_rings$(rings)_hist.jld2")
#     plot!(data.times,data.Cd;color,label=rings)
# end; plot!(legend=:topleft,legendtitle="rings",xlabel="time",ylabel="Cd")
# savefig("kirigami_rings.png")

# H sweep
# N = 3*2^7
# Hs = 0.5 .^ (-2:2); times = 0.05:0.05:3
# Hs = [-Hs; 0; reverse(Hs)] # include negative H for checking symmetry
# for H ∈ Hs
#     @show H; flush(stdout)
#     sim = kirigami(N;H,mem=CUDA.CuArray)
#     data = Table(times=times,Cd=4drag!(sim,times))
#     save_object("kirigami_N$(N)_H$(H)_hist.jld2",data)
#     save!("kirigami_N$(N)_H$(H).jld2",sim.flow)
# end

# using TypedTables,JLD2,Plots
# begin
#     N = 3*2^7
#     data = load_object("kirigami_N$(N)_H0.0_hist.jld2")
#     plot(data.times,data.Cd,label="H=0";color=:black,xlabel="time",ylabel="Drag");
#     Hs = 2.0 .^ (-2:2); colors = palette(:hot, length(Hs)+2)[2:end-1]
#     for (color,H) ∈ zip(colors,Hs)
#         data = load_object("kirigami_N$(N)_H$(H)_hist.jld2")
#         plot!(data.times,data.Cd,label="H=$(H)";color)
#         data = load_object("kirigami_N$(N)_H-$(H)_hist.jld2")
#         plot!(data.times,data.Cd,label="H=-$(H)";color,ls=:dash)
#     end
#     hline!([8/3],label="disk Ma",color=:grey,legend=:topleft);
#     hline!([π^2/4/16],label="thin ring Ma",color=:grey,ls=:dash)
# end
# savefig("kirigami_Cd_time.png")

# using GLMakie
# viz!(sim,colorrange=(0.1,0.85),body_color=:blue,body2mesh=true,colormap=:amp)
# Makie.save("examples/kirigami.png", Makie.current_figure())

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
    xᵢ,uᵢ,aᵢ,Fᵢ = u
    # unpack constant params
    m,mₐ,g = p
    # rates (du[3:4] are unused)
    du[1] = uᵢ
    du[2] = (Fᵢ - mₐ*aᵢ + m*g)/(m + mₐ)
end

# Free-falling simulation
freefalling!(sim,times,gravity,R=sim.L,x₀=SA[R,0,0];x0=0.f0,vel=0.f0,acc=0.f0,remeasure=false) = map(times) do t
    @show t; flush(stdout)
    while sim_time(sim) < t
         # compute pressure force
        force = -WaterLily.total_force(sim)
        # update ODE, first pack current state, solve and extract
        SciMLBase.set_u!(gravity,[x0,vel,acc,force[1]])
        OrdinaryDiffEq.step!(gravity,sim.flow.Δt[end],true)
        x0,vel,acc = gravity.u[1:3]
        # remeasure the sim
        remeasure && measure!(sim)
        biot_mom_step_fall!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                            fmm=sim.fmm,udf=fall!,acceleration=-SA[acc,0.0f0,0.0f0],U=-SA[vel,0.0f0,0.0f0]) # change of frame
    end
    Cd,Cl = -8WaterLily.total_force(sim)[1:2]/R^2
    Cm = 8WaterLily.pressure_moment(x₀,sim)[3]/R^3
    (;t,Cd,Cl,Cm,vel)
end |> Table

using OrdinaryDiffEq
# free falling
N = 2^7; times = 0.05:0.05:6
# all quantities for 1/4 of the disk, assumes thickness of disk is 3 for mass, ρ is density ratios
ρ=10.f0; R=2N/3.f0; U=1.f0
u₀ = [0.f0,0.f0,0.f0,0.f0]; params = (ρ*3.f0*π*R^2,2/3.f0R^2,-U^2/2R)
for (H,name) in zip((0,0.5f0,1.f0),("0", "half", "1"))
    @show name; flush(stdout)
    sim = kirigami(N;mem=CuArray,H,R,U,fall=true);
    # make an ODE problem
    gravity = init(ODEProblem(gravity!,u₀,extrema(times),params),Tsit5(),abstol=1e-6,reltol=1e-6,save_everystep=false)
    data = freefalling!(sim,times,gravity,remeasure=false)
    save_object("kirigami_N$(N)_H$(name)_fall_hist.jld2",data)
    save!("kirigami_N$(N)_H$(name)_fall.jld2",sim)
end
plot();for (color,a,label) = zip(palette(:amp,4)[2:end],("0","half","1"), ("0","1/2","1"))
    data = load_object("kirigami_N$(N)_H$(a)_fall_hist.jld2")
    plot!(data.t,data.Cd,label=label;color)
end;plot!(legend=:topright,legendtitle="H",xlabel="time",ylabel="Cd")
savefig("kirigami_fall.png")