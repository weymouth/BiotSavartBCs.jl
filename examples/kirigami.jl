using WaterLily,BiotSavartBCs,CUDA,StaticArrays
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1) # good idea when accelerating from rest
function kirigami(N;R=2N/3,H=zero,rings=16,U=1,a=1,Re=1e4,mem=Array,T=Float32,ϵ=T(1/2),half_thk=ϵ+1/T(√2))
    R = T(R); δR = R/rings; @inline δH(t) = δR*H(U*t/2R)/rings
    @inline mapped(f) = AutoBody(f,(x,t)->(x-SA[R,0,0])+SA[2R/3*H(U*t/2R),0,0])
    @inline ring(R₀,R₁,x₀,x₁,ϕ) = mapped() do (x,y,z),t 
        r,θ = hypot(y,z),atan(z,y)
        δx = x₀(t)+tanh(π*r/δR)*(x₁(t)-x₀(t))*(1+cos(4θ+ϕ))/2
        hypot(x-δx,r-clamp(r,R₀+half_thk,R₁-half_thk))-half_thk
    end
    body = sum(i -> ring(δR*(i-1), δR*i, t->δH(t)*(i-1)^2, t->δH(t)*i^2, π*(i%2)), 1:rings)
    H == zero && (body = ring(0,R,H,H,0)) # flat disk case
    Ut(i,x,t) = i==1 ? min(a*t/2R,U*one(t)) : zero(t) # velocity BC
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
drag!(sim,times,R=sim.L,x₀=SA[R,0,0];remeasure=false) = map(times) do t
    @show t; flush(stdout)
    sim_step!(sim,t;remeasure)
    Cd,Cl = -4WaterLily.total_force(sim)[1:2]/R^2
    Cm = 4WaterLily.pressure_moment(x₀,sim)[3]/R^3
    (;t,Cd,Cl,Cm)
end |> Table

# Dynamic opening
using TypedTables,JLD2,Plots
N = 2^8; times = 0.05:0.05:3
H(t,k=30) = (t+1)/2-(t-1)/2*tanh(k*(t-1))
sim = kirigami(N;mem=CuArray,H);
data = drag!(sim,times,remeasure=true)
save_object("kirigami_N$(N)_Hdynamic_hist.jld2",data)
begin
    data = load_object("kirigami_N$(N)_Hdynamic_hist.jld2")
    plot(data.t,data.Cd,label="dynamic",ylabel="Cd",xlabel="time")
    data = load_object("kirigami_N384_H0.0_hist.jld2")
    plot!(data.times,data.Cd,label="H=0",ylabel="Cd",xlabel="time",color=:grey)
    data = load_object("kirigami_N384_H1.0_hist.jld2")
    plot!(data.times,data.Cd,label="H=1",ylabel="Cd",xlabel="time",color=:grey,ls=:dash)
end
savefig("kirigami_opening.png")
0
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