using WaterLily,BiotSavartBCs,CUDA
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1) # good idea when accelerating from rest
function kirigami(N;R=2N/3,H=0,rings=16,U=1,a=1,Re=1e4,mem=Array,T=Float32,ϵ=T(1/2),half_thk=ϵ+1/T(√2))
    ring(R₀,R₁,x₀,x₁,ϕ) = AutoBody() do (x,y,z),t 
        r,θ = hypot(y,z),atan(z,y); δx,xₘ = (x₁-x₀)/2,(x₁+x₀)/2
        hypot(x-xₘ-δx*cos(4θ+ϕ),r-clamp(r,R₀+half_thk,R₁-half_thk))-half_thk
    end
    δR = T(R/rings); δH = T(R*H/rings^2); x₀ = T(max(R*(1-H)/2,δR+half_thk-min(0,R*H)))
    body = sum(i -> ring(δR*(i-1), δR*i, x₀+δH*(i-1)^2, x₀+δH*i^2, π*(i%2)), 1:rings)
    H == 0 && (body = ring(0,R,x₀,x₀,0)) # flat disk case
    Ut(i,x,t::T) where T = i==1 ? convert(T,min(a*t/2R,U)) : zero(T) # velocity BC
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
drag!(sim,times) = map(times) do t
    @show t; flush(stdout)
    sim_step!(sim,t;remeasure=false)
    -2WaterLily.total_force(sim)[1]/sim.L^2
end

# Rings sweep
using TypedTables,JLD2
N,H = 3*2^7,1
times = 0.05:0.05:3
for rings = 4:4:20
    rings == 16 && continue # skip this one, done below
    @show rings; flush(stdout)
    sim = kirigami(N;H,rings,mem=CUDA.CuArray)
    data = Table(times=times,Cd=4drag!(sim,times))
    save_object("kirigami_N$(N)_H$(H)_rings$(rings)_hist.jld2",data)
    save!("kirigami_N$(N)_H$(H)_rings$(rings).jld2",sim.flow)
end

# using Plots
# scatter(Ca_H.H,Ca_H.Ca,label="simulation",xlabel="H/R",ylabel="Ca",ylims=(0,3));
# hline!([8/3],label="disk limit",ls=:dash,legend=:topright);
# hline!([π^2/4rings],label="thin ring limit",ls=:dash)
# savefig("examples/kirigami_CaH.png")

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

# using TypedTables,JLD2
# using Plots
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