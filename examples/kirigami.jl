using WaterLily,BiotSavartBCs,CUDA
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1) # good idea when accelerating from rest
function kirigami(N;R=2N/3,H=0,rings=16,U=1,a=1,Re=1e4,mem=ArrayU=1,T=Float32,ϵ=T(1/2),half_thk=ϵ+1/T(√2))
    ring(R₀,R₁,x₀,x₁,ϕ) = AutoBody() do (x,y,z),t 
        r,θ = hypot(y,z),atan(z,y); δx,xₘ = (x₁-x₀)/2,(x₁+x₀)/2
        hypot(x-xₘ-δx*cos(4θ+ϕ),r-clamp(r,R₀,R₁-half_thk))-half_thk
    end
    δR = T(R/rings); δH = T(R*H/rings^2); x₀ = T(max(R*(1-H)/2,δR+half_thk-min(0,R*H)))
    body = sum(i -> ring(δR*(i-1), δR*i, x₀+δH*(i-1)^2, x₀+δH*i^2, π*(i%2)), 1:rings)
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
    sim_step!(sim,t;remeasure=false)
    -2WaterLily.total_force(sim)[1]/sim.L^2
end

using TypedTables,JLD2
rings = 16
Ca_H = map([0.5.^(-2:2);0]) do H
    sim = kirigami(2^8;H,rings,mem=CUDA.CuArray,T=Float32)
    (;H,rings,Ca=4drag!(sim,0.025))
end |> Table

using Plots
scatter(Ca_H.H,Ca_H.Ca,label="simulation",xlabel="H/R",ylabel="Ca",ylims=(0,3));
hline!([8/3],label="disk limit",ls=:dash,legend=:topright);
hline!([π^2/4rings],label="thin ring limit",ls=:dash)
savefig("examples/kirigami_CaH.png")
# using GLMakie
# viz!(sim,colorrange=(0.1,0.85),body_color=:blue,body2mesh=true,colormap=:amp)
# Makie.save("examples/kirigami.png", Makie.current_figure())