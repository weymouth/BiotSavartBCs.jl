using WaterLily,BiotSavartBCs,TypedTables,CUDA
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1) # good idea when accelerating from rest
function kirigami(N;R=2N/3,H=R/2,rings=16,U=1,a=1,Re=1e4,mem=ArrayU=1,T=Float32,thk=T(3/2),ϵ=T(1/2))
    ring(R₀,R₁,x₀,x₁,ϕ) = AutoBody() do (x,y,z),t 
        r,θ = hypot(y,z),atan(z,y); δx,xₘ = (x₁-x₀)/2,(x₁+x₀)/2
        hypot(x-xₘ-δx*cos(4θ+ϕ),r-clamp(r,R₀+thk,R₁-thk))-thk
    end
    δR = T(R/rings); δH = T(H/rings^2); x₀ = T(max((R-H)/2,4thk-min(0,H)))
    body = sum(i -> ring(δR*(i-1), δR*i, x₀+δH*(i-1)^2, x₀+δH*i^2, π*(i%2)), 1:rings)
    Ut(i,x,t::T) where T = i==1 ? convert(T,min(a*t/R,U)) : zero(T) # velocity BC
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
    -2WaterLily.total_force(sim)[1]/sim.L^3
end

N = 2^7; R = 2N/3; H = 0
t = 0:0.01:0.1
sim = kirigami(N;R,H,mem=CUDA.CuArray,a=1,T=Float32);
CD = drag!(sim,t)
using Plots
plot(t,4CD); hline([8/3])
# using GLMakie
# viz!(sim,colorrange=(0.1,0.85),body_color=:blue,body2mesh=true,colormap=:amp)
# Makie.save("examples/kirigami.png", Makie.current_figure())