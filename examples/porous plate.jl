using WaterLily,BiotSavartBCs,Plots,CUDA,StaticArrays

function porous(r;g=r/4,α=0.9,θ=π/20,Re=15e3,T=Float32,mem=Array)
    g,α,θ,thk = T(g),T(α),T(θ),T(1+0.5√2)                                  # fix the variable types
    s,c = sincos(θ); map(xy,t) = SA[s c; -c s]*(xy-SA[2r,r])               # shift and rotate
    @inline segment(x,y,h) = hypot(x,y-clamp(y,-h+thk,h-thk))-thk          # thickened line segment
    segments((x,y),t) = segment(x,y-round(y/g)*g,g*α/2)                    # multiple segments, modulo g
    body = AutoBody(segments,map) ∩ AutoBody((xy,t)->segment(xy...,r),map) # clip plate length
    BiotSimulation((6r,3r),(1,0),2r;ν=2r/Re,body,T,mem)                    # Simulation with Biot-Savart BCs
end

# totally solid
sim = porous(64,mem=CuArray,α=2);sim_step!(sim)
sim_step!(sim,30,remeasure=false);#sim_gif!(sim,duration=30,plotbody=true,clims=(-50,50))
mean = MeanFlow(sim.flow)
while sim_time(sim)<60
    sim_step!(sim)
    WaterLily.update!(mean,sim.flow)
end
flood(mean.P[inside(sim.flow.p)],clims=(-1,1)); body_plot!(sim)
savefig("solid_P.png")
@inside sim.flow.σ[I] = WaterLily.curl(3,I,mean.U)*sim.L
flood(sim.flow.σ[inside(sim.flow.p)],clims=(-20,20)); body_plot!(sim)
savefig("solid_ω.png")

# 90% solid
sim = porous(64,mem=CuArray,α=0.9);sim_step!(sim)
sim_gif!(sim,duration=30,plotbody=true,clims=(-50,50))
mean = MeanFlow(sim.flow)
while sim_time(sim)<60
    sim_step!(sim)
    WaterLily.update!(mean,sim.flow)
end
flood(mean.P[inside(sim.flow.p)],clims=(-1,1)); body_plot!(sim)
savefig("porous_90_P.png")
@inside sim.flow.σ[I] = WaterLily.curl(3,I,mean.U)*sim.L
flood(sim.flow.σ[inside(sim.flow.p)],clims=(-20,20)); body_plot!(sim)
savefig("porous_90_ω.png")
