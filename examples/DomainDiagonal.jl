using WaterLily,StaticArrays,CUDA,BiotSavartBCs
using JLD2,Plots

norm(x) = √sum(abs2,x)
function circ(D,n,m;Re=200,U=(1/√2,1/√2),mem=Array) 
    body = AutoBody((x,t)->√sum(abs2,x.-2.2*D)-D÷2)
    BiotSimulation((n*D,m*D), U, D; body, ν=norm(U)*D/Re,mem)
end

# domain size
D,n,m = (64,7,7)
sim = circ(D,n,m;mem=CUDA.CuArray)
u = Float32[]; Is = CartesianIndex(D+m*D÷2,m*D÷2); forces = []
stats = @timed anim = @animate for tᵢ in range(0.,100.,step=0.1)
    while sim_time(sim) < tᵢ
        sim_step!(sim;remeasure=false)
        pres,visc = WaterLily.pressure_force(sim),WaterLily.viscous_force(sim)
        push!(forces,[sim_time(sim),pres...,visc...])
    end
    push!(u,norm(sim.flow.u[Is,:]))
    println("tU/L=",round(sim_time(sim),digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ[inside(sim.flow.σ)]|>Array,clims=(-10,10))
end
gif(anim,"rotated_circle_$(n)Dx$(m)D_omega_true.gif")
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
jldsave("rotated_circle_$(n)Dx$(m)D.jld2"; θ=(n,m,true), u=u, p=Array(sim.flow.p), 
        f=forces, ω=Array(sim.flow.σ), time=stats.time)