using WaterLily,StaticArrays,CUDA,BiotSavartBCs
using JLD2,Plots,FFTW

#Quantify domain sensitivity
function circ(D,n,m;Re=200,U=1,mem=Array,use_biotsavart=true)
    body = AutoBody((x,t)->√sum(abs2,x .- m*D÷2)-D÷2)
    use_biotsavart && return BiotSimulation((n*D,m*D), (U,0), D; body, ν=U*D/Re, mem)
    Simulation((n*D,m*D), (U,0), D; body, ν=U*D/Re, mem)
end
function wake_velocity(n=5,m=3;D=64,use_biotsavart=true,t_end=100)
    sim = circ(D,n,m;mem=Array,use_biotsavart)
    u = Float32[]; Is = CartesianIndex(D+m*D÷2,m*D÷2); forces = []; 
    sim_step!(sim); @assert !all(sim.pois.n .== 32) "pressure problem"
    @time anim = @animate for tᵢ in range(0.,t_end,step=0.1)
        while sim_time(sim) < tᵢ
            sim_step!(sim;remeasure=false)
            pres,visc = WaterLily.pressure_force(sim),WaterLily.viscous_force(sim)
            push!(forces,[sim_time(sim),pres...,visc...])
        end
        push!(u,sim.flow.u[Is,2])
        println("tU/L=",round(sim_time(sim),digits=4),
                ", Δt=",round(sim.flow.Δt[end],digits=3))
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ[inside(sim.flow.σ)]|>Array,clims=(-10,10))
    end
    gif(anim,"circle_$(n)Dx$(m)D_omega_$use_biotsavart.gif")
    return sim,u,forces
end
params = [(5,3,true) (8,5,true) (10,8,true) (5,3,false) (8,5,false) (10,8,false) (20,16,false) (30,24,false)]
# run all the cases
for (i,θ) ∈ enumerate(params)
    n,m,use_biotsavart = θ; @show θ
    # run wake velocity
    (sim,u,f) = wake_velocity(n,m;t_end=100,use_biotsavart);
    # add vorticity
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    jldsave("domain_$(n)_$(m)_$(use_biotsavart).jld2";
        θ=θ, u=u, p=sim.flow.p, f=f, ω=sim.flow.σ
    )
end
# make the plots
let
    plt = plot(dpi=300);
    for (i,θ) ∈ enumerate(params)
        file = jldopen("/home/marinlauber/domain_$(θ[1])_$(θ[2])_$(θ[3]).jld2")
        n,m,use_biotsavart = file["θ"]
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        u = Float32.(file["u"])
        plot!(plt,range(0,10,length=100),u[1:100],label="$(n)Dx$(m)D using "*BC;ls)
    end
    title!("Domain size study");xlabel!("Convective time");ylabel!("v/U near centerline")
    savefig("start.png")

    plt = plot(dpi=300);
    for (i,θ) ∈ enumerate(params)
        file = jldopen("/home/marinlauber/domain_$(θ[1])_$(θ[2])_$(θ[3]).jld2")
        n,m,use_biotsavart = file["θ"]
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        u = Float32.(file["u"][700:end])
        u_hat=fft(u)./0.5length(u)
        plot!(plt,range(0,2.5,length=50),abs.(u_hat[1:50]),label="$(n)Dx$(m)D using "*BC;ls)
    end
    title!("Domain size study");xlabel!("Strouhal");ylabel!("PSD(v) near centerline")
    savefig("fft.png")

    plt = plot(dpi=300);
    for (i,θ) ∈ enumerate(params)
        file = jldopen("/home/marinlauber/domain_$(θ[1])_$(θ[2])_$(θ[3]).jld2")
        n,m,use_biotsavart = file["θ"]
        BC = ifelse(θ[3],"Biot-Savart","Reflection")
        ls = ifelse(θ[3],:solid,:dash)
        f = Float32.(reduce(vcat,file["f"]')[1:end,:])
        plot!(plt,f[:,1],(f[:,2].+f[:,4])./32,label="$(n)Dx$(m)D using "*BC;ls)
    end
    file = jldopen("/home/marinlauber/rotated_cylinder.jld2", "r")
    f = Float32.(reduce(vcat,file["f"]')[1:end,:])
    force = .√((f[:,2].+f[:,4]).^2 .+ (f[:,3].+f[:,5]).^2)
    plot!(plt,f[:,1],force./32,label="Rotated Cylinder Biot-Savart")
    xlims!(0,6);ylims!(0,2.0);
    title!("Domain size study");xlabel!("Convective time");ylabel!("2Force/ρUR")
    savefig("force.png")
end
