using BiotSavartBCs
using WaterLily
using CUDA
include("Diagnostics.jl")
include("TwoD_plots.jl")
function Cd(t;Re=550)
    # W. Collins, S. Dennis, The initial ﬂow past an impulsively started circular cylinder, Q. J. Mech. Appl. Math. 26 (1973) 53–75.
    k = 4√(t/Re)
    return π/√(Re*t)*((2.257+k-0.141k^2+0.031k^3)+
                      (8.996-41k+143.8k^2+45.4k^3)*t^2 +
                      (20.848-314.08k-1851.36k^2-194.8k^3)*t^4 +
                      (28.864+6.272k)*t^6)
end
#Quantify domain sensitivity
circ(D,n,m;Re=550,U=1,mem=CUDA.CuArray) = Simulation((n*D,m*D), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m*D÷2)-D÷2),ν=U*D/Re,mem)
function wake_velocity(n=5,m=3;D=64,use_biotsavart=true,t_end=100)
    sim = circ(D,n,m); ω = MLArray(sim.flow.σ)
    u = Float32[]; Is = CartesianIndex(D+m*D÷2,m*D÷2); forces = []; iter = 0
    use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
    @assert !all(sim.pois.n .== 32) "pressure problem"
    while sim_time(sim)<t_end
        use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
        pres,visc = diagnostics(sim;κ=2/D); 
        push!(forces,[sim_time(sim),pres...,visc...])
        sim_time(sim)%0.1<sim.flow.Δt[end]/sim.L && push!(u,sim.flow.u[Is,2])
        sim_time(sim)%1<sim.flow.Δt[end]/sim.L && @show sim_time(sim)
        if sim_time(sim)%0.2<sim.flow.Δt[end]/sim.L
            @inside sim.flow.σ[I] = BiotSavartBCs.centered_curl(3,I,sim.flow.u)*sim.L/sim.U
            flood(sim.flow.σ[inside(sim.flow.σ)]|>Array,clims=(-10,10))
            savefig("Disk_$(n)Dx$(m)D_omega_$(iter)_$use_biotsavart.png")
            iter += 1
        end
    end
    return sim,u,forces
end
params = [(5,3,true) (8,5,true) (10,8,true) (5,3,false) (8,5,false) (10,8,false) (20,16,false) (30,24,false)]
using JLD2
# for (i,θ) ∈ enumerate(params)
#     jldopen("domain_$(θ[1])_$(θ[2])_$(θ[3]).jld2", "w") do file
#         mygroup = JLD2.Group(file,"case")
#         mygroup["θ"] = θ
#         @show θ
#         n,m,use_biotsavart = θ
#         (sim,u,f) = wake_velocity(n,m;t_end=100,use_biotsavart);
#         mygroup["u"] = u
#         mygroup["p"] = sim.flow.p
#         mygroup["f"] = f
#         @inside sim.flow.σ[I] = BiotSavartBCs.centered_curl(3,I,sim.flow.u)*sim.L/sim.U
#         mygroup["ω"] = sim.flow.σ
#     end
# end
using Plots,FFTW
let
    # file = jldopen("reference_data.jld2", "r")
    # koumou = file["koumou"]
    # gillis = file["gillis"]
    # close(file)

    plt = plot(dpi=300);
    for (i,θ) ∈ enumerate(params)
        file = jldopen("/home/marinlauber/domain_$(θ[1])_$(θ[2])_$(θ[3]).jld2", "r")
        mygroup = file["case"]
        n,m,use_biotsavart = mygroup["θ"]
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        u = Float32.(mygroup["u"])
        plot!(plt,range(0,10,length=100),u[1:100],label="$(n)Dx$(m)D using "*BC;ls)
    end
    title!("Domain size study");xlabel!("Convective time");ylabel!("v/U near centerline")
    savefig("start.png")

    plt = plot(dpi=300);
    for (i,θ) ∈ enumerate(params)
        file = jldopen("/home/marinlauber/domain_$(θ[1])_$(θ[2])_$(θ[3]).jld2", "r")
        mygroup = file["case"]
        n,m,use_biotsavart = mygroup["θ"]
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        u = Float32.(mygroup["u"][700:end])
        u_hat=fft(u)./0.5length(u)
        plot!(plt,range(0,2.5,length=50),abs.(u_hat[1:50]),label="$(n)Dx$(m)D using "*BC;ls)
    end
    title!("Domain size study");xlabel!("Strouhal");ylabel!("PSD(v) near centerline")
    savefig("fft.png")

    plt = plot(dpi=300);
    for (i,θ) ∈ enumerate(params)
        file = jldopen("/home/marinlauber/domain_$(θ[1])_$(θ[2])_$(θ[3]).jld2", "r")
        mygroup = file["case"]
        n,m,use_biotsavart = mygroup["θ"]
        BC = ifelse(θ[3],"Biot-Savart","Reflection")
        ls = ifelse(θ[3],:solid,:dash)
        f = Float32.(reduce(vcat,mygroup["f"]')[1:end,:])
        plot!(plt,f[:,1],(f[:,2].+f[:,4])./32,label="$(n)Dx$(m)D using "*BC;ls)
    end
    file = jldopen("/home/marinlauber/rotated_cylinder.jld2", "r")
    mygroup = file["case1"]
    f = Float32.(reduce(vcat,mygroup["f"]')[1:end,:])
    force = .√((f[:,2].+f[:,4]).^2 .+ (f[:,3].+f[:,5]).^2)
    plot!(plt,f[:,1],force./32,label="Rotated Cylinder Biot-Savart")
    xlims!(0,10);ylims!(0,2.0);
    title!("Domain size study");xlabel!("Convective time");ylabel!("2Force/ρUR")
    savefig("force.png")
end
