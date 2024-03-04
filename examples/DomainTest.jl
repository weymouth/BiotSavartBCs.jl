using BiotSavartBCs
using WaterLily
using CUDA
include("Diagnostics.jl")
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
    u = Float32[]; I = CartesianIndex(D+m*D÷2,m*D÷2); forces = []
    use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
    @assert !all(sim.pois.n .== 32) "pressure problem"
    while sim_time(sim)<t_end
        use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
        pres,visc = diagnostics(sim;κ=2/D); 
        # pres,visc = -2WaterLily.∮nds(sim.flow.p,sim.flow.f,sim.body,0.0),SA[0.,0.];
        push!(forces,[sim_time(sim),pres...,visc...])
        sim_time(sim)%0.1<sim.flow.Δt[end]/sim.L && push!(u,sim.flow.u[I,2])
        sim_time(sim)%1<sim.flow.Δt[end]/sim.L && @show sim_time(sim)
    end
    return sim,u,forces
end
params = [(5,2,true) (8,5,true) (10,8,true) (5,2,false) (8,5,false) (10,8,false)]
using JLD2
jldopen("examples/smalltol.jld2", "w") do file
    for (i,θ) ∈ enumerate(params)
        mygroup = JLD2.Group(file,"case$(i)")
        mygroup["θ"] = θ
        @show θ
        n,m,use_biotsavart = θ
        (sim,u,f) = wake_velocity(n,m;t_end=100,use_biotsavart);
        mygroup["u"] = u
        mygroup["p"] = sim.flow.p
        mygroup["f"] = f
        @inside sim.flow.σ[I] = BiotSavartBCs.centered_curl(3,I,sim.flow.u)*sim.L/sim.U
        mygroup["ω"] = sim.flow.σ
    end
end

using Plots,FFTW
jldopen("examples/smalltol.jld2", "r") do file

    koumou = [[0.036789297658864184, 1.4994350282485878],
                [0.06354515050167331, 1.2135593220338983],
                [0.137123745819399, 0.8677966101694914],
                [0.4983277591973254, 0.7548022598870054],
                [0.8628762541806032, 1.0203389830508476],
                [1.2541806020066906, 1.2689265536723167],
                [1.729096989966557, 1.2723163841807907],
                [2.210702341137125, 1.1762711864406776],
                [2.6588628762541813, 1.108474576271186],
                [3.147157190635452, 1.0542372881355924],
                [3.622073578595319, 1.0135593220338974],
                [4.096989966555183, 0.9819209039548012],
                [4.561872909698996, 0.9593220338983038]]

    gillis = [[0.04013377926421513, 1.3581920903954803],
                [0.060200668896322584, 1.1209039548022601],
                [0.11036789297658989, 0.9231638418079098],
                [0.32441471571906466, 0.7163841807909602],
                [0.5819397993311053, 0.8090395480225988],
                [0.7658862876254195, 0.9480225988700564],
                [1.030100334448162, 1.15819209039548],
                [1.4648829431438153, 1.301694915254237],
                [1.9230769230769242, 1.2327683615819205],
                [2.374581939799332, 1.1514124293785306],
                [2.8896321070234134, 1.0836158192090388],
                [3.374581939799332, 1.0316384180790952],
                [3.8628762541806014, 0.9977401129943493],
                [4.307692307692307, 0.9717514124293775],
                [4.806020066889634, 0.9435028248587555],
                [5.197324414715718, 0.9276836158192074],
                [5.595317725752508, 0.9107344632768345]]

    plt = plot(dpi=300);
    for i ∈ 1:length(params)
        mygroup = file["case$(i)"]
        n,m,use_biotsavart = mygroup["θ"]
        m==3 && use_biotsavart && continue
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        u = Float32.(mygroup["u"])
        plot!(plt,range(0,10,length=100),u[1:100],label="$(n)Dx$(m)D using "*BC;ls)
    end
    title!("Domain size study");xlabel!("Convective time");ylabel!("v/U near centerline")
    savefig("start.png")

    plt = plot(dpi=300);
    for i ∈ 1:length(params)
        mygroup = file["case$(i)"]
        n,m,use_biotsavart = mygroup["θ"]
        m==3 && use_biotsavart && continue
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        u = Float32.(mygroup["u"][700:end])
        u_hat=fft(u)./0.5length(u)
        plot!(plt,range(0,2.5,length=50),abs.(u_hat[1:50]),label="$(n)Dx$(m)D using "*BC;ls)
    end
    title!("Domain size study");xlabel!("Strouhal");ylabel!("PSD(v) near centerline")
    savefig("fft.png")

    plt = plot(dpi=300);
    for i ∈ 1:length(params)
        mygroup = file["case$(i)"]
        n,m,use_biotsavart = mygroup["θ"]
        m==3 && use_biotsavart && continue
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        f = Float32.(reduce(vcat,mygroup["f"]')[1:end,:])
        plot!(plt,f[:,1],(f[:,2].+f[:,4])./32,label="$(n)Dx$(m)D using "*BC;ls)
        # plot!(plt,f[:,1],f[:,4]./32,label="$(n)Dx$(m)D using "*BC;ls)
    end
    koumou = reduce(hcat,koumou)
    gillis = reduce(hcat,gillis)
    scatter!(plt,koumou[1,:],koumou[2,:],label="Koumoutsakos",marker=:x)
    scatter!(plt,gillis[1,:],gillis[2,:],label="Gillis",marker=:s)
    plot!(plt,0:0.01:0.35,Cd.(0:0.01:0.35),label="Analytical",ls=:dash,color=:red)
    xlims!(0,6);ylims!(0,3.5);
    title!("Domain size study");xlabel!("Convective time");ylabel!("pressure force on the body")
    savefig("force.png")
end
