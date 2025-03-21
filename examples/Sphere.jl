using WaterLily,StaticArrays,BiotSavartBCs,CUDA
using JLD2

function make_sphere(domain; N=2^6, R=N÷3, U=1, Re=3700, T=Float32, mem = Array)
    body = AutoBody((x,t)->√sum(abs2,x .- domain[2]÷2)-R)
    BiotSimulation(domain, (U,0,0), R; ν=U*R/Re, body, T, mem)
end
# size
N=2^7
params = [(5N÷2,N,N) (15N÷4,3N÷2,3N÷2) (5N,2N,2N)]
for domain in params
    # make the sim
    sim = make_sphere(domain;N=N,R=44,mem=CUDA.CuArray)
    sim_step!(sim;remeasure=false)
    time = 0.1:0.1:200
    # run
    drag = map(time) do t
        sim_step!(sim,t;remeasure=false)
        @show t
        -WaterLily.pressure_force(sim)[1]/(0.5π*sim.L^2)
    end
    t = map(i->string(i),domain)
    jldsave("sphere_$(t[1])x$(t[2])x$(t[3]).jld2"; p=Array(sim.flow.p), 
            u=Array(sim.flow.u), time=time, drag=drag)
end

using Plots
let 
    path = "/home/marin/Workspace/BiotSavartBCs.jl/examples/"
    small = jldopen(path*"sphere_320x128x128.jld2")
    medium = jldopen(path*"sphere_480x192x192.jld2")
    large = jldopen(path*"sphere_640x256x256.jld2")
    blues = colormap("Blues", 8)[3:end] # Biot savart
 
    blockage = plot(ylims=(0.25,0.5), xlims=(0,1),
                    xlabel="πR²/A", lw=0, legend=:bottomright, size=(400,400),
                    right_margin=Plots.Measures.Length(:mm, 5),
                    ylabel="Mean drag coefficient", )
    drag = plot(xlims=(0,200),ylims=(0.25,0.5),
                xlabel="Convective time", legend=:bottomright, size=(400,400),
                right_margin=Plots.Measures.Length(:mm, 5),
                ylabel="Drag coefficient")
    labels = ["3.6Dx1.5Dx1.5D","5.5Dx2.2Dx2.2D","7.2Dx2.9Dx2.9D"]
    for (i,case,D) in zip([2,4,6],[small,medium,large],[128,192,256])
        t = case["time"]; idx = t .> 100
        plot!(drag, t, case["drag"], label=labels[i÷2], c=blues[i])
        fx, t = case["drag"][idx], t[idx]
        CD_mean = sum(fx[2:end].*diff(t))/sum(diff(t))
        println("▷ ΔT [CTU] = $(t[end]-t[1])")
        println("▷ CD mean = $CD_mean")
        scatter!(blockage, [(π*44^2)/D^2], [CD_mean], label=:none, c=blues[i])
    end
    for (i,pl) in enumerate([blockage drag])
        hline!(pl, [0.394], linestyle=:dash, color=:black, label=ifelse(i==1,"Rodriguez et al. (DNS)",:none))
        hline!(pl, [0.355], linestyle=:dot, color=:grey, label=ifelse(i==1,"Yun et al. (LES)",:none))
    end
    savefig(drag,"drag.png")
    savefig(blockage,"validation_sphere.png")
end