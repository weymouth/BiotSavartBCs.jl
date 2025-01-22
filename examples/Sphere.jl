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

using Plots, LaTeXStrings
let 
    path = "/home/marin/Workspace/BiotSavartBCs.jl/examples/"
    small = jldopen(path*"sphere_320x128x128.jld2")
    medium = jldopen(path*"sphere_480x192x192.jld2")
    large = jldopen(path*"sphere_640x256x256.jld2")

    plot(small["time"], small["drag"], label="320x128x128", xlims=(0,200),ylims=(0.,0.5),
         xlabel=L"$tU/L$", framestyle=:box, size=(600, 600), legend=:bottomright,
         legendfontsize=14, tickfontsize=18, labelfontsize=18, left_margin=Plots.Measures.Length(:mm, 5),
         ylabel=L"$C_D$"
    )
    plot!(medium["time"], medium["drag"], label="480x192x192")
    plot!(large["time"], large["drag"], label="640x256x256")
    savefig("drag.png")
 
    p_cd = plot()
    for (case,D) in zip([small,medium,large],[128,192,256])
        t = case["time"]
        idx = t .> 100
        fx, t = case["drag"][idx], t[idx]
        CD_mean = sum(fx[2:end].*diff(t))/sum(diff(t))
        println("▷ ΔT [CTU] = $(t[end]-t[1])")
        println("▷ CD mean = $CD_mean")
        scatter!(p_cd, [D^2/(π*44^2)], [CD_mean], grid=true, ms=8, ylims=(0.25,0.4501), xlims=(2,12),
                 xlabel=L"$A/\pi R^2$", lw=0, framestyle=:box, size=(600, 600), legend=:bottomright,
                 legendfontsize=14, tickfontsize=18, labelfontsize=18, left_margin=Plots.Measures.Length(:mm, 5),
                 ylabel=L"$\overline{C_D}$", label=:none
            )
    end
    hline!(p_cd, [0.394], linestyle=:dash, color=:blue, label=L"\mathrm{Rodriguez}\,\,et\,\,al\mathrm{.\,\,(DNS)}")
    hline!(p_cd, [0.355], linestyle=:dashdot, color=:green, label=L"\mathrm{Yun}\,\,et\,\,al\mathrm{.\,\,(LES)}")
    savefig(p_cd,"validation_sphere.png")


    p = small["p"]; u = small["u"]
    R = inside(p[:,:,1]); zin = size(p,3) ÷ 2 -1
    p1=contourf(p[R,zin]',cfill=:viridis,lw=0.0,dpi=600,aspect_ratio=:equal)
    p2=contourf(u[R,zin,1]',cfill=:viridis,lw=0.0,dpi=600,aspect_ratio=:equal)
    p3=contourf(u[R,zin,2]',cfill=:viridis,lw=0.0,dpi=600,aspect_ratio=:equal)
    plot(p1,p2,p3)
    savefig("small.png")

    p = medium["p"]; u = medium["u"]
    R = inside(p[:,:,1]); zin = size(p,3) ÷ 2 -1
    p1=contourf(p[R,zin]',cfill=:viridis,lw=0.0,dpi=600,aspect_ratio=:equal)
    p2=contourf(u[R,zin,1]',cfill=:viridis,lw=0.0,dpi=600,aspect_ratio=:equal)
    p3=contourf(u[R,zin,2]',cfill=:viridis,lw=0.0,dpi=600,aspect_ratio=:equal)
    plot(p1,p2,p3)
    savefig("medium.png")

    p = large["p"]; u = large["u"]
    R = inside(p[:,:,1]); zin = size(p,3) ÷ 2 -1
    p1=contourf(p[R,zin]',cfill=:viridis,lw=0.0,dpi=600,aspect_ratio=:equal)
    p2=contourf(u[R,zin,1]',cfill=:viridis,lw=0.0,dpi=600,aspect_ratio=:equal)
    p3=contourf(u[R,zin,2]',cfill=:viridis,lw=0.0,dpi=600,aspect_ratio=:equal)
    plot(p1,p2,p3)
    savefig("large.png")
end