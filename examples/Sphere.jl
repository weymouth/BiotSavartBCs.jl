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
    small = jldopen("sphere_320x128x128.jld2")
    medium = jldopen("sphere_480x192x192.jld2")
    large = jldopen("sphere_640x256x256.jld2")

    plot(small["time"], small["drag"], label="320x128x128")
    plot!(medium["time"], medium["drag"], label="480x192x192")
    plot!(large["time"], large["drag"], label="640x256x256")
    savefig("drag.png")

    p = small["p"]; u = small["u"]
    R = inside(p[:,:,1]); zin = size(p,3) ÷ 2 -1
    p1=contourf(p[R,zin]',cfill=:viridis,lw=0.0,dpi=600)
    p2=contourf(u[R,zin,1]',cfill=:viridis,lw=0.0,dpi=600)
    p3=contourf(u[R,zin,2]',cfill=:viridis,lw=0.0,dpi=600)
    plot(p1,p2,p3)
    savefig("small.png")

    p = medium["p"]; u = medium["u"]
    R = inside(p[:,:,1]); zin = size(p,3) ÷ 2 -1
    p1=contourf(p[R,zin]',cfill=:viridis,lw=0.0,dpi=600)
    p2=contourf(u[R,zin,1]',cfill=:viridis,lw=0.0,dpi=600)
    p3=contourf(u[R,zin,2]',cfill=:viridis,lw=0.0,dpi=600)
    plot(p1,p2,p3)
    savefig("medium.png")

    p = large["p"]; u = large["u"]
    R = inside(p[:,:,1]); zin = size(p,3) ÷ 2 -1
    p1=contourf(p[R,zin]',cfill=:viridis,lw=0.0,dpi=600)
    p2=contourf(u[R,zin,1]',cfill=:viridis,lw=0.0,dpi=600)
    p3=contourf(u[R,zin,2]',cfill=:viridis,lw=0.0,dpi=600)
    plot(p1,p2,p3)
    savefig("large.png")
end