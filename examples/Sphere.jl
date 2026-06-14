using WaterLily,StaticArrays,BiotSavartBCs,CUDA
using JLD2

function make_sphere(domain; N=2^6, R=N÷3, U=1, Re=3700, T=Float32, mem = Array)
    body = AutoBody((x,t)->√sum(abs2,x .- domain[2]÷2)-R)
    BiotSimulation(domain, (U,0,0), R; ν=U*R/Re, body, T, mem)
end
# size
N=2^7
params = [(5N÷2,N,N) (15N÷4,3N÷2,3N÷2) (5N,2N,2N)]
# for domain in params
#     # make the sim
#     sim = make_sphere(domain;N=N,R=44,mem=CUDA.CuArray)
#     sim_step!(sim;remeasure=false)
#     time = 0.1:0.1:200
#     # run
#     drag = map(time) do t
#         sim_step!(sim,t;remeasure=false)
#         @show t
#         -WaterLily.pressure_force(sim)[1]/(0.5π*sim.L^2)
#     end
#     t = map(i->string(i),domain)
#     jldsave("sphere_$(t[1])x$(t[2])x$(t[3]).jld2"; p=Array(sim.flow.p),
#             u=Array(sim.flow.u), time=time, drag=drag)
# end

using CairoMakie,ColorSchemes,FileIO
# let
#     small = jldopen(joinpath(@__DIR__,"sphere_320x128x128.jld2"))
#     medium = jldopen(joinpath(@__DIR__,"sphere_480x192x192.jld2"))
#     large = jldopen(joinpath(@__DIR__,"sphere_640x256x256.jld2"))
#     blues = colormap("Blues", 8)[3:end] # Biot savart

#     blockage = plot(ylims=(0.25,0.5), xlims=(0,1),
#                     xlabel="πR²/A", lw=0, legend=:bottomright, size=(400,400),
#                     right_margin=Plots.Measures.Length(:mm, 5),
#                     ylabel="Mean drag coefficient")
#     drag = plot(xlims=(0,200),ylims=(0.25,0.5),
#                 xlabel="Convective time", legend=:bottomright, size=(400,400),
#                 right_margin=Plots.Measures.Length(:mm, 5),
#                 ylabel="Drag coefficient")
#     labels = ["3.6Dx1.5Dx1.5D","5.5Dx2.2Dx2.2D","7.2Dx2.9Dx2.9D"]
#     for (i,case,D) in zip([2,4,6],[small,medium,large],[128,192,256])
#         t = case["time"]; idx = t .> 100
#         plot!(drag, t, case["drag"], label=labels[i÷2], c=blues[i])
#         fx, t = case["drag"][idx], t[idx]
#         CD_mean = sum(fx[2:end].*diff(t))/sum(diff(t))
#         println("▷ ΔT [CTU] = $(t[end]-t[1])")
#         println("▷ CD mean = $CD_mean")
#         scatter!(blockage, [(π*44^2)/D^2], [CD_mean], label=:none, c=blues[i])
#     end
#     for (i,pl) in enumerate([blockage drag])
#         hline!(pl, [0.394], linestyle=:dash, color=:black, label=ifelse(i==1,"Rodriguez et al. (DNS)",:none))
#         hline!(pl, [0.355], linestyle=:dot, color=:grey, label=ifelse(i==1,"Yun et al. (LES)",:none))
#     end
#     savefig(drag,"drag.png")
#     savefig(blockage,"validation_sphere.png")
# end
# let
small = jldopen("sphere_320x128x128.jld2")
medium = jldopen("sphere_480x192x192.jld2")
large = jldopen("sphere_640x256x256.jld2")
blues = get(ColorSchemes.Blues, range(0.0, 1.0, length=8))[3:end]
img = load("sphere3_zoom.png")
f = Figure(size=(1000,300), figure_padding=5)
drag = Axis(f[2, 1], xlabel="Convective time", ylabel="Drag coefficient")
labels = ["3.6Dx1.5Dx1.5D","5.5Dx2.2Dx2.2D","7.2Dx2.9Dx2.9D"]
lines = []; labs = []
for (i,case,D) in zip([2,4,6],[small,medium,large],[128,192,256])
    t = case["time"]; idx = t .> 100
    lines!(drag, t, case["drag"], color=blues[i],alpha=0.6)
    fx, t = case["drag"][idx], t[idx]
    CD_mean = sum(fx[2:end].*diff(t))/sum(diff(t))
    println("▷ ΔT [CTU] = $(t[end]-t[1])")
    println("▷ CD mean = $CD_mean")
    l=hlines!(drag, [CD_mean], color=blues[i], linewidth=2)
    push!(lines, l)
    push!(labs, (π*44^2)/D^2)
end
# image is of size
nx,ny=size(img')
# domain is 3.6Dx1.5D, this means
Flow = Axis(f[1:2, 2], aspect = DataAspect(), xlabel="X/R", ylabel="Y/R", xticks=(0:Int(nx÷3.6):nx,["0","2","4","6"]),
            yticks=(0:ny÷2:ny,["-1.5","0","1.5"]))
image!(Flow, img')
hlines!(drag, [0.394], linestyle=:dash, color=:black, label="Rodriguez et al. (DNS)")
hlines!(drag, [0.355], linestyle=:dot, color=:black, label="Yun et al. (LES)")
hlines!(blockage, [0.394], linestyle=:dash, color=:black, label="Rodriguez et al. (DNS)")
hlines!(blockage, [0.355], linestyle=:dot, color=:black, label="Yun et al. (LES)")
xlims!(drag,0,200); ylims!(drag,0.3,0.5)
axislegend(drag, position=:rt, labelsize=12, rowgap=0)
Legend(f[1,1], lines, map(i->"=$(round(labs[i];digits=3))",1:3), "Blockage ratio πR²/A", labelsize=12, nbanks=3,
        framevisible=false, position=:rt, colgap=5, titlegap=0)
# hidespines!(ax.axis); hidedecorations!(ax.axis)
colsize!(f.layout, 1, Relative(1/4))
rowsize!(f.layout, 1, Auto(0.15))
rowgap!(f.layout, 1, Relative(0.02))
save("validation_sphere.png", f)
f
