using WaterLily,StaticArrays,CUDA,BiotSavartBCs
function circ(D,m;Re=550,U=1,shift=0,makeSim=BiotSimulation,kwargs...)
    body = AutoBody((x,t)->hypot(x[1]-m÷2,x[2]-m÷2-shift)-D÷2)
    makeSim((2m,m), (U,0), D; body, ν=U*D/Re, kwargs...)
end

using Plots,TypedTables
function update_Ix!(sim,t₀)
    sim_step!(sim,t₀;remeasure=false)
    ωy = sim.flow.σ
    @inside ωy[I] = WaterLily.curl(3,I,sim.flow.u)*WaterLily.loc(2,I)[2]
    (t=sim_time(sim),Ix=sum(ωy[inside(ωy)])/sim.L^2)
end
with_drag(data) = map(2:length(data)-1) do i
    (t=data.t[i],Ix=data.Ix[i],Cd=-2*(data.Ix[i+1]-data.Ix[i-1])/(data.t[i+1]-data.t[i-1]))
end |> Table

# Published data
koumoutsakos = hcat([[0.043110735418427915, 1.4994160125588691],
                [0.06339814032121693, 1.213695930443183],
                [0.1344040574809806, 0.8673899287525665],
                [0.5029585798816567, 0.7552846274604519],
                [0.8613693998309375, 1.0203250815118947],
                [1.256973795435333, 1.269354908827436],
                [1.730346576500422, 1.272648230889989],
                [2.2104818258664394, 1.175368192247313],
                [2.656804733727809, 1.108954957130781],
                [3.1470836855452236, 1.053957734573119],
                [3.6238377007607756, 1.0138215191402002],
                [4.097210481825865, 0.9816862697741817],
                [4.567202028740489, 0.9586948436179206]]...)
Gillis = hcat([[0.04311073541842814, 1.3588445839874406],
          [0.06339814032121693, 1.1211245018717544],
          [0.11411665257819092, 0.9245385822968241],
          [0.3237531699070164, 0.7164786861490161],
          [0.5807269653423499, 0.8089766936360349],
          [0.7633136094674544, 0.9483530974519988],
          [1.0304311073541839, 1.1585624924526021],
          [1.4666103127641579, 1.301295012679628],
          [1.926458157227387, 1.2337350561526386],
          [2.376162299239221, 1.1513208549692062],
          [2.8901098901098896, 1.0826025842289577],
          [3.3770076077768367, 1.0321777563096242],
          [3.867286559594252, 0.997751962323391],
          [4.310228233305155, 0.970196836130902],
          [4.8072696534235, 0.9437691100108682],
          [5.206255283178354, 0.9253693998309381],
          [5.598478444632289, 0.9092573360705228]]...)

function small_time(t;Re=550, k=4√(t/Re))
    t₁ = 2.257 + k - 0.141k^2 + 0.031k^3
    t₂ = (8.996 - 41k + 143.8k^2 + 45.4k^3)*t^2
    t₃ = (20.848 - 314.08k - 1851.36k^2 - 194.8k^3)*t^4
    t₄ = (28.864 + 6.272k)*t^6
    return π/√(Re*t)*(t₁ + t₂ + t₃ + t₄)
end

# Generate present method's data
D,m_2,m_1 = 128,(4,5,6),(5,10,20,40)
@time biot = map(m_2) do m
    sim = circ(D,m*D÷2,shift=D÷8,mem=CUDA.CuArray)
    [update_Ix!(sim,t₀) for t₀ in 0:0.02:min(m+1,6)] |>Table |> with_drag
end;
@time refl = map(m_1) do m
    sim = circ(D,m*D÷2,mem=CUDA.CuArray,makeSim=Simulation)
    [update_Ix!(sim,t₀) for t₀ in 0:0.02:6] |>Table |> with_drag
end;

# Plot
begin
    scatter(koumoutsakos[1,:],koumoutsakos[2,:],label="Koumoutsakos and Leonard",m=(5, :square, :white, stroke(1, color)))
    scatter!(Gillis[1,:],Gillis[2,:],label="Gillis et al.",m=(5, :white, stroke(1, color)),c=:black,marker=:circle)
    plot!(collect(0:0.01:0.35),small_time.(0:0.01:0.35),ls=:dash,c=:black,label="Theoretical curve")
    bmap = palette(:Blues,5)
    for (m,dat) in zip(m_2,biot)
        m2 = m/2
        mod(m2,1)==0 && (m2 = Int(m2))
        plot!(dat.t,dat.Cd,c=bmap[m-2],label="Present, D/W=1/$m2")
    end
    rmap = palette(:Reds,6)
    for (i,m,dat) in zip(1:4,m_1,refl)
        m2 = m/2
        mod(m2,1)==0 && (m2 = Int(m2))
        plot!(dat.t,dat.Cd,c=rmap[i+1],ls=:dashdot,label="Reflection, D/W=1/$m2")
    end
end; plot!(dpi=300,size=(550,340),ylabel="Drag coefficient",xlabel="convective time",ylims=(0,1.6),legend=:bottomright)
savefig("ImpCircle_Cd.png")

# Flow plot
using Measures,Plots,PyPlot
D=128
sim = circ(D,2*D);sim_step!(sim,4,remeasure=false);
ω = sim.flow.σ
@inside ω[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
pyplot(dpi=300)
contourf(clamp.(ω[inside(ω)],-6,6)';aspect_ratio=:equal,
    framestyle=nothing,axis=nothing,size=(355,200),
    cbar=:top,c=:RdBu,clims=(-6,6),lw=0,levels=(union(-6:-1,1:6)))
savefig("ImpCircle_4_vort.png")
