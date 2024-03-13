using BiotSavartBCs
using WaterLily
using StaticArrays
using CUDA
include("TwoD_plots.jl")

function ellipse(D,n,m,Λ=5.0;A₀=1.0,St=0.6,U=1,Re=100,T=Float32,mem=Array)
    h₀=T(A₀*D/2); ω=T(2π*St*U/D)
    function sdf(x,t)
        √sum(abs2,SA[x[1]/Λ,x[2]])-D÷2/Λ
    end
    function map(x,t)
        x .- SA[n*D÷4,m*D÷2-h₀*sin(ω*t)]
    end
    Simulation((n*D,m*D), (U,0), D; body=AutoBody(sdf,map), ν=U*D/Re, T, mem)
end
# do we use Biot-Savart?
for use_biotsavart in [false]
    (n,m) = use_biotsavart ? (6,4) : (32,24)
    sim = use_biotsavart ? ellipse(64,6,4,5;mem=CUDA.CuArray) : ellipse(64,30,20,5;mem=CUDA.CuArray); 
    ω = use_biotsavart ? MLArray(sim.flow.σ) : nothing
    t₀,duration,tstep = round(sim_time(sim)), 100., 0.1
    R = use_biotsavart ? inside(sim.flow.p) : CartesianIndices((386:769, 514:769)) # show the same part
    anim = @animate for tᵢ in range(t₀,t₀+duration,step=tstep)
        while sim_time(sim) < tᵢ
            measure!(sim)
            use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : mom_step!(sim.flow,sim.pois)
        end
        println("tU/L=",round(sim_time(sim),digits=4),
                ", Δt=",round(sim.flow.Δt[end],digits=3))
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ[R],clims=(-10,10))
    end
    gif(anim, "deflected_wake_$(n)L_$(m)L_$use_biotsavart.gif")
end