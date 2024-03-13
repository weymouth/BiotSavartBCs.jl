using BiotSavartBCs
using WaterLily
using CUDA
using StaticArrays
y⁺(a::Simulation) = √(0.026/(2*(sim.L*sim.U/sim.flow.ν)^(1/7)))/sim.flow.ν
include("TwoD_plots.jl")
#Quantify domain sensitivity
circ(D,n,m;Re=200,U=1,mem=Array) = Simulation((n*D,m*D), (U,0), D;
                                               body=AutoBody((x,t)->√sum(abs2,x-SA{Float32}[m*4D÷9,m*D÷2])-D÷2),ν=U*D/Re,mem)
# do we use Biot-Savart?
use_biotsavart = true
sim = circ(1024,2,2;Re=100_000,mem=CUDA.CuArray);
@show y⁺(sim)
ω = MLArray(sim.flow.σ)
t₀=0.0;duration=2.5;step=0.05
@gif for tᵢ in range(t₀,t₀+duration;step)
    while WaterLily.time(sim) < tᵢ*sim.L/sim.U
        use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
    end
    @show sim_time(sim)
    @inside sim.flow.σ[I] = BiotSavartBCs.centered_curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ|>Array,clims=(-10,10),dpi=300)
end

