using WaterLily,StaticArrays,CUDA

function make_sim(; N=128, R=32, a=0.5, U=1, Re=1e3, mem=Array)
    disk(x,t) = (r = √(x[2]^2+x[3]^2); √(x[1]^2+(r-min(r,R))^2)-1.5)
    s(t) = ifelse(t<U/a,0.5a*t^2,U*(t-0.5U/a)) # displacement
    move(x,t) = x - SA[R*s(t/R)-R,0,0] .- N/2  # move the center
    Simulation((N,N,N), (0,0,0), R; U,ν=U*R/Re, body=AutoBody(disk,move), mem)
end
function make_sim_square(; N=128, R=32, a=0.5, U=1, Re=1e3, mem=Array)
    square(x,t) = (y = x.-SA[0,clamp(x[2],-R,R),clamp(x[3],-R,R)]; √sum(abs2,y)-1.5)
    s(t) = ifelse(t<U/a,0.5a*t^2,U*(t-0.5U/a)) # displacement
    move(x,t) = x - SA[R*s(t/R)-R,0,0] .- N/2  # move the center
    Simulation((N,N,N), (0,0,0), R; U,ν=U*R/Re, body=AutoBody(square,move), mem)
end

include("TwoD_plots.jl")
CIs = CartesianIndices
N = 2^8; R = N/3
domain = (2:N+1,2:N+1,N÷2+1)
sim = make_sim_square(mem=CUDA.CuArray;N,R);
for t in 1:3
    @time sim_step!(sim,t)
    flood(sim.flow.p[CIs(domain[1:2]),domain[3]]|>Array,clims=(-2,2))
    savefig("Disk_reflect_press_$(t).png")
    WaterLily.@loop sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U over I ∈ CIs(domain)
    flood(sim.flow.σ[CIs(domain[1:2]),domain[3]]|>Array,clims=(-20,20))
    savefig("Disk_reflect_omega_$(t).png")
end