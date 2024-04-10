using WaterLily,StaticArrays,CUDA,BiotSavartBCs,WriteVTK
y⁺(a::Simulation) = √(0.026/(2*(sim.L*sim.U/sim.flow.ν)^(1/7)))/sim.flow.ν
function make_sim_acc(; N=128, R=32, a=0.5, U=1, Re=1e3, mem=Array)
    disk(x,t) = (z=x-SA[-R,0,0].-N/2; y=z.-SA[0,clamp(z[2],-R,R),clamp(z[3],-R,R)]; √sum(abs2,y)-1.5)
    Ut(i,t::T) where T = i==1 ? convert(T,a*t/R+(1.0+tanh(31.4*(t/R-1.0/a)))/2.0*(1-a*t/R)) : zero(T) # velocity BC
    Simulation((N,N,N), Ut, R; U,ν=U*R/Re, body=AutoBody(disk), mem)
end
# make a writer with some attributes, need to output to CPU array to save file (|> Array)
vort(a::Simulation) = (@WaterLily.loop sim.flow.f[I,:] .= WaterLily.ω(I,sim.flow.u) over I in inside(sim.flow.p);
                       a.flow.f |> Array)
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); 
                                     a.flow.σ |> Array;)
lamda(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u);
                        a.flow.σ |> Array;)

custom_attrib = Dict(
    "vort" => vort,
    "Body" => _body,
    "Lambda" => lamda
)# this maps what to write to the name in the file
# make the writer
writer = vtkWriter("Disk_high_Re_3"; attrib=custom_attrib,
                   dir="/scratch/marinlauber/vtk_data")

CIs = CartesianIndices
N = 3*2^7; R = N/4
domain = (2:N+1,2:N+1,N÷2+1)
use_biotsavart = true
sim = make_sim_acc(mem=CUDA.CuArray;N,R,Re=125_000);
ω = ntuple(i->MLArray(sim.flow.σ),3)

@show y⁺(sim)
forces = []
for t in range(0,10;step=0.02)#1:6
    while sim_time(sim)<t #sim_step!(sim,t)
        measure!(sim,sum(sim.flow.Δt)) # update the body compute at timeNext
        biot_mom_step!(sim.flow,sim.pois,ω)
        f = -2WaterLily.∮nds(sim.flow.p,sim.flow.f,sim.body,sum(sim.flow.Δt[1:end-1]))/R^2
        push!(forces,[sim_time(sim),f[1]])
    end
    write!(writer,sim);
    @show t
    flush(stdout)
end
using JLD2
jldopen("disk_high_re_forces","w") do file
    mygroup = JLD2.Group(file,"case")
    mygroup["forces"] = forces
end
close(writer)