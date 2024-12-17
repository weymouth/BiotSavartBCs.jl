using WaterLily,StaticArrays,CUDA,BiotSavartBCs,WriteVTK
using JLD2

y⁺(a::AbstractSimulation) = √(0.026/(2*(sim.L*sim.U/sim.flow.ν)^(1/7)))/sim.flow.ν
function make_sim_acc(; N=128, R=32, a=0.5, U=1, Re=1e3, mem=Array, use_biotsavart=false)
    disk(x,t) = (z=x-SA[-R,0,0].-N/2; y=z.-SA[0,clamp(z[2],-R,R),clamp(z[3],-R,R)]; √sum(abs2,y)-1.5)
    Ut(i,t::T) where T = i==1 ? convert(T,min(a*t/R,U)) : zero(T) # velocity BC
    body = AutoBody(disk)
    use_biotsavart && return BiotSimulation((N,N,N), Ut, R; U, ν=U*R/Re, body, mem)
    Simulation((N,N,N), Ut, R; U, ν=U*R/Re, body, mem)
end

# make a writer with some attributes, need to output to CPU array to save file (|> Array)
import WaterLily: @loop,ω,λ₂
vort(a) = (@loop sim.flow.f[I,:] .= ω(I,sim.flow.u) over I in inside(sim.flow.p); a.flow.f |> Array)
_body(a) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); a.flow.σ |> Array)
lamda(a) = (@inside a.flow.σ[I] = λ₂(I, a.flow.u); a.flow.σ |> Array)

custom_attrib = Dict("ω"=>vort, "b"=>_body, "λ₂"=>lamda)
# make the writer
writer = vtkWriter("Disk_high_Re_3"; attrib=custom_attrib,
                   dir="vtk_data")

# dimensions
N = 3*2^7; R = N/4
use_biotsavart = true
sim = make_sim_acc(mem=CUDA.CuArray;N,R,Re=125_000,use_biotsavart);

@show y⁺(sim)
forces = []
for t in range(0,10;step=0.02)#1:6
    while sim_time(sim)<t
        sim_step!(sim;remeasure=false)# update the body compute at timeNext
        f = -2WaterLily.pressure_force(sim)/R^2
        push!(forces,[sim_time(sim),f[1]])
    end
    write!(writer,sim);
    @show t
    flush(stdout)
end
jldsave("disk_high_re_forces.jld2"; f=forces)
close(writer)