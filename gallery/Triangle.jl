using WaterLily,StaticArrays,CUDA,BiotSavartBCs
using LinearAlgebra: ×,⋅
using WriteVTK

function make_sim_acc(; N=128, R=32, a0=0.5, U=1, Re=1e3, mem=Array)
    # triangle Position
    a,b,c = SA[0,-0.75R,0],SA[0,R,R],SA[0,R,-R]
    dot2(x) = x ⋅ x

    # sdf
    function triangle(p,t)
        ba = b - a; pa = p - a;
        cb = c - b; pb = p - b;
        ac = a - c; pc = p - c;
        nor = ba × ac;
        
        return √(
            (sign((ba × nor) ⋅ pa) +
             sign((cb × nor) ⋅ pb) +
             sign((ac × nor) ⋅ pc)<2)
            ?
            min( min(
                dot2(ba*clamp((ba ⋅ pa)/dot2(ba),0,1)-pa),
                dot2(cb*clamp((cb ⋅ pb)/dot2(cb),0,1)-pb) ),
                dot2(ac*clamp((ac ⋅ pc)/dot2(ac),0,1)-pc) )
                :
            (nor ⋅ pa)^2/dot2(nor) )-(1+√3)/2
    end
    map(x,t) = x-SA[N/2-R,N/2-R/4,N/2]
    
    Ut(i,t::T) where T = i==1 ? convert(T,a0*t/R+(1.0+tanh(31.4*(t/R-1.0/a0)))/2.0*(1-a0*t/R)) : zero(T) # velocity BC
    Simulation((N,N,N), Ut, R; U,ν=U*R/Re, body=AutoBody(triangle,map), mem)
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
writer = vtkWriter("Triangle_high_Re"; attrib=custom_attrib,
                   dir="/scratch/marinlauber/vtk_data")

CIs = CartesianIndices
#N = 3*2^7; R = N/4
N = 2^6; R = N/4
domain = (2:N+1,2:N+1,N÷2+1)
use_biotsavart = true
sim = make_sim_acc(mem=CuArray;N,R,Re=125_000);
ω = ntuple(i->MLArray(sim.flow.σ),3)

forces = []
for t in range(0,6;step=0.02)#1:6
    while sim_time(sim)<t #sim_step!(sim,t)
        measure!(sim,sum(sim.flow.Δt)) # update the body compute at timeNext
        biot_mom_step!(sim.flow,sim.pois,ω)
        #mom_step!(sim.flow,sim.pois)
    end
    write!(writer,sim);
    @show t
    flush(stdout)
end
close(writer)