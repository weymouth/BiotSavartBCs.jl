using WaterLily,StaticArrays,BiotSavartBCs,CUDA,WriteVTK

# make a writer with some attributes, need to output to CPU array to save file (|> Array)
vtk_velocity(a::AbstractSimulation) = a.flow.u |> Array;
vtk_pressure(a::AbstractSimulation) = a.flow.p |> Array;
vtk_body(a::AbstractSimulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); a.flow.σ |> Array;)
vtk_lambda(a::AbstractSimulation) = (@inside a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u); a.flow.σ |> Array;)

# starting  from rest
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=0.5)

# overwrite to apply symmetric BiotBCs
import BiotSavartBCs: interaction,_interaction!
@inline function _interaction!(ml,lT)
    (l,T) = lT     # level & target
    n = length(ml) # total levels
    ω = ml[l]      # vorticity sources

    T⁺,sgn = image(T,size(ω),-2) # image target & sign
    ml[l][T] = interaction(ω,T,l,n)+sgn*interaction(ω,T⁺,l,n)
end

# make simulation following Dickinson's setup
function Dickinson(L=64;U=1,Re=5e2,ε=0.5f0,thk=2ε+√3,AR=2,mem=Array,T=Float32)

    # ellipse sdf
    function sdf(x,t)
        # offset it to put origin at tip, and stretch it by AR. remove radius for sdf
        √sum(abs2, SA[x[1], (x[2]-1.5f0L/2.f0)/AR, 0]) - L/2/AR
    end

    # the mapping
    function map(x,t)
        # Dickinson kinemtics
        _α = π/2.f0 - π/4.f0*sin(π*t/L)  # positive pitch increase AoA
        _ϕ = 0.35f0π*cos(π*t/L)      # positive to the rear of the mosquito
        # rotation mmatrix
        Ry = SA[cos(_α) 0 sin(_α); 0 1 0; -sin(_α) 0 cos(_α)] # alpha
        Rz = SA[cos(_ϕ) -sin(_ϕ) 0; sin(_ϕ) cos(_ϕ) 0; 0 0 1] # phi
        return Ry*Rz*(x .- SA[1.5f0L,0,0.5f0L]) # the order matters
    end

    # Build the mosquito from a mapped ellipsoid and two plane that trim it to the correct thickness
    ellipsoid = AutoBody(sdf, map)
    upper_lower = AutoBody((x,t)->(abs(x[3])-thk/2.f0), map)
    body = ellipsoid ∩ upper_lower # intersection of sets

    # Return initialized simulation, the y=0 plane is the symmetry plane
    return BiotSimulation((3L,2L,3L),(0,0,0),L;ν=U*L/Re,U,body,mem,T,nonbiotfaces=(-2,))
end

# make a simulation
sim = Dickinson(32;mem=CuArray,T=Float32)

# make a vtk writer
custom_attrib = Dict("u"=>vtk_velocity, "p"=>vtk_pressure, "d"=>vtk_body, "λ2"=>vtk_lambda)
# make the writer
writer = vtkWriter("Dickinson"; attrib=custom_attrib)

# a simulation
t₀ = sim_time(sim); duration = 12; tstep = 0.05 # print time
forces = []# empty list to store forces

# simulate
@time for tᵢ in range(t₀,t₀+duration;step=tstep)
    while sim_time(sim) < tᵢ
        #update the body and flow
        sim_step!(sim;remasure=true)
        # compute and save pressure forces
        force = -2WaterLily.pressure_force(sim)
        push!(forces,[sim_time(sim),force...])
    end
    # tsave vti files
    save!(writer,sim);
    # print time
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
forces = reduce(vcat,forces')
println("Done...")
close(writer)