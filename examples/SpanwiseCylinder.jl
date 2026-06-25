using WaterLily, BiotSavartBCs, WriteVTK

function spanwise_cylinder(; D=32, Lz=D÷2, Re=500, U=1, T=Float32, mem=Array)
    body = AutoBody((x,t) -> √sum(abs2,(x.-D)[1:2]) - D/3)
    # Simulation((4D,2D,Lz), (U,0,0), D; ν=U*D/Re, body, T, mem, perdir=(3,))
    BiotSimulation((4D,2D,Lz), (U,0,0), D; ν=U*D/Re, body, T, mem, perdir=(3,), nimages=4)
end

vtk_sdf(a::Simulation)      = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); a.flow.σ |> Array)
vtk_velocity(a::Simulation) = a.flow.u |> Array
vtk_mu0(a::Simulation)      = a.flow.μ₀ |> Array
vtk_pressure(a::Simulation) = a.flow.p |> Array

attrib = Dict("d"=>vtk_sdf, "u"=>vtk_velocity, "μ₀"=>vtk_mu0, "p"=>vtk_pressure)

sim = spanwise_cylinder()
writer = vtkWriter("SpanwiseCylinder"; attrib)
# run
for t in range(0, 10.0; step=0.05)
    sim_step!(sim, t; remeasure=false)
    save!(writer, sim)
    @show t
end
close(writer)
