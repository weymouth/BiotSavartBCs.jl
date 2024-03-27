using WaterLily
using BiotSavartBCs
using BiotSavartBCs: ml_restrict!,get_backend
using Plots
include("/home/marinlauber/Workspace/WaterLily.jl/examples/TwoD_plots.jl")
function vortexring(;L=32,mem=Array)

    # vortex ring parameters
    δ,R,Γ,Re = L/4/5,L/4,50.0,800

    function ω_ring(i,xyz)
        # move to domain center
        x,y,z = @. xyz - [L,L,L] + 0.5*WaterLily.δ(i,Val{3}()).I
        r,θ = √(x^2+y^2),atan(y,x)
        u_θ = Γ/π*exp(-(z^2+(r-R)^2)/δ^2)/δ^2
        i==1 && return u_θ*sin(θ)
        i==2 && return u_θ*cos(θ)
        return 0.0  # u_z
    end

    # Simulation parameters
    sim = Simulation((2L,2L,2L),(0,0,0),R;U=1,ν=Γ/Re,mem=mem)

    # fill vorticity field
    ω = ntuple(i->MLArray(sim.flow.σ),3)
    apply!(ω_ring,sim.flow.f);
    for i ∈ 1:3 # set each components
        ω[i][1] .= sim.flow.f[:,:,:,i]
        ml_restrict!(ω[i])# downsample each component
    end

    # update velocity field
    for i ∈ 1:3
        @WaterLily.loop sim.flow.u[I,i] = u_ω(i,I,ω) over I in inside(sim.flow.p)
    end
    return sim,ω
end

sim,ω = vortexring(;L=32);

t₀,duration,step=0.0,2.0,0.1
@gif for tᵢ in range(t₀,t₀+duration;step)
    # update until time tᵢ in the background
    while WaterLily.time(sim) < tᵢ*sim.L/sim.U
        mom_step!(sim.flow,sim.pois)
    end
    println("tU/L=",round(sim_time(sim),digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    @WaterLily.loop sim.flow.f[I,:] .= WaterLily.ω(I,sim.flow.u) over I ∈ inside(sim.flow.p)
    flood(sim.flow.f[:,33,:,2])
end
# sim_step!(sim,2;verbose=true)
# @WaterLily.loop sim.flow.f[I,:] .= WaterLily.ω(I,sim.flow.u) over I ∈ inside(sim.flow.p)
