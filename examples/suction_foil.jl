using WaterLily,ParametricBodies,StaticArrays,Plots,BiotSavartBCs,CUDA
function wing(L=2^6; U=1, Re=1e4, α=π/20, x₀=0., wdth=0.02, Γ=0, mem=Array)
    # Define a wing section with a slot
    cps = SA_F32[L/2 L/2 L/2  L/4   0 -L/2 -L/2
                 0   1.5 1.5 L/20 L/4  L/8    0] .+ SA_F32[x₀*L;0]
    s,c = sincos(Float32(α))
    wdth = Float32(wdth*L)
    function map(x,t)
        ξ=SA[c -s; s c]*(SA[x[1],x[2]]-L*SA_F32[0.625-x₀,0.5])
        SA[ξ[1],abs(ξ[2])]
    end
    section = ParametricBody(BSplineCurve(cps,degree=3);map,ndims=3)
    slot = AutoBody((x,t)->abs(x[1])-wdth,map)
    body = section-slot # set difference
    
    # User-Defined-Function adding a body force at the slot center
    function udf(a,t,kwargs...) 
        @fastmath @inline function body_force!(f,Ii,t)
            last(Ii) == 3 && return 
            ξ = map(WaterLily.loc(Ii),t)        # slot-based coord
            d = √min(ξ'ξ/4wdth^2,1)             # scaled distance from center
            amp = Float32(Γ*cos(d*0.5π)/wdth^2) # force amplitude (scale by slot width²)
            f[Ii] += -amp*SA[s c][last(Ii)]     # rotated force component
        end
        WaterLily.@loop body_force!(a.f,Ii,t) over Ii ∈ CartesianIndices(a.f)
    end
    @show (2L,L,L÷4)
    return BiotSimulation((2L,L,L÷4),(U,0,0),L;nonbiotfaces=(-3,3),ν=U*L/Re,body,T=Float32,mem),udf
end

import WaterLily:sim_gif!,flood,body_plot!
function sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                    remeasure=false,plotbody=false,udf=nothing,kv...)
    t₀ = round(sim_time(sim))
    @time @gif for tᵢ in range(t₀,t₀+duration;step)
        sim_step!(sim,tᵢ;remeasure,udf)
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U        
        flood(sim.flow.σ[R]; kv...)
        plotbody && body_plot!(sim;R)
        verbose && println("tU/L=",round(tᵢ,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
function flood(f::AbstractArray;shift=(0.,0.),cfill=:RdBu_11,clims=(),levels=10,kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    Plots.contourf(axes(f,1).+shift[1],axes(f,2).+shift[2],(@view f[:,:,1])' |>Array,
                   linewidth=0, levels=levels, color=cfill, clims = clims,
                   aspect_ratio=:equal; kv...)
end
function body_plot!(sim;levels=[0],lines=:black,R=inside(sim.flow.p))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    contour!((@view sim.flow.σ[R][:,:,1])' |>Array;levels,lines)
end

L=2^7;sim,udf = wing(L,Γ=2,Re=1e5,mem=CuArray); sim_step!(sim;udf,remeasure=false)
sim_gif!(sim;duration=20,udf,clims=(-50,50),plotbody=true,R=CartesianIndices((2:2L,2:L,L÷8:L÷8)))
