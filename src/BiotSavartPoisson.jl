"""
    BiotSavartPoisson <: WaterLily.AbstractPoisson

Custom Poisson type for Biot-Savart boundary conditions. Wraps a `MultiLevelPoisson`
(which holds all pressure system state) and layers Biot-Savart state on top.

Fields:
- `ml`   : wrapped standard pressure solver
- `ω`    : multi-level vorticity (top level aliases `flow.f`)
- `tar`  : domain boundary target index arrays per multigrid level
- `ftar` : flattened target list for kernel dispatch
- `x₀`  : pressure solution accumulator across outer Newton iterations
- `fmm`  : use Fast Multi-level Method (`true`) or tree-sum (`false`)
"""
struct BiotSavartPoisson{T,S,V} <: WaterLily.AbstractPoisson{T,S,V}
    ml   :: MultiLevelPoisson{T,S,V}
    ω    :: NTuple
    tar  :: NTuple
    ftar :: AbstractVector
    x₀   :: AbstractArray
    fmm  :: Bool
    function BiotSavartPoisson(flow; nonbiotfaces=(), fmm=true, mem=Array)
        ml = MultiLevelPoisson(flow.p, flow.μ₀, flow.σ; perdir=flow.perdir)
        ω  = MLArray(flow.f)   # top level aliases flow.f — no copy
        tar  = mem.(collect_targets(ω, nonbiotfaces))
        ftar = flatten_targets(tar)
        x₀   = copy(flow.p)
        new{eltype(flow.p),typeof(flow.p),typeof(flow.μ₀)}(ml,ω,tar,ftar,x₀,fmm)
    end
end
WaterLily.update!(b::BiotSavartPoisson) = WaterLily.update!(b.ml)

# project using Biot-Savart BCs
import WaterLily: Vcycle!,smooth!
function WaterLily.mom_project!(a::WaterLily.AbstractFlow, b::BiotSavartPoisson, w, t, tol=1e-4,itmx=32)
    dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt

    apply_grad_p!(a.u,b.ω,a.p,a.μ₀)                       # Apply u-=μ₀∇p & ω=∇×u
    b.x₀ .= a.p; fill!(a.p,0)                            # x₀ holds p solution
    biotBC!(a.u,a.uBC,b.ω,b.tar,b.ftar,t;fmm=b.fmm)    # Apply domain BCs with fresh ω

    # Set residual
    p = b.ml.levels[1]; p.r .= 0
    @inside p.r[I] = ifelse(p.iD[I]==0,0,WaterLily.div(I,a.u))
    fix_resid!(p.r,a.u,b.tar[1]) # only fix on the boundaries

    nᵖ,nᵇ,r₂ = 0,0,L₂(p)
    @log ", $nᵖ, $(WaterLily.L∞(p)), $r₂, $nᵇ\n"
    while nᵖ<itmx
        rtol = max(tol,0.1r₂)
        while nᵖ<itmx
            Vcycle!(b.ml); smooth!(p)
            r₂ = L₂(p); nᵖ+=1
            r₂<rtol && break
        end
        apply_grad_p!(a.u,b.ω,a.p,a.μ₀)          # Update u,ω
        b.x₀ .+= a.p; fill!(a.p,0)               # Update solution
        biotBC_r!(p.r,a.u,a.uBC,b.ω,b.tar,b.ftar,t;fmm=b.fmm) # Update BC+residual
        r₂ = L₂(p); nᵇ+=1
        @log ", $nᵖ, $(WaterLily.L∞(p)), $r₂, $nᵇ\n"
        r₂<tol && break
    end
    push!(b.ml.n,nᵖ)
    pflowBC!(a.u)        # Update ghost BCs (domain is already correct)
    a.p .= b.x₀/dt       # copy-scaled pressure solution
end

# Apply u-=μ₀∇p & ω=∇×u
function apply_grad_p!(u,ω,p,μ₀)
    @loop u[Ii] -= μ₀[Ii]*∂(last(Ii),front(Ii),p) over Ii ∈ inside_u(u)
    fill_ω!(ω,u)
end
