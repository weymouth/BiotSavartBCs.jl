# momentum step using bio_project
import WaterLily: scale_u!,conv_diff!,BDIM!,CFL,accelerate!,time,udf!,@log
function biot_mom_step!(a::Flow{N},b,ω...;λ=quick,udf=nothing,fmm=true,kwargs...) where N
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    U = BCTuple(a.uBC,t₁,N); # BCs at t₁
    # predictor u → u'
    @log "p"
    conv_diff!(a.f,a.u⁰,a.σ,λ,ν=a.ν)
    udf!(a,udf,t₀; kwargs...)
    accelerate!(a.f,t₀,a.g,a.uBC)
    BDIM!(a);
    biot_project!(a,b,ω...,U;fmm) # new
    # corrector u → u¹
    @log "c"
    conv_diff!(a.f,a.u,a.σ,λ,ν=a.ν)
    udf!(a,udf,t₁; kwargs...)
    accelerate!(a.f,t₁,a.g,a.uBC)
    BDIM!(a); scale_u!(a,0.5)
    biot_project!(a,b,ω...,U;fmm,w=0.5) # new
    push!(a.Δt,CFL(a))
end
BCTuple(f::Function,t::T,N) where T = ntuple(i->f(i,zero(SVector{N,T}),t),N)
BCTuple(f::Tuple,t,N) = f

# project using biot BCs
import WaterLily: Vcycle!,smooth!
function biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ω,x₀,tar,ftar,U;fmm=true,w=1,tol=1e-4,itmx=32) where n
    dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
    apply_grad_p!(a.u,ω,a.p,a.μ₀) # Apply u-=μ₀∇p & ω=∇×u
    x₀ .= a.p; fill!(a.p,0)       # x₀ holds p solution
    biotBC!(a.u,U,ω,tar,ftar;fmm) # Apply domain BCs

    # Set residual
    b = ml_b.levels[1]; b.r .= 0
    @inside b.r[I] = ifelse(b.iD[I]==0,0,WaterLily.div(I,a.u))
    fix_resid!(b.r,a.u,tar[1]) # only fix on the boundaries

    nᵖ,nᵇ,r₂ = 0,0,L₂(b)
    @log ", $nᵖ, $(WaterLily.L∞(b)), $r₂, $nᵇ\n"
    while nᵖ<itmx
        rtol = max(tol,0.1r₂)
        while nᵖ<itmx
            Vcycle!(ml_b); smooth!(b)
            r₂ = L₂(b); nᵖ+=1
            r₂<rtol && break
        end
        apply_grad_p!(a.u,ω,a.p,a.μ₀)   # Update u,ω
        x₀ .+= a.p; fill!(a.p,0)        # Update solution
        biotBC_r!(b.r,a.u,U,ω,tar,ftar;fmm) # Update BC+residual
        r₂ = L₂(b); nᵇ+=1
        @log ", $nᵖ, $(WaterLily.L∞(b)), $r₂, $nᵇ\n"
        r₂<tol && break
    end
    push!(ml_b.n,nᵖ)
    pflowBC!(a.u)  # Update ghost BCs (domain is already correct)
    a.p .= x₀/dt   # copy-scaled pressure solution
end

# Apply u-=μ₀∇p & ω=∇×u
function apply_grad_p!(u,ω,p,μ₀)
    @loop u[Ii] -= μ₀[Ii]*∂(last(Ii),front(Ii),p) over Ii ∈ inside_u(u)
    fill_ω!(ω,u)
end
