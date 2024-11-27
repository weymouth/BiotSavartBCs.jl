# momentum step using bio_project
import WaterLily: scale_u!,conv_diff!,BDIM!,CFL,accelerate!,time,BCTuple
function biot_mom_step!(a::Flow{N},b,ω...;fmm=true) where N
    a.u⁰ .= a.u; scale_u!(a,0)
    # predictor u → u'
    U = BCTuple(a.U,@view(a.Δt[1:end-1]),N)
    conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν);
    accelerate!(a.f,@view(a.Δt[1:end-1]),a.g,a.U)
    BDIM!(a);
    biot_project!(a,b,ω...,U;fmm) # new
    # corrector u → u¹
    U = BCTuple(a.U,a.Δt,N)
    conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    accelerate!(a.f,a.Δt,a.g,a.U)
    BDIM!(a); scale_u!(a,0.5)
    biot_project!(a,b,ω...,U;fmm,w=0.5) # new
    push!(a.Δt,CFL(a))
end

# project using biot BCs
import WaterLily: residual!,Vcycle!,smooth!
function biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ω,x₀,tar,ftar,U;fmm=true,w=1,log=false,tol=1e-5,itmx=8) where n
    dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
    apply_grad_p!(a.u,ω,a.p,a.μ₀) # Apply u-=μ₀∇p & ω=∇×u
    x₀ .= a.p; fill!(a.p,0)       # x₀ holds p solution
    biotBC!(a.u,U,ω,tar,ftar;fmm) # Apply domain BCs

    b = ml_b.levels[1]
    @inside b.r[I] = WaterLily.div(I,a.u)   # Set σ=∇⋅u
    fix_resid!(b.r,tar[1]) # only fix on the boundaries

    nᵖ,nᵇ,r₂ = 0,0,L₂(b)
    while nᵖ<itmx
        rtol = max(tol,0.1r₂)
        while r₂>rtol && nᵖ<itmx
            Vcycle!(ml_b); smooth!(b)
            r₂ = L₂(b); nᵖ+=1
        end
        apply_grad_p!(a.u,ω,a.p,a.μ₀)   # Update u,ω
        x₀ .+= a.p; fill!(a.p,0)        # Update solution
        biotBC_r!(b.r,a.u,U,ω,tar,ftar;fmm) # Update BC+residual
        r₂ = L₂(b); nᵇ+=1
        log && @show nᵖ,nᵇ,r₂
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
