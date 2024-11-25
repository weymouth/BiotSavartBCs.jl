# momentum step using bio_project
import WaterLily: scale_u!,conv_diff!,BDIM!,CFL,accelerate!,time,BCTuple
function biot_mom_step!(a::Flow{N},b,ω...) where N
    a.u⁰ .= a.u; scale_u!(a,0)
    # predictor u → u'
    U = BCTuple(a.U,@view(a.Δt[1:end-1]),N)
    conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν);
    accelerate!(a.f,@view(a.Δt[1:end-1]),a.g,a.U)
    BDIM!(a);
    biot_project!(a,b,ω...,U) # new
    # corrector u → u¹
    U = BCTuple(a.U,a.Δt,N)
    conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    accelerate!(a.f,a.Δt,a.g,a.U)
    BDIM!(a); scale_u!(a,0.5)
    biot_project!(a,b,ω...,U,w=0.5) # new
    push!(a.Δt,CFL(a))
end

# project using biot BCs
import WaterLily: residual!,Vcycle!,smooth!
function biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ω,x₀,tar,ftar,U;w=1,log=false,tol=1e-5,itmx=8) where n
    dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
    apply_grad_p!(a.u,ω,a.p,a.μ₀) # Apply u-=μ₀∇p & ω=∇×u
    x₀ .= a.p; fill!(a.p,0)       # x₀ holds p solution
    biotBC!(a.u,U,ω,tar,ftar)     # Apply domain BCs

    b = ml_b.levels[1]
    @inside b.z[I] = WaterLily.div(I,a.u)   # Set σ=∇⋅u
    residual!(b); nᵖ,nᵇ,r₂ = 0,0,L₂(b)
    while nᵖ<itmx
        rtol = max(tol,0.1r₂)
        while r₂>rtol && nᵖ<itmx
            Vcycle!(ml_b); smooth!(b)
            r₂ = L₂(b); nᵖ+=1
        end
        apply_grad_p!(a.u,ω,a.p,a.μ₀)   # Update u,ω
        x₀ .+= a.p; fill!(a.p,0)        # Update solution
        biotBC_r!(b.r,a.u,U,ω,tar,ftar) # Update BC+residual
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

# update domain velocity and residual
function biotBC_r!(r,u,U,ω,tar,ftar)
    interaction!(ω,ftar)
    project!(ω,tar)
    @vecloop _update_resid!(r,u,U,ω[1],ω[2],Ii) over Ii ∈ tar[1]
    fix_resid!(r)
end 
Base.@propagate_inbounds @fastmath function _update_resid!(r,u,U,a,b,Ii)
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    uI = lower ? Ii+δ(i,Ii) : Ii

    # Update velocity and residual
    uₙ = U[i]+(a[Ii]+0.25f0project(Ii,b))/Float32(4π)
    uₙ⁰ = u[uI]; u[uI] = uₙ
    sgn = lower ? -1 : 1; r[I-sgn*δ(i,I)] += sgn*(uₙ-uₙ⁰)
end

function fix_resid!(r)
    N = size(r); n = length(N); A(i) = 2prod(N.-2)/(N[i]-2)
    res = sum(r)/sum(A,1:n)
    for i ∈ 1:n
        @loop r[I] -= res over I ∈ WaterLily.slice(N.-1,2,i,2)
        @loop r[I] -= res over I ∈ WaterLily.slice(N.-1,N[i]-1,i,2)
    end
end 
