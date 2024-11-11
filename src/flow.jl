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
function biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ω,v,x₀,tar,ftar,U;w=1,log=true,tol=1e-5,itmx=8) where n
    dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
    fill_ω!(ω,a.u,v,a.μ₀,a.p)       # Compute ω=∇×(u-μ₀∇p)
    biotBC!(a.u,U,ω,tar,ftar)     # Apply domain BCs

    b = ml_b.levels[1]
    @inside b.z[I] = WaterLily.div(I,a.u)   # Set σ=∇⋅u
    residual!(b); # fix_resid!(b.r) # Set r=Ax-σ, and ensure sum(r)=0
    nᵖ = 0
    log && @show nᵖ,L₂(b)
    while nᵖ<itmx
        x₀ .= b.x                 # Remember current solution
        Vcycle!(ml_b); smooth!(b) # Improve solution
        log && @show nᵖ+1/2,L₂(b)
        b.ϵ .= b.x .-x₀           # soln update: ϵ = x-x₀
        fill_ω!(ω,v,a.μ₀,b.ϵ)     # vort update: Δω = -∇×μ₀∇ϵ
        update_resid!(b.r,a.u,ω,tar,ftar) # Update domain BC and resid
        r₂ = L₂(b); nᵖ+=1
        log && @show nᵖ,r₂
        r₂<tol && break
    end
    push!(ml_b.n,nᵖ)
    
    @loop a.u[Ii] -= b.L[Ii]*∂(last(Ii),front(Ii),b.x) over Ii ∈ inside_u(a.u)
    pflowBC!(a.u) # Update ghost BCs (domain is already correct)
    a.p ./= dt    # Rescale pressure
end

# update domain velocity and residual
function update_resid!(r,u,ω,tar,ftar)
    interaction!(ω,ftar)
    project!(ω,tar)
    @loop _update_resid!(r,u,ω[1],ω[2],Ii) over Ii ∈ tar[1]
    fix_resid!(r)
end 
Base.@propagate_inbounds @fastmath function _update_resid!(r,u,a,b,Ii)
    duₙ = (a[Ii]+0.25f0project(Ii,b))/Float32(4π) # correction
    I,i = front(Ii),last(Ii); lower = I.I[i]==1   # indices
    
    # Update velocity and residual
    u[lower ? Ii+δ(i,Ii) : Ii] += duₙ
    sgn = lower ? -1 : 1; r[I-sgn*δ(i,I)] += sgn*duₙ
end

function fix_resid!(r)
    N = size(r); n = length(N); A(i) = 2prod(N.-2)/(N[i]-2)
    res = sum(r)/sum(A,1:n)
    for i ∈ 1:n
        @loop r[I] -= res over I ∈ WaterLily.slice(N.-1,2,i,2)
        @loop r[I] -= res over I ∈ WaterLily.slice(N.-1,N[i]-1,i,2)
    end
end 
