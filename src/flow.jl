# momentum step using bio_project
import WaterLily: scale_u!,conv_diff!,BDIM!,CFL,accelerate!,time
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
import WaterLily: residual!,Vcycle!,smooth!,L∞,restrictML,BCTuple
function biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ω,x₀,tar,ftar,U;w=1,log=false,tol=1e-6,itmx=32) where n    
    dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
    fill_ω!(ω,a.u,a.μ₀,a.p)       # Compute ω=∇×(u-μ₀∇p)
    biotBC!(a.u,U,ω,tar,ftar)     # Apply domain BCs

    b = ml_b.levels[1]
    @inside b.z[I] = div(I,a.u)   # Set σ=∇⋅u
    residual!(b); fix_resid!(b.r) # Set r=Ax-σ, and ensure sum(r)=0

    r₂ = L₂(b); nᵖ = 0
    while nᵖ<itmx
        x₀ .= b.x                 # Remember current solution
        Vcycle!(ml_b); smooth!(b) # Improve solution
        b.ϵ .= b.x .-x₀           # soln update: ϵ = x-x₀
        fill_ω!(ω,a.μ₀,b.ϵ)       # vort update: Δω = -∇×μ₀∇ϵ
        update_resid!(b.r,a.u,b.z,ω) # Update domain BC and resid
        r₂ = L₂(b); nᵖ+=1
        log && @show nᵖ,r₂
        r₂<tol && break
    end
    push!(ml_b.n,nᵖ)
    
    @loop a.u[Ii] -= b.L[Ii]*∂(last(Ii),front(Ii),b.x) over Ii ∈ inside_u(a.u)
    pflowBC!(a.u) # Update ghost BCs (domain is already correct)
    a.p ./= dt    # Rescale pressure
end

function fix_resid!(r)
    N = size(r); n = length(N); A(i) = 2prod(N.-2)/(N[i]-2)
    res = sum(r)/sum(A,1:n)
    for i ∈ 1:n
        @loop r[I] -= res over I ∈ slice(N.-1,2,i,2)
        @loop r[I] -= res over I ∈ slice(N.-1,N[i]-1,i,2)
    end
end 
