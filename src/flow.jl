# momentum step using bio_project
import WaterLily: scale_u!,conv_diff!,BDIM!,CFL
function biot_mom_step!(a,b,ω)
    a.u⁰ .= a.u; scale_u!(a,0)
    # predictor u → u'
    conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν);
    BDIM!(a);
    biot_project!(a,b,ω) # new
    # corrector u → u¹
    conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    BDIM!(a); scale_u!(a,0.5)
    biot_project!(a,b,ω,w=0.5) # new
    push!(a.Δt,CFL(a))
end

# project using biot BCs
import WaterLily: residual!,Vcycle!,smooth!
function biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ω;w=1,log=false,tol=1e-3,itmx=32) where n    
    dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
    fill_ω!(ω,a.u,a.μ₀,a.p)       # Compute ω=∇×(u-μ₀∇p)
    biotBC!(a.u,a.U,ω)            # Apply domain BCs

    b = ml_b.levels[1]
    @inside b.z[I] = div(I,a.u)   # Set σ=∇⋅u
    residual!(b); fix_resid!(b.r) # Set r=Ax-σ, and ensure sum(r)=0

    r₂ = L₂(b); nᵖ = 0; x₀ = point(ω)
    while r₂>tol && nᵖ<itmx
        x₀ .= b.x                 # Remember current solution
        Vcycle!(ml_b); smooth!(b) # Improve solution
        b.ϵ .= b.x .-x₀; x₀ .= 0  # soln update: ϵ = x-x₀
        fill_ω!(ω,a.μ₀,b.ϵ)       # vort update: Δω = -∇×μ₀∇ϵ
        update_resid!(b.r,a.u,b.z,b.L,ω) # Update domain BC and resid
        r₂ = L₂(b); nᵖ+=1
        log && @show nᵖ,r₂
    end
    push!(ml_b.n,nᵖ)

    for i ∈ 1:n   # Project u -= μ₀∇p
        @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
    end
    pflowBC!(a.u) # Update ghost BCs (domain is already correct)
    a.p ./= dt    # Rescale pressure
end

# update domain velocity and residual
function update_resid!(r,u,u_ϵ,L,ω_ϵ)
    N,n = size_u(L); inN(I,N) = all(@. 2 ≤ I.I ≤ N-1)
    for i ∈ 1:n
        @loop (u_ϵ[I]=u_ω(i,I,ω_ϵ); u[I,i]+=u_ϵ[I]; inN(I,N) && (r[I]-=u_ϵ[I])) over I ∈ slice(N,2,i)
        @loop (u_ϵ[I]=u_ω(i,I,ω_ϵ); u[I,i]+=u_ϵ[I]; inN(I-δ(i,I),N) && (r[I-δ(i,I)]+=u_ϵ[I])) over I ∈ slice(N,N[i],i)
    end
    fix_resid!(r)
end 