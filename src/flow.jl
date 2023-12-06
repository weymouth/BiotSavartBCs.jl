# biotsavart momentum step
function biot_mom_step!(a,b,ml)
    a.u⁰ .= a.u; WaterLily.scale_u!(a,0)
    # predictor u → u'
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν);
    WaterLily.BDIM!(a);
    biot_project!(a,b,ml)
    # corrector u → u¹
    WaterLily.conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); WaterLily.scale_u!(a,0.5)
    biot_project!(a,b,ml,w=0.5)
    push!(a.Δt,WaterLily.CFL(a))
end
function biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ml_ω;w=1,log=false,tol=1e-3,itmx=32) where n
    b = ml_b.levels[1]; dt = w*a.Δt[end]; b.x .*= dt
    fill_ω!(ml_ω,a.u,a.μ₀,a.p); biotBC!(a.u,a.U,ml_ω)
    @inside b.z[I] = div(I,a.u); WaterLily.residual!(b); fix_resid!(b.r)
    r₂ = L₂(b); nᵖ = 0
    temp = point(ml_ω)
    while r₂>tol && nᵖ<itmx
        temp .= b.x
        WaterLily.Vcycle!(ml_b); WaterLily.smooth!(b)
        b.ϵ .= b.x .-temp; temp .= 0; fill_ω!(ml_ω,a.μ₀,b.ϵ)
        update_resid!(b.r,a.u,b.z,b.L,ml_ω)
        r₂ = L₂(b); nᵖ+=1
        log && @show nᵖ,r₂
    end
    push!(ml_b.n,nᵖ)
    for i ∈ 1:n
        @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
    end
    pflowBC!(a.u)
    b.x ./= dt
end
function update_resid!(r,u,u_ϵ,L,ω_ϵ)
    # update residual on boundaries
    N,n = size_u(L); inN(I,N) = all(@. 2 ≤ I.I ≤ N-1)
    for i ∈ 1:n
        @loop (u_ϵ[I]=u_ω(i,I,ω_ϵ); u[I,i]+=u_ϵ[I]; inN(I,N) && (r[I]-=u_ϵ[I])) over I ∈ slice(N,2,i)
        @loop (u_ϵ[I]=u_ω(i,I,ω_ϵ); u[I,i]+=u_ϵ[I]; inN(I-δ(i,I),N) && (r[I-δ(i,I)]+=u_ϵ[I])) over I ∈ slice(N,N[i],i)
    end
    fix_resid!(r)
end 