using WaterLily,StaticArrays

# 2D Multi-level Biot-Savart functions 
import WaterLily: divisible,restrict!
function MLArray(x)
    N = size(x)
    levels = [N]
    while all(N .|> divisible) prod(N .-2) > 1000
        N = @. 1+N÷2
        push!(levels,N)
    end
    zeros_like_x(N) = (y = similar(x,N); fill!(y,0); y)
    return Tuple(zeros_like_x(N) for N in levels)
end
ml_restrict!(ml) = for l ∈ 2:lastindex(ml)
    restrict!(ml[l],ml[l-1])
end

import WaterLily: up,@loop,permute
function _u_ω(x,dis,l,R,biotsavart,u=0f0)
    # loop levels
    while l>1
        # find Region close to x
        dx = 2f0^(l-1)
        Rclose = inR(x/dx .-dis,R):inR(x/dx .+dis,R)

        # get contributions outside Rclose
        for I ∈ R
            !(I ∈ Rclose) && (u += biotsavart(r(x,I,dx),I,l))
        end

        # move "up" one level within Rclose
        l -= 1
        R = first(up(first(Rclose))):last(up(last(Rclose)))
    end

    # top level contribution
    for I ∈ R
        u += biotsavart(r(x,I),I)
    end; u
end
u_ω(i,I::CartesianIndex{2},ω) = _u_ω(loc(i,I,Float32),7,lastindex(ω),inside(ω[end]),
    @inline (r,I,l=1) -> @inbounds(ω[l][I]*r[i%2+1])/(r'*r))*(2i-3)/Float32(2π)
u_ω(i,I::CartesianIndex{3},ω) = _u_ω(loc(i,I,Float32),3,lastindex(ω[1]),inside(ω[1][end]),
    @inline (r,I,l=1) -> permute((j,k)->@inbounds(ω[j][l][I]*r[k]),i)/√(r'*r)^3)/Float32(4π)
r(x,I::CartesianIndex,dx=1) = x-dx*(SA_F32[I.I...] .- 1.5f0) # faster than loc(0,I,Float32)
inR(x,R) = clamp(CartesianIndex(round.(Int,x .+ 1.5f0)...),R)
Base.clamp(I::CartesianIndex,R::CartesianIndices) = CartesianIndex(clamp.(I.I,first(R).I,last(R).I))

# Fill ghosts assuming potential flow outside the domain
import WaterLily: size_u,@loop,slice,div
function biotBC!(u,U,ω)
    N,n = size_u(u)
    for i ∈ 1:n
        for s ∈ (2,N[i]) # Domain faces, biotsavart+background
            @loop u[I,i] = u_ω(i,I,ω)+U[i] over I ∈ slice(N,s,i)
        end
    end
end
function pflowBC!(u)
    N,n = size_u(u)
    for i ∈ 1:n
        for j ∈ 1:n # Tangential direction ghosts, curl=0
            j==i && continue
            @loop u[I,j] = u[I+δ(i,I),j]-WaterLily.∂(j,CartesianIndex(I+δ(i,I),i),u) over I ∈ slice(N.-1,1,i,3)
            @loop u[I,j] = u[I-δ(i,I),j]+WaterLily.∂(j,CartesianIndex(I,i),u) over I ∈ slice(N.-1,N[i],i,3)
        end
        # Normal direction ghosts, div=0
        @loop u[I,i] += div(I,u) over I ∈ slice(N.-1,1,i,3)
    end
end
# compute cell-centered ωᵢ and restrict down to lower levels
function _fill_ω!(ω,i,u)
    top = ω[1]
    @loop top[I] = centered_curl(i,I,u) over I ∈ inside(top,buff=2)
    ml_restrict!(ω)
end
fill_ω!(ω::NTuple{3,NTuple},u) = foreach(i->_fill_ω!(ω[i],i,u),1:3)
fill_ω!(ω::NTuple{N,AbstractArray},u) where N = _fill_ω!(ω,3,u)
centered_curl(i,I,u) = permute((j,k)->WaterLily.∂(k,j,I,u),i)

# Check reconstruction on lamb dipole
using CUDA,SpecialFunctions,ForwardDiff
CUDA.allowscalar(false)
function lamb_dipole(N;D=3N/4,U=1,mem=Array)
    β = 2.4394π/D
    C = -2U/(β*besselj0(β*D/2))
    function ψ(x,y)
        r = √(x^2+y^2)
        ifelse(r ≥ D/2, U*((D/2r)^2-1)*y, C*besselj1(β*r)*y/r)
    end
    center = SA[N/2,N/2]
    function uλ(i,xy)
        x,y = xy-center
        ifelse(i==1,ForwardDiff.derivative(y->ψ(x,y),y)+1+U,-ForwardDiff.derivative(x->ψ(x,y),x))
    end
    Simulation((N, N), (1,0), D; uλ, mem) # Don't overwrite ghosts with BCs
end

using BenchmarkTools
begin
    sim = lamb_dipole(3*512,mem=Array); σ = sim.flow.σ; u = sim.flow.u;
    ω = MLArray(σ);
    @btime CUDA.@sync fill_ω!(ω,u) # 3ms,2.5ms
    @btime CUDA.@sync biotBC!(u,sim.flow.U,ω); #4ms,6ms
    @btime CUDA.@sync pflowBC!(u); #0.1ms,0.1ms
    @assert sum(abs2,u-sim.flow.u⁰)/sim.L < 2e-4
end

function hill_vortex(N;D=3N/4,U=1,mem=Array)
    function uλ(i,xyz)
        q = xyz .- N/2; x,y,z = q; r = √(q'*q); θ = acos(z/r); ϕ = atan(y,x)
        v_r = ifelse(2r<D,-1.5*(1-(2r/D)^2),1-(D/2r)^3)*U*cos(θ)
        v_θ = ifelse(2r<D,1.5-3(2r/D)^2,-1-0.5*(D/2r)^3)*U*sin(θ)
        i==1 && return sin(θ)*cos(ϕ)*v_r+cos(θ)*cos(ϕ)*v_θ
        i==2 && return sin(θ)*sin(ϕ)*v_r+cos(θ)*sin(ϕ)*v_θ
        cos(θ)*v_r-sin(θ)*v_θ
    end
    Simulation((N, N, N), (0,0,U), D; uλ, mem) # Don't overwrite ghosts with BCs
end

begin
    sim = hill_vortex(128,mem=CuArray); σ = sim.flow.σ; u = sim.flow.u;
    ω = ntuple(i->MLArray(σ),3);
    @btime CUDA.@sync fill_ω!(ω,u)  # 10ms, 8ms
    @btime CUDA.@sync biotBC!(u,sim.flow.U,ω); #550ms,200ms
    @btime CUDA.@sync pflowBC!(u); #0.4ms,0.3ms
    extrema(u-sim.flow.u⁰) # -7e-4,3e-3
end

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
import WaterLily: residual!,Vcycle!,smooth!,∂
function biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ml_ω;w=1,log=false,tol=1e-3,itmx=32) where n
    b = ml_b.levels[1]; dt = w*a.Δt[end]; b.x .*= dt
    fill_ω!(ml_ω,a.u,a.μ₀,a.p); biotBC!(a.u,a.U,ml_ω)
    @inside b.z[I] = div(I,a.u); residual!(b); fix_resid!(b.r)
    r₂ = L₂(b); nᵖ = 0
    temp = point(ml_ω)
    while r₂>tol && nᵖ<itmx
        temp .= b.x
        Vcycle!(ml_b); smooth!(b)
        b.ϵ .= b.x .-temp; temp .= 0; fill_ω!(ml_ω,a.μ₀,b.ϵ)
        update_resid!(b.r,a.u,b.z,b.L,ml_ω)
        r₂ = L₂(b); nᵖ+=1
        log && @show nᵖ,r₂
    end
    push!(ml_b.n,nᵖ)
    for i ∈ 1:n
        @loop a.u[I,i] -= b.L[I,i]*WaterLily.∂(i,I,b.x) over I ∈ inside(b.x)
    end
    pflowBC!(a.u)
    b.x ./= dt
end
point(ω::NTuple{N,AbstractArray}) where N = ω[1]
point(ω::NTuple{3,NTuple}) = ω[1][1]
function update_resid!(r,u,u_ϵ,L,ω_ϵ)
    # update residual on boundaries
    N,n = size_u(L); inN(I,N) = all(@. 2 ≤ I.I ≤ N-1)
    for i ∈ 1:n
        @loop (u_ϵ[I]=u_ω(i,I,ω_ϵ); u[I,i]+=u_ϵ[I]; inN(I,N) && (r[I]-=u_ϵ[I])) over I ∈ slice(N,2,i)
        @loop (u_ϵ[I]=u_ω(i,I,ω_ϵ); u[I,i]+=u_ϵ[I]; inN(I-δ(i,I),N) && (r[I-δ(i,I)]+=u_ϵ[I])) over I ∈ slice(N,N[i],i)
    end
    fix_resid!(r)
end 
function fix_resid!(r)
    N = size(r); n = length(N); A(i) = 2prod(N.-2)/(N[i]-2)
    res = sum(r)/sum(A,1:n)
    for i ∈ 1:n
        @loop r[I] -= res over I ∈ slice(N.-1,2,i,2)
        @loop r[I] -= res over I ∈ slice(N.-1,N[i]-1,i,2)
    end
    # @assert abs(sum(r[inside(r)]))<1e-4
end 
function _fill_ω!(ω,i,μ₀,p)
    top = ω[1]
    @loop top[I] = ω_from_p(i,I,μ₀,p) over I ∈ inside(top,buff=2)
    ml_restrict!(ω)
end
fill_ω!(ω::NTuple{3,NTuple},μ₀,p) = foreach(i->_fill_ω!(ω[i],i,μ₀,p),1:3)
fill_ω!(ω::NTuple{N,AbstractArray},μ₀,p) where N = _fill_ω!(ω,3,μ₀,p)
function _fill_ω!(ω,i,u,μ₀,p)
    top = ω[1]
    @loop top[I] = centered_curl(i,I,u)+ω_from_p(i,I,μ₀,p) over I ∈ inside(top,buff=2)
    ml_restrict!(ω)
end
fill_ω!(ω::NTuple{3,NTuple},u,μ₀,p) = foreach(i->_fill_ω!(ω[i],i,u,μ₀,p),1:3)
fill_ω!(ω::NTuple{N,AbstractArray},u,μ₀,p) where N = _fill_ω!(ω,3,u,μ₀,p)
@fastmath function ω_from_p(i,I::CartesianIndex,L,ϵ)
    @inline u(I,i) = @inbounds(-L[I,i]*WaterLily.∂(i,I,ϵ))
    @inline ∂(i,j,I,u) = (u(I+δ(j,I),i)+u(I+δ(j,I)+δ(i,I),i)
                 -u(I-δ(j,I),i)-u(I-δ(j,I)+δ(i,I),i))/4
    return permute((j,k)->∂(k,j,I,u),i)
end

# Check pressure solver convergence on circle
include("examples/TwoD_plots.jl")
circ(D,U=1,m=11D÷8;mem=Array) = Simulation((2D,m), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=U*D/1e4,mem)
sim = circ(256,mem=Array); ω = MLArray(sim.flow.σ);
@time biot_mom_step!(sim.flow,sim.pois,ω)
flood(sim.flow.p,border=:none)
@time biot_mom_step!(sim.flow,sim.pois,ω)
flood(sim.flow.p,border=:none)
@show sim.pois.n
# @time while sim_time(sim)<1.2
#     biot_mom_step!(sim.flow,sim.pois,ω)
#     sim_time(sim)%0.1<sim.flow.Δt[end]/sim.L && @show sim_time(sim),sim.flow.Δt[end],sim.pois.n[end]
# end
# @inside sim.flow.σ[I] = centered_curl(3,I,sim.flow.u)*sim.L/sim.U
# flood(sim.flow.σ|>Array,border=:none,legend=false,clims=(-25,25))

sphere(D,U=1,m=3D÷2;mem=Array) = Simulation((2D,m,m), (U,0,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=U*D/1e4,mem)
sim = sphere(128,mem=CuArray); ω = ntuple(i->MLArray(sim.flow.σ),3);
@time biot_mom_step!(sim.flow,sim.pois,ω)
@show sim.pois.n
flood(sim.flow.p[:,:,3sim.L÷4]|>Array,border=:none,legend=false)
flood(sim.flow.u[:,:,3sim.L÷4,2]|>Array,border=:none,legend=false)