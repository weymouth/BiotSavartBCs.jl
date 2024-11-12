# compute ω=∇×u excluding boundaries
import WaterLily: permute,∂
fill_ω!(ml::Tuple,u) = (ω=first(ml); fill!(ω,zero(eltype(ω))); fill_ω!(ω,u); restrict!(ml))
fill_ω!(ω,u) = @loop ω[Ii] = centered_curl(Ii,u) over Ii ∈ inside_u(ω,buff=2)
Base.@propagate_inbounds centered_curl(Ii,u) = (I=front(Ii); i=last(Ii); permute((j,k)->∂(k,j,I,u),i))

# inverse distance weighted source
using StaticArrays
Base.@propagate_inbounds @fastmath function weighted(i,T,S,ω)
    r = shifted(T,i)+SVector{3,Float32}((T-S).I)
    permute((j,k)->@inbounds(ω[S,j]*r[k]),i)/√(r'*r)^3
end
shifted(T::CartesianIndex{N},i) where N = SVector{N,Float32}(ntuple(j-> j==i ? (T.I[i]==1 ? 0.5 : -0.5) : 0,N))

# Sum over sources for a target (excluding adjacent points)
Base.@propagate_inbounds @fastmath function biot(ω,Ti,l,depth)
    i,T = last(Ti),front(Ti)
    val = zero(eltype(ω))
    domain = inside(size_u(ω)[1])
    Router,Rinner = remaining(T,domain),close(T,domain)
    l == depth && (Router = domain)
    if l == 1 # Top level
        # Do everything remaining inside buff=2
        for S in inR(Router,inside(size_u(ω)[1],buff=2))
            val += weighted(i,T,S,ω)
        end
    elseif Rinner≠Router
        # Should test against https://github.com/JuliaArrays/TiledIteration.jl?tab=readme-ov-file#edgeiterator
        for S in Router
            S ∉ Rinner && (val += weighted(i,T,S,ω))
        end
    end; val
end
close(T,R) = inR(T-2oneunit(T):T+2oneunit(T),R)
remaining(T,R) = up(close(down(T),down(R)))
inR(x,R) = max(first(x),first(R)):min(last(x),last(R))

# Interaction on targets
@inline _interaction!(ω,lT) = ((l,T) = lT; ω[l][T] = biot(ω[l],T,l,length(ω)))
interaction!(ω,flat_targets) = @loop _interaction!(ω,lT) over lT ∈ flat_targets

# Biot-Savart BC
function biotBC!(u,U,ω,targets,flat_targets)
    interaction!(ω,flat_targets)
    project!(ω,targets)
    @loop biot_finish!(u,U,ω[1],ω[2],Ii) over Ii ∈ targets[1]
end
Base.@propagate_inbounds @fastmath function biot_finish!(u,U,a,b,Ii)
    i,I = last(Ii),front(Ii)
    I.I[i]==1 && (I = I+δ(i,I)) # shift for "left" vector field face
    u[I,i] = U[i]+(a[Ii]+0.25f0project(Ii,b))/Float32(4π)
end

# Incompressible & irrotational ghosts
function pflowBC!(u)
    N,n = size_u(u)
    @inline edge(I,j,val) = 2<I.I[j]<N[j] ? val : zero(eltype(u))
    for i ∈ 1:n
        for j ∈ 1:n # Tangential direction ghosts, curl=0
            j==i && continue
            @loop u[I,j] = u[I+δ(i,I),j] - edge(I,j,∂(j,CartesianIndex(I+δ(i,I),i),u)) over I ∈ slice_u(N,i,j,1)
            @loop u[I,j] = u[I-δ(i,I),j] + edge(I,j,∂(j,CartesianIndex(I,i),u)) over I ∈ slice_u(N,i,j,N[i])
        end # Normal direction ghosts, div=0
        @loop u[I,i] += WaterLily.div(I,u) over I ∈ WaterLily.slice(N.-1,1,i,2)
    end
end
slice_u(N::NTuple{n},i,j,s) where n = CartesianIndices(ntuple(k-> k==i ? (s:s) : k==j ? (2:N[k]) : (2:N[k]-1),n))
