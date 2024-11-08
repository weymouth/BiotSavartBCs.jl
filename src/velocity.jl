# inverse distance weighted source
using StaticArrays
Base.@propagate_inbounds @fastmath function weighted(i,T,S,ω)
    r = shifted(T,i)+SVector{3,Float32}((T-S).I)
    permute((j,k)->@inbounds(ω[S,j]*r[k]),i)/√(r'*r)^3
end
shifted(T::CartesianIndex{N},i) where N = SVector{N,Float32}(ntuple(j-> j==i ? (T.I[i]==1 ? 0.5 : -0.5) : 0,N))

# Sum over sources for a target (excluding adjacent points)
Base.@propagate_inbounds @fastmath function biot(ω,Ti)
    i,T = last(Ti),front(Ti)
    val = zero(eltype(ω))
    domain = CartesianIndices(map(N->(2:N-1),size_u(ω)[1]))
    R,Rclose = remaining(T,domain),close(T,domain)
    for S in R
        S ∉ Rclose && (val += weighted(i,T,S,ω))
    end; val
end
close(T,R) = inR(T-oneunit(T):T+oneunit(T),R)
remaining(T,R) = up(close(down(T),down(R)))
inR(x,R) = max(first(x),first(R)):min(last(x),last(R))

# Interaction on targets
@inline _interaction!(ω,lT) = ((l,T) = lT; ω[l][T] = biot(ω[l],T))
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
    for i ∈ 1:n
        for j ∈ 1:n # Tangential direction ghosts, curl=0
            j==i && continue
            @loop u[I,j] = u[I+δ(i,I),j]-∂(j,CartesianIndex(I+δ(i,I),i),u) over I ∈ WaterLily.slice(N.-1,1,i,2)
            @loop u[I,j] = u[I-δ(i,I),j]+∂(j,CartesianIndex(I,i),u) over I ∈ WaterLily.slice(N.-1,N[i],i,2)
        end
        # Normal direction ghosts, div=0
        @loop u[I,i] += div(I,u) over I ∈ WaterLily.slice(N.-1,1,i,2)
    end
end
