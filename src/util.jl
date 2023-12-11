# Fill ghosts assuming potential flow outside the domain
import WaterLily: size_u,slice,div,∂
function biotBC!(u,U,ω)
    N,n = size_u(u)
    for i ∈ 1:n, s ∈ (2,N[i]) # Domain faces, biotsavart+background
        @loop u[I,i] = u_ω(i,I,ω)+U[i] over I ∈ slice(N,s,i)
    end
end
function pflowBC!(u)
    N,n = size_u(u)
    for i ∈ 1:n
        for j ∈ 1:n # Tangential direction ghosts, curl=0
            j==i && continue
            @loop u[I,j] = u[I+δ(i,I),j]-∂(j,CartesianIndex(I+δ(i,I),i),u) over I ∈ slice(N.-1,1,i,2)
            @loop u[I,j] = u[I-δ(i,I),j]+∂(j,CartesianIndex(I,i),u) over I ∈ slice(N.-1,N[i],i,2)
        end
        # Normal direction ghosts, div=0
        @loop u[I,i] += div(I,u) over I ∈ slice(N.-1,1,i,2)
    end
end

point(ω::NTuple{N,AbstractArray}) where N = ω[1]
point(ω::NTuple{3,NTuple}) = ω[1][1]

function fix_resid!(r)
    N = size(r); n = length(N); A(i) = 2prod(N.-2)/(N[i]-2)
    res = sum(r)/sum(A,1:n)
    for i ∈ 1:n
        @loop r[I] -= res over I ∈ slice(N.-1,2,i,2)
        @loop r[I] -= res over I ∈ slice(N.-1,N[i]-1,i,2)
    end
end 