# multi-level array generator
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

# Fill mult-level ω using overloading for different dimensions and arguments
fill_ω!(ω::NTuple{N,AbstractArray},kw...) where N = _fill_ω!(ω,3,kw...) # 2D only uses ω₃
fill_ω!(ω::NTuple{3,NTuple},kw...) = foreach(i->_fill_ω!(ω[i],i,kw...),1:3)
_fill_ω!(ω,i,kw...) = (top_ω!(ω[1],i,kw...); ml_restrict!(ω))

# compute ω=∇×u excluding boundaries
import WaterLily: @loop,permute,inside,∂
@inline top_ω!(ω,i,u) = @loop ω[I] = centered_curl(i,I,u) over I ∈ inside(ω,buff=2)
centered_curl(i,I,u) = permute((j,k)->∂(k,j,I,u),i)

# compute ω=-∇×μ₀∇p excluding boundaries (non-zero on body where ∇μ₀≠0)
@inline top_ω!(ω,i,μ₀,p) = @loop ω[I] = ω_from_p(i,I,μ₀,p) over I ∈ inside(ω,buff=2)
@fastmath function ω_from_p(i,I,μ₀,p)
    @inline u(I,i) = @inbounds(-μ₀[I,i]*∂(i,I,p))
    @inline ∂u(i,j,I,u) = (u(I+δ(j,I),i)+u(I+δ(j,I)+δ(i,I),i)
                 -u(I-δ(j,I),i)-u(I-δ(j,I)+δ(i,I),i))/4
    return permute((j,k)->∂u(k,j,I,u),i)
end

# use both u and μ₀∇p
@inline top_ω!(ω,i,u,μ₀,p) = @loop ω[I] = centered_curl(i,I,u)+ω_from_p(i,I,μ₀,p) over I ∈ inside(ω,buff=2)