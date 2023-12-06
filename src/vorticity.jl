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

# use velocity directly
import WaterLily: @loop,permute,inside
function _fill_ω!(ω,i,u)
    top = ω[1]
    @loop top[I] = centered_curl(i,I,u) over I ∈ inside(top,buff=2)
    ml_restrict!(ω)
end
fill_ω!(ω::NTuple{3,NTuple},u) = foreach(i->_fill_ω!(ω[i],i,u),1:3)
fill_ω!(ω::NTuple{N,AbstractArray},u) where N = _fill_ω!(ω,3,u)
centered_curl(i,I,u) = permute((j,k)->WaterLily.∂(k,j,I,u),i)

# use pressure 
function _fill_ω!(ω,i,μ₀,p)
    top = ω[1]
    @loop top[I] = ω_from_p(i,I,μ₀,p) over I ∈ inside(top,buff=2)
    ml_restrict!(ω)
end
fill_ω!(ω::NTuple{3,NTuple},μ₀,p) = foreach(i->_fill_ω!(ω[i],i,μ₀,p),1:3)
fill_ω!(ω::NTuple{N,AbstractArray},μ₀,p) where N = _fill_ω!(ω,3,μ₀,p)
@fastmath function ω_from_p(i,I::CartesianIndex,L,ϵ)
    @inline u(I,i) = @inbounds(-L[I,i]*WaterLily.∂(i,I,ϵ))
    @inline ∂(i,j,I,u) = (u(I+δ(j,I),i)+u(I+δ(j,I)+δ(i,I),i)
                 -u(I-δ(j,I),i)-u(I-δ(j,I)+δ(i,I),i))/4
    return permute((j,k)->∂(k,j,I,u),i)
end

# use both
function _fill_ω!(ω,i,u,μ₀,p)
    top = ω[1]
    @loop top[I] = centered_curl(i,I,u)+ω_from_p(i,I,μ₀,p) over I ∈ inside(top,buff=2)
    ml_restrict!(ω)
end
fill_ω!(ω::NTuple{3,NTuple},u,μ₀,p) = foreach(i->_fill_ω!(ω[i],i,u,μ₀,p),1:3)
fill_ω!(ω::NTuple{N,AbstractArray},u,μ₀,p) where N = _fill_ω!(ω,3,u,μ₀,p)
