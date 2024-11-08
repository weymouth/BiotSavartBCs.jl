# compute ω=∇×u excluding boundaries
import WaterLily: permute,∂
fill_ω!(ml::Tuple,args...) = (ω=first(ml); fill!(ω,zero(eltype(ω))); fill_ω!(ω,args...); restrict!(ml))
fill_ω!(ω,u) = @loop ω[Ii] = centered_curl(Ii,u) over Ii ∈ inside_u(ω,buff=2)
centered_curl(Ii,u) = (I=front(Ii); i=last(Ii); permute((j,k)->∂(k,j,I,u),i))

# compute ω=-∇×μ₀∇p excluding boundaries (non-zero on body where ∇μ₀≠0)
fill_ω!(ω,μ₀,p) = @loop ω[Ii] = ω_from_p(Ii,μ₀,p) over Ii ∈ inside_u(ω,buff=2)
@fastmath function ω_from_p(Ii,μ₀,p)
    I,i=front(Ii),last(Ii)
    @inline u(I,i) = @inbounds(-μ₀[I,i]*∂(i,I,p))
    @inline ∂u(i,j,I,u) = (u(I+δ(j,I),i)+u(I+δ(j,I)+δ(i,I),i)
                          -u(I-δ(j,I),i)-u(I-δ(j,I)+δ(i,I),i))/4
    return permute((j,k)->∂u(k,j,I,u),i)
end

# use both u and μ₀∇p
fill_ω!(ω,u,μ₀,p) = @loop ω[Ii] = centered_curl(Ii,u)+ω_from_p(Ii,μ₀,p) over Ii ∈ inside_u(ω,buff=2)