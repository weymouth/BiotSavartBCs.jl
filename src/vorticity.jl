# compute ω=∇×u excluding boundaries
import WaterLily: permute,∂
fill_ω!(ml::Tuple,args...) = (ω=first(ml); fill!(ω,zero(eltype(ω))); fill_ω!(ω,args...); restrict!(ml))
fill_ω!(ω,u) = @loop ω[Ii] = centered_curl(Ii,u) over Ii ∈ inside_u(ω,buff=2)
centered_curl(Ii,u) = (I=front(Ii); i=last(Ii); permute((j,k)->∂(k,j,I,u),i))

# compute ω=-∇×μ₀∇p excluding boundaries (non-zero on body where ∇μ₀≠0)
function fill_ω!(ω,v,μ₀,p)
    @loop v[Ii] = -μ₀[Ii]*∂(last(Ii),front(Ii),p) over Ii ∈ inside_u(v)
    @loop ω[Ii] = centered_curl(Ii,v) over Ii ∈ inside_u(ω,buff=2)
end

# use both u and μ₀∇p
function fill_ω!(ω,u,v,μ₀,p) 
    @loop v[Ii] = -μ₀[Ii]*∂(last(Ii),front(Ii),p) over Ii ∈ inside_u(v)
    @loop ω[Ii] = centered_curl(Ii,u)+centered_curl(Ii,v) over Ii ∈ inside_u(ω,buff=2)
end