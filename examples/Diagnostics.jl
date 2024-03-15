using StaticArrays
using ForwardDiff
using LinearAlgebra: tr
using WaterLily: kern, ∂, AbstractBody, size_u
using WaterLily

# viscous stress tensor, 
∇²u(I::CartesianIndex{2},u) = @SMatrix [∂(i,j,I,u)+∂(j,i,I,u) for i ∈ 1:2, j ∈ 1:2]
∇²u(I::CartesianIndex{3},u) = @SMatrix [∂(i,j,I,u)+∂(j,i,I,u) for i ∈ 1:3, j ∈ 1:3]
"""
Curvature corrected kernel evaluated ε away from the body
"""
@inline function nds_ϵ(body::AbstractBody,x,t,ε,κ)
    d,n,_ = measure(body,x,t); 
    isnothing(κ) &&  (κ=0.5tr(ForwardDiff.hessian(y -> body.sdf(y,t),x)); κ=ifelse(isnan(κ),0.0,κ))
    n*WaterLily.kern(clamp(d-ε,-1,1))/prod(1.0.+κ*d)
end
"""
Surface integral of pressure and viscous stress tensor, can provide a curvature if known
"""
function diagnostics(a::Simulation;κ=nothing)
    # get time
    t = WaterLily.time(a); _,N = size_u(a.flow.u); T = eltype(a.flow.u)
    # compute the pressure force
    @WaterLily.loop a.flow.f[I,:] .= -a.flow.p[I]*nds_ϵ(a.body,loc(0,I,T),t,a.ϵ,κ) over I ∈ inside(a.flow.p)
    pressure=[sum(@inbounds(a.flow.f[inside(a.flow.p),i])) for i ∈ 1:N] |> Array
    # compute the viscous force
    @WaterLily.loop a.flow.f[I,:] .= a.flow.ν.*∇²u(I,a.flow.u)*nds_ϵ(a.body,loc(0,I,T),t,a.ϵ,κ) over I ∈ inside(a.flow.p)
    viscous=[sum(@inbounds(a.flow.f[inside(a.flow.p),i])) for i ∈ 1:N] |> Array
    return pressure,viscous
end
