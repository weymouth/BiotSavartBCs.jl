# compute ω=∇×u excluding boundaries
import WaterLily: permute,∂
fill_ω!(ml::Tuple,u) = (ω=first(ml); fill!(ω,zero(eltype(ω))); fill_ω!(ω,u); restrict!(ml))
fill_ω!(ω,u) = @loop ω[Ii] = centered_curl(Ii,u) over Ii ∈ inside_u(ω,buff=2)
Base.@propagate_inbounds centered_curl(Ii,u) = (I=front(Ii); i=last(Ii); permute((j,k)->∂(k,j,I,u),i))

# Incompressible & irrotational ghosts
function pflowBC!(u)
    N,n = size_u(u)
    @inline edge(I,j,val) = 2<I.I[j]<N[j] ? val : zero(eltype(u))
    for i ∈ 1:n # we know this is slow on GPUs!!
        for j ∈ 1:n # Tangential direction ghosts, curl=0
            j==i && continue
            @loop u[I,j] = u[I+δ(i,I),j] - edge(I,j,∂(j,CartesianIndex(I+δ(i,I),i),u)) over I ∈ slice_u(N,i,j,1)
            @loop u[I,j] = u[I-δ(i,I),j] + edge(I,j,∂(j,CartesianIndex(I,i),u)) over I ∈ slice_u(N,i,j,N[i])
        end # Normal direction ghosts, div=0
        @loop u[I,i] += WaterLily.div(I,u) over I ∈ WaterLily.slice(N.-1,1,i,2)
    end
end
slice_u(N::NTuple{n},i,j,s) where n = CartesianIndices(ntuple(k-> k==i ? (s:s) : k==j ? (2:N[k]) : (2:N[k]-1),n))

# Biot-Savart BCs
biotBC!(u,U,ω,targets,flat_targets;fmm=true) = fmm ? fmmBC!(u,U,ω,targets,flat_targets) : treeBC!(u,U,ω,targets[1])
_biotBC_r!(r,u,U,ω,targets,flat_targets,fmm) = fmm ? fmmBC_r!(r,u,U,ω,targets,flat_targets) : treeBC_r!(r,u,U,ω,targets[1])
biotBC_r!(r,u,U,ω,targets,flat_targets;fmm=true) = (_biotBC_r!(r,u,U,ω,targets,flat_targets,fmm); fix_resid!(r,targets[1]))

fix_resid!(r,targets,fix=sum(r)/length(targets)) = @vecloop _fix_resid!(r,fix,Ii) over Ii ∈ targets
@inline function _fix_resid!(r,fix,Ii)
    I,i = front(Ii),last(Ii)
    step = I.I[i]==1 ? δ(i,I) : -δ(i,I)
    r[I+step] -= fix
end