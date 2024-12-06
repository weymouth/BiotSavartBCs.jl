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
Base.@propagate_inbounds @fastmath function set_velo!(u,U,ω,Ii,fnc)
    i,I = last(Ii),front(Ii)
    I.I[i]==1 && (I = I+δ(i,I)) # shift for "left" vector field face
    u[I,i] = U[i]+fnc(ω,Ii)
end

using Atomix
_biotBC_r!(r,u,U,ω,targets,flat_targets,fmm) = fmm ? fmmBC_r!(r,u,U,ω,targets,flat_targets) : treeBC_r!(r,u,U,ω,targets[1])
biotBC_r!(r,u,U,ω,targets,flat_targets;fmm=true) = (_biotBC_r!(r,u,U,ω,targets,flat_targets,fmm); fix_resid!(r,u,targets[1]))
Base.@propagate_inbounds @fastmath function velo_resid!(r,u,U,ω,Ii,fnc)
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    uI = lower ? Ii+δ(i,Ii) : Ii

    # Set velocity and update residual
    uₙ = U[i]+fnc(ω,Ii)
    uₙ⁰ = u[uI]; u[uI] = uₙ
    Atomix.@atomic r[I+(lower ? δ(i,I) : -δ(i,I))] += (uₙ-uₙ⁰)*(lower ? -1 : 1)
end

fix_resid!(r,u,targets,fix=sum(r)/length(targets)) = @vecloop _fix_resid!(r,u,fix,Ii) over Ii ∈ targets
@inline function _fix_resid!(r,u,fix,Ii)
    I,i = front(Ii),last(Ii)
    left = I.I[i]==1
    Atomix.@atomic r[I+ (left ? δ(i,I) : -δ(i,I))] -= fix
    u[I+ (left ? δ(i,I) : zero(I)),i] += fix*(left ? 1 : -1)
end