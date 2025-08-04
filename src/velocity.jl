# compute ω=∇×u excluding boundaries
import WaterLily: permute,∂
fill_ω!(ml::Tuple,u) = (ω=first(ml); fill!(ω,zero(eltype(ω))); fill_ω!(ω,u); restrict!(ml))
fill_ω!(ω,u) = @loop ω[Ii] = centered_curl(Ii,u) over Ii ∈ inside_u(ω,buff=2)
Base.@propagate_inbounds centered_curl(Ii::CartesianIndex{4},u) = (I=front(Ii); i=last(Ii); permute((j,k)->∂(k,j,I,u),i))
Base.@propagate_inbounds centered_curl(Ii::CartesianIndex{3},u) = (I=front(Ii); i=last(Ii); i==1 ? permute((j,k)->∂(k,j,I,u),3) : zero(eltype(u)))

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
function biotBC!(u,U,ml,targets,flat_targets;fmm=true)
    fmm ? fmmBC!(ml,targets,flat_targets) : treeBC!(ml,targets[1]) # Fill ml[targets]=uᵥ
    @vecloop _biotBC!(u,U,ml[1],Ii) over Ii ∈ targets[1]           # Set u = uᵥ+U
end
@inline function _biotBC!(u,U,uᵥ,Ii)
    i,I = last(Ii),front(Ii); lower = I.I[i]==1 
    u[I+(lower ? δ(i,I) : zero(I)),i] = U[i]+uᵥ[Ii]
end

using Atomix
# Biot-Savart BCs + residual update
function biotBC_r!(r,u,U,ml,targets,flat_targets;fmm=true)
    fmm ? fmmBC!(ml,targets,flat_targets) : treeBC!(ml,targets[1]) # Fill ml[targets]=uᵥ
    @vecloop _biotBC_r!(r,u,U,ml[1],Ii) over Ii ∈ targets[1]       # Update the u,r
    fix_resid!(r,u,targets[1])                                     # Fix u,r 
end
@inline function _biotBC_r!(r,u,U,uᵥ,Ii)
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    uₙ = U[i]+uᵥ[Ii]
    uI = lower ? Ii+δ(i,Ii) : Ii; uₙ⁰ = u[uI]; u[uI] = uₙ
    Atomix.@atomic r[I+(lower ? δ(i,I) : -δ(i,I))] += (uₙ-uₙ⁰)*(lower ? -1 : 1)
end

# Correct the global residual s.t. sum(r)=0 
fix_resid!(r,u,targets,fix=sum(r)/length(targets)) = @vecloop _fix_resid!(r,u,fix,Ii) over Ii ∈ targets
@inline function _fix_resid!(r,u,fix,Ii)
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    u[I+ (lower ? δ(i,I) : zero(I)),i] += fix*(lower ? 1 : -1)
    Atomix.@atomic r[I+ (lower ? δ(i,I) : -δ(i,I))] -= fix
end