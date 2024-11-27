# inverse distance weighted source
using StaticArrays
@inline weighted(r::SVector{3,Float32},S::CartesianIndex{3},i,ω) = permute((j,k)->@inbounds(ω[S,j]*r[k]),i)/√(r'*r)^3

# Sum over sources at one interaction level
Base.@propagate_inbounds @fastmath function interaction(ω,Ti,l,depth)
    i,T = last(Ti),front(Ti)
    x = shifted(T,i)+SVector{3,Float32}(T.I)
    val = zero(eltype(ω))
    domain = inside(size_u(ω)[1])
    Router,Rinner = remaining(T,domain),close(T,domain)
    l == depth && (Router = domain)
    if l == 1 # Top level
        # Do everything remaining inside buff=2
        for S in inR(Router,inside(size_u(ω)[1],buff=2))
            val += weighted(x-SVector{3,Float32}(S.I),S,i,ω)
        end
    elseif Rinner≠Router
        for S in Router
            S ∉ Rinner && (val += weighted(x-SVector{3,Float32}(S.I),S,i,ω))
        end
    end; val
end
shifted(T::CartesianIndex{N},i) where N = SVector{N,Float32}(ntuple(j-> j==i ? (T.I[i]==1 ? 0.5 : -0.5) : 0,N))

# Interaction on targets
@inline _interaction!(ω,lT) = ((l,T) = lT; ω[l][T] = interaction(ω[l],T,l,length(ω)))
interaction!(ω,flat_targets) = @vecloop _interaction!(ω,lT) over lT ∈ flat_targets

# Biot-Savart BC using FMM
function fmmBC!(u,U,ω,targets,flat_targets)
    interaction!(ω,flat_targets)
    project!(ω,targets)
    @vecloop fmm_velo!(u,U,ω[1],ω[2],Ii) over Ii ∈ targets[1]
end
Base.@propagate_inbounds @fastmath function fmm_velo!(u,U,a,b,Ii)
    i,I = last(Ii),front(Ii)
    I.I[i]==1 && (I = I+δ(i,I)) # shift for "left" vector field face
    u[I,i] = U[i]+(a[Ii]+0.25f0project(Ii,b))/Float32(4π)
end
function fmmBC_r!(r,u,U,ω,tar,ftar)
    interaction!(ω,ftar)
    project!(ω,tar)
    @vecloop fmm_velo_resid!(r,u,U,ω[1],ω[2],Ii) over Ii ∈ tar[1]
end 
Base.@propagate_inbounds @fastmath function fmm_velo_resid!(r,u,U,a,b,Ii)
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    uI = lower ? Ii+δ(i,Ii) : Ii

    # Update velocity and residual
    uₙ = U[i]+(a[Ii]+0.25f0project(Ii,b))/Float32(4π)
    uₙ⁰ = u[uI]; u[uI] = uₙ
    sgn = lower ? -1 : 1; r[I-sgn*δ(i,I)] += sgn*(uₙ-uₙ⁰)
end
