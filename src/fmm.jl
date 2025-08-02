# inverse distance weighted source
using StaticArrays
@inline weighted(r::SVector{3,Float32},S::CartesianIndex{3},i,ω) = permute((j,k)->@inbounds(ω[S,j]*r[k]),i)/√(r'*r)^3/π/4
@inline weighted(r::SVector{2,Float32},S::CartesianIndex{2},i,ω) = (-1)^i*@inbounds(ω[S,1]*r[i%2+1])/(r'*r)/π/2

# Sum over sources at one interaction level
Base.@propagate_inbounds @fastmath function interaction(ω,Ti::CartesianIndex{Np1},l,depth) where Np1
    i,T,N = last(Ti),front(Ti),Np1-1
    x = shifted(T,i)+SVector{N,Float32}(T.I)
    val = zero(eltype(ω))
    domain = inside(size_u(ω)[1])
    Router,Rinner = remaining(T,domain),close(T,domain)
    l == depth && (Router = domain)
    if l == 1 # Top level
        # Do everything remaining inside buff=2
        for S in inR(Router,inside(size_u(ω)[1],buff=2))
            val += weighted(x-SVector{N,Float32}(S.I),S,i,ω)
        end
    elseif Rinner≠Router
        for S in Router
            S ∉ Rinner && (val += weighted(x-SVector{N,Float32}(S.I),S,i,ω))
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
    @vecloop set_velo!(u,U,ω,Ii,fmm) over Ii ∈ targets[1]
end
function fmmBC_r!(r,u,U,ω,tar,ftar)
    interaction!(ω,ftar)
    project!(ω,tar)
    @vecloop velo_resid!(r,u,U,ω,Ii,fmm) over Ii ∈ tar[1]
end
@inline fmm(ml,Ii) = ml[1][Ii]+project(Ii,ml[2])