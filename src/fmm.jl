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

# Interaction on targets. `symmetry` is a per-level extension hook that case scripts can override.
interaction!(ml,flat_targets) = @vecloop _interaction!(ml,lT) over lT ∈ flat_targets
@inline _interaction!(ml,lT) = ((l,T) = lT; ml[l][T] = symmetry(ml[l],T,l,length(ml)))
@inline symmetry(ω,T,args...) = interaction(ω,T,args...) # default: no applied symmetry

# Periodic interaction: at each FMM level, call interaction() at the ±nL shifted targets.
# Router/Rinner logic naturally gives fine resolution for near images (sources close to
# T±nL at l=1) and coarser resolution for far images — same principle as tree(ml, T±Lδ).
periodic_interaction!(ml,flat_targets,perdir,nimages) = @vecloop _periodic_interaction!(ml,lT,perdir,nimages) over lT ∈ flat_targets
@inline function _periodic_interaction!(ml,lT,perdir,nimages)
    l,Ti = lT; ω = ml[l]
    val = symmetry(ω,Ti,l,length(ml))
    i,T = last(Ti),front(Ti); N = length(T.I)
    Nl = size_u(ω)[1]
    for p in perdir, n in 1:nimages
        nL = CartesianIndex(ntuple(k->k==p ? n*(Nl[p]-2) : 0, N))
        val += interaction(ω,CartesianIndex((T+nL).I...,i),l,length(ml)) +
               interaction(ω,CartesianIndex((T-nL).I...,i),l,length(ml))
    end
    ml[l][Ti] = val
end

# Biot-Savart BC using FMM. Tuple{} dispatch gives zero overhead on the non-periodic path.
fmmBC!(ml,targets,flat_targets,::Tuple{}=(),nimages=0) = (interaction!(ml,flat_targets); project!(ml,targets))
function fmmBC!(ml,targets,flat_targets,perdir,nimages=0)
    periodic_interaction!(ml,flat_targets,perdir,nimages); project!(ml,targets)
end