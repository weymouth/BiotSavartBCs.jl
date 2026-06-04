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
interaction!(ml,flat_targets) = @vecloop _interaction!(ml,lT) over lT ∈ flat_targets
@inline _interaction!(ml,lT) = ((l,T) = lT; ml[l][T] = symmetry(ml[l],T,l,length(ml)))
@inline symmetry(ω,T,args...) = interaction(ω,T,args...) # default is no applied symmetry

# Periodic image contribution using the coarsest ML level
Base.@propagate_inbounds @fastmath @inline function _periodic_sum(ω,Ti::CartesianIndex{Np1},perdir,nimages,l_deep,dims_fine) where Np1
    i,T = last(Ti),front(Ti); N = Np1-1
    x = shifted(T,i)+SVector{N,Float32}(T.I) .- 1.5f0  # tree-level coordinate
    scale = 2f0^(l_deep-1)
    val = zero(eltype(ω))
    for p in perdir
        Lp = Float32(dims_fine[p]-2)
        ep = SVector{N,Float32}(ntuple(j->Float32(j==p),N))
        for S in inside(size_u(ω)[1])
            xS = (SVector{N,Float32}(S.I) .- 1.5f0)*scale
            r  = x-xS
            for n in 1:nimages
                nLep = Float32(n)*Lp*ep
                val += weighted(r-nLep,S,i,ω)+weighted(r+nLep,S,i,ω)
            end
        end
    end
    val
end

add_periodic!(ml,targets_1,perdir,nimages) = @vecloop ml[1][Ti] +=
    _periodic_sum(last(ml),Ti,perdir,nimages,lastindex(ml),size_u(first(ml))[1]) over Ti ∈ targets_1

# Biot-Savart BC using FMM
fmmBC!(ml,targets,flat_targets,::Tuple{},nimages=0) = (interaction!(ml,flat_targets); project!(ml,targets))
function fmmBC!(ml,targets,flat_targets,perdir,nimages=0)
    interaction!(ml,flat_targets); project!(ml,targets)
    add_periodic!(ml,targets[1],perdir,nimages)
end