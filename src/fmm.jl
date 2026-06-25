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

# Periodic-aware interaction for ANY level. Same near/far shell structure as interaction(), but
# the shell boxes are clipped against a domain EXTENDED by `nimages` periods in `perdir` (so the
# periodic boundary is never hit within range), and ω is read at the periodically-WRAPPED index
# while the kernel distance uses the UNWRAPPED position. interaction()'s plain clipping to the
# finite domain breaks z-translation invariance in the periodic directions — near a periodic
# boundary the wrapped neighbours that fall in a given shell depend on the target's position, so a
# z-uniform source gives a z-varying induced velocity that the pressure solve turns into a spurious
# spanwise velocity. Clipping against the extended domain + wrapping removes that, at every level
# (the coarse levels were the dominant offender, where the coarse-direction interior shrinks to ~
# the close() half-width so the box can no longer tile periodically).
Base.@propagate_inbounds @fastmath function pinteraction(ω,Ti::CartesianIndex{Np1},l,depth,perdir,nimages) where Np1
    i,T,N = last(Ti),front(Ti),Np1-1
    Nu = size_u(ω)[1]
    x = shifted(T,i)+SVector{N,Float32}(T.I)
    per = ntuple(k->Nu[k]-2, N)
    lo_d = ntuple(k-> k in perdir ? 2 - nimages*per[k]       : 2,       N)   # domain extended
    hi_d = ntuple(k-> k in perdir ? Nu[k]-1 + nimages*per[k] : Nu[k]-1, N)   # by nimages periods
    pdom = CartesianIndices(ntuple(k->lo_d[k]:hi_d[k],N))
    Router = l == depth ? pdom : remaining(T,pdom)
    Rinner = close(T,pdom)
    val = zero(eltype(ω))
    (l != 1 && Rinner == Router) && return val
    ilo,ihi = first(Rinner),last(Rinner)
    for S in Router
        ok = true                          # non-perdir interior clip (buff=2 at l==1, else buff=1)
        for k in 1:N
            (k in perdir) && continue
            (l == 1 ? (3 ≤ S[k] ≤ Nu[k]-2) : (2 ≤ S[k] ≤ Nu[k]-1)) || (ok = false)
        end
        ok || continue
        if l != 1                          # exclude the inner near box (handled by finer levels)
            inn = true
            for k in 1:N
                (ilo[k] ≤ S[k] ≤ ihi[k]) || (inn = false)
            end
            inn && continue
        end
        Sw = CartesianIndex(ntuple(k-> k in perdir ? mod(S[k]-2,per[k])+2 : S[k], N))
        val += weighted(x-SVector{N,Float32}(S.I),Sw,i,ω)
    end
    val
end

# Periodic interaction: periodic-aware shell sum at every level (see pinteraction).
periodic_interaction!(ml,flat_targets,perdir,nimages) = @vecloop _periodic_interaction!(ml,lT,perdir,nimages) over lT ∈ flat_targets
@inline function _periodic_interaction!(ml,lT,perdir,nimages)
    l,Ti = lT; ml[l][Ti] = pinteraction(ml[l],Ti,l,length(ml),perdir,nimages)
end

# Biot-Savart BC using FMM. Tuple{} dispatch gives zero overhead on the non-periodic path.
fmmBC!(ml,targets,flat_targets,::Tuple{}=(),nimages=0) = (interaction!(ml,flat_targets); project!(ml,targets))
function fmmBC!(ml,targets,flat_targets,perdir,nimages=0)
    periodic_interaction!(ml,flat_targets,perdir,nimages); project!(ml,targets)
end