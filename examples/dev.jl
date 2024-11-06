# improved loop function
using WaterLily
using KernelAbstractions: get_backend,@kernel,@index,@Const
using KernelAbstractions
KernelAbstractions.get_backend(nt::NTuple) = get_backend(first(nt))
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args; sym = []
    WaterLily.grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    @gensym kern i    # generate unique kernel function name
    return quote
        @kernel function $kern($(WaterLily.rep.(sym)...)) # replace composite arguments
            $i = @index(Global,Linear)
            $I = $R[$i]
            @fastmath @inbounds $ex
        end
        $kern(get_backend($(sym[1])),64)($(sym...),ndrange=length($R))
    end |> esc
end

# inverse distance weighted source
using WaterLily: permute
using StaticArrays
Base.@propagate_inbounds @fastmath function weighted(i,T,S,ω)
    r = shifted(T,i)+SVector{3,Float32}((T-S).I)
    permute((j,k)->@inbounds(ω[S,j]*r[k]),i)/√(r'*r)^3
end
shifted(T::CartesianIndex{N},i) where N = SVector{N,Float32}(ntuple(j-> j==i ? (T.I[i]==1 ? 0.5 : -0.5) : 0,N))

# Sum over sources for a target (excluding adjacent points)
using Base: front,last
using WaterLily: up,down
Base.@propagate_inbounds function biot(ω,Ti)
    i,T = last(Ti),front(Ti)
    val = zero(eltype(ω))
    domain = CartesianIndices(ntuple(i->2:size(ω,i)-1,3))
    R,Rclose = remaining(T,domain),close(T,domain)
    for S in R
        S ∉ Rclose && (val += weighted(i,T,S,ω))
    end; val
end
close(T,R) = inR(T-oneunit(T):T+oneunit(T),R)
remaining(T,R) = up(close(down(T),down(R)))
inR(x,R) = max(first(x),first(R)):min(last(x),last(R))
WaterLily.up(R::CartesianIndices) = first(up(first(R))):last(up(last(R)))
WaterLily.down(R::CartesianIndices) = down(first(R)):down(last(R))

# Interaction on targets
serial!(mlω,mltargets) = for (ω,targets) ∈ zip(mlω,mltargets), T ∈ targets
    @inbounds ω[T] = biot(ω,T)
end
@inline _interaction!(ω,lT) = ((l,T) = lT; ω[l][T] = biot(ω[l],T))
interaction!(ω,flat_targets) = @loop _interaction!(ω,lT) over lT ∈ flat_targets
using Base.Iterators
slice(dims::NTuple{N},i,s) where N = CartesianIndices((ntuple( k-> k==i ? (s:s) : (2:dims[k]-1), N-1)...,(i:i)))
faces(dims::NTuple{N}) where N = flatmap(i->flatmap(s->slice(dims,i,s),(1,dims[i])),1:N-1)
collect_targets(ω) = map(ωᵢ->collect(faces(size(ωᵢ))),ω)
flatten_targets(targets) = mapreduce(((level,targets),)->map(T->(level,T),targets),vcat,enumerate(targets))

# vector restriction 
import WaterLily: divisible,inside_u,size_u
function MLArray(u)
    N,n = size_u(u)
    levels = []
    while all(N .|> divisible)
        N = @. 1+N÷2
        push!(levels,N)
    end
    zeros_like_u(N,n) = (y = similar(u,N...,n); fill!(y,0); y)
    return (u,map(N->zeros_like_u(N,n),levels)...)
end
restrict!(ml::NTuple) = for l ∈ 2:lastindex(ml)
    restrict!(ml[l],ml[l-1])
end
restrict!(a,b) = @loop a[Ii] = restrict(Ii,b) over Ii ∈ inside_u(a)
@fastmath @inline function restrict(Ii::CartesianIndex,b)
    s = zero(eltype(b))
    for J ∈ up(front(Ii))
     s += @inbounds(b[J,last(Ii)])
    end; s
end
inside_u(a) = inside_u(size_u(a)[1])

# Target (only) projection
project!(ml::Tuple,mltargets::Tuple) = for l ∈ reverse(2:lastindex(ml)-1)
    project!(ml[l-1],ml[l],mltargets[l-1])
end
project!(a,b,targets) = @loop a[Ii] += 0.25f0project(Ii,b) over Ii ∈ targets
project(Ii::CartesianIndex,b) = @inbounds(b[down(front(Ii)),last(Ii)])
Base.@propagate_inbounds @fastmath function top_project!(u,U,a,b,Ii)
    i,I = last(Ii),front(Ii)
    I.I[i]==1 && (I = I+δ(i,I)) # shift for "left" vector field face
    u[I,i] = U[i]+(a[Ii]+0.25f0project(Ii,b))/Float32(4π)
end

# Biot-Savart BC
function biotBC!(u,U,ω,targets,flat_targets)
    restrict!(ω)
    interaction!(ω,flat_targets)
    project!(ω,targets)
    @loop top_project!(u,U,ω[1],ω[2],Ii) over Ii ∈ targets[1]
end

# Set up example
pow = 9; N = 2+2^pow
ω = MLArray(rand(Float32,(N,N,N,3)));
u = zeros(Float32,(N,N,N,3)); U = (1,0,0)
targets = collect_targets(ω);
flat_targets = flatten_targets(targets);

using CUDA
ω_G = CuArray.(ω); u_G = CuArray(u); targets_G = CuArray.(targets); flat_targets_G = CuArray(flat_targets);

restrict!(ω);
ω2 = deepcopy(ω);
restrict!(ω_G)
@assert last(ω) ≈ Array(last(ω_G))

serial!(ω,targets)
interaction!(ω2,flat_targets)
@assert first(ω) ≈ first(ω2)
@assert last(ω) ≈ last(ω2)
interaction!(ω_G,flat_targets_G)
@assert first(ω) ≈ Array(first(ω_G))
@assert last(ω) ≈ Array(last(ω_G))

project!(ω,targets)
project!(ω_G,targets_G)
@assert first(ω) ≈ Array(first(ω_G))

biotBC!(u,U,ω,targets,flat_targets)
biotBC!(u_G,U,ω_G,targets_G,flat_targets_G)
@assert u ≈ Array(u_G)

using BenchmarkTools
@btime restrict!($ω)
@btime @CUDA.sync restrict!($ω_G)
@btime serial!($ω,$targets)
@btime interaction!($ω,$flat_targets)
@btime @CUDA.sync interaction!($ω_G,$flat_targets_G)
@btime project!($ω,$targets)
@btime @CUDA.sync project!($ω_G,$targets_G)
@btime biotBC!($u,$U,$ω,$targets,$flat_targets)
@btime @CUDA.sync biotBC!($u_G,$U,$ω_G,$targets_G,$flat_targets_G)