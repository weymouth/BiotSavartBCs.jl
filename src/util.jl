# improved loop function
using KernelAbstractions
using KernelAbstractions: get_backend,@kernel,@index,@Const
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args; 
    sym,symI = [],[]
    # grab arguments and replace composites
    WaterLily.grab!(sym,ex)
    WaterLily.grab!(symI,I)
    setdiff!(sym,symI) # don't want to pass index as an argument
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

# Extend some functions
using WaterLily: up,down
KernelAbstractions.get_backend(nt::NTuple) = get_backend(first(nt))
WaterLily.up(R::CartesianIndices) = first(up(first(R))):last(up(last(R)))
WaterLily.down(R::CartesianIndices) = down(first(R)):down(last(R))

# Vector multi-level constructor (top level points to u, doesn't copy)
using WaterLily: divisible,size_u
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

# Extend restrict(!) for MLArrays
using Base: front,last
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
inside_u(a;buff=1) = inside_u(size_u(a)[1],buff)
inside_u(ndims::NTuple{n},buff) where n = CartesianIndices((map(N->(1+buff:N-buff),ndims)...,1:n))

# Collect "targets" on the faces of a MLArray
using Base.Iterators
slice(dims::NTuple{N},i,s) where N = CartesianIndices((ntuple( k-> k==i ? (s:s) : (2:dims[k]-1), N-1)...,(i:i)))
faces(dims::NTuple{N}) where N = flatmap(i->flatmap(s->slice(dims,i,s),(1,dims[i])),1:N-1)
collect_targets(ω) = map(ωᵢ->collect(faces(size(ωᵢ))),ω)
flatten_targets(targets) = mapreduce(((level,targets),)->map(T->(level,T),targets),vcat,enumerate(targets))

# Vector MLArray projection on targets
project!(ml::Tuple,mltargets::Tuple) = for l ∈ reverse(2:lastindex(ml)-1)
    project!(ml[l],ml[l+1],mltargets[l])
end
project!(a,b,targets) = @loop a[Ii] += 0.25f0project(Ii,b) over Ii ∈ targets
project(Ii::CartesianIndex,b) = @inbounds(b[down(front(Ii)),last(Ii)])
