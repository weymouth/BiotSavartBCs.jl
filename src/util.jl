# Loop macro over an index vector R
using KernelAbstractions
using KernelAbstractions: get_backend,@kernel,@index,@Const
macro vecloop(args...)
    ex,_,itr = args
    _,I,R = itr.args; sym = []
    # grab arguments and replace composites
    WaterLily.grab!(sym,ex)
    setdiff!(sym,[I]) # don't want to pass index as an argument
    @gensym kern ind  # generate unique names
    return quote
        @kernel function $kern($(WaterLily.rep.(sym)...)) # replace composite arguments
            $ind = @index(Global,Linear) # linear index
            @inbounds $I = $R[$ind]      # this is expensive unless R is a vector
            @fastmath @inbounds $ex
        end
        $kern(get_backend($(sym[1])),64)($(sym...),ndrange=length($R))
    end |> esc
end

# Extend some functions
using WaterLily: up,down,inside,@loop
KernelAbstractions.get_backend(nt::NTuple) = get_backend(first(nt))
WaterLily.up(R::CartesianIndices) = first(up(first(R))):last(up(last(R)))
WaterLily.down(R::CartesianIndices) = down(first(R)):down(last(R))
WaterLily.inside(ndims::NTuple{n};buff=1) where n = CartesianIndices(map(N->(1+buff:N-buff),ndims))
inside_u(a;buff=1) = inside_u(size_u(a)[1],buff)
inside_u(ndims::NTuple{n},buff) where n = CartesianIndices((map(N->(1+buff:N-buff),ndims)...,1:n))

# Vector multi-level constructor (top level points to u, doesn't copy)
using WaterLily: size_u
function MLArray(u)
    N,n = size_u(u)
    levels = []
    I = CartesianIndex(ntuple(i-> i==1 ? 1 : 2, n))
    while true
        N = @. 1+N÷2; R = inside(N)
        close(I,R) == R && break
        push!(levels,N)
    end
    zeros_like_u(N,n) = (y = similar(u,N...,n); fill!(y,0); y)
    return (u,map(N->zeros_like_u(N,n),levels)...)
end
close(T::CartesianIndex{2}) = T-4oneunit(T):T+4oneunit(T)
close(T::CartesianIndex{3}) = T-2oneunit(T):T+2oneunit(T)
close(T,R) = inR(close(T),R)
remaining(T,R) = up(close(down(T),down(R)))
inR(x,R) = max(first(x),first(R)):min(last(x),last(R))

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

# Collect "targets" on the faces of a MLArray
using Base.Iterators
slice(dims::NTuple{N},i,s) where N = CartesianIndices((ntuple( k-> k==i ? (s:s) : (2:dims[k]-1), N-1)...,(i:i)))
faces(dims::NTuple{N},off) where N = flatmap(i->flatmap(s->slice(dims,i,s), ((-i∈off ? () : (1,))...,(i∈off ? () : (dims[i],))...)),1:N-1)
collect_targets(ω,off=()) = map(ωᵢ->collect(faces(size(ωᵢ),off)),ω)
flatten_targets(targets) = mapreduce(((level,targets),)->map(T->(level,T),targets),vcat,enumerate(targets))

# Vector MLArray projection on targets
project!(ml::Tuple,mltargets::Tuple) = for l ∈ reverse(2:lastindex(ml)-1)
    project!(ml[l],ml[l+1],mltargets[l])
end
project!(a,b,targets) = @vecloop a[Ii] += project(Ii,b) over Ii ∈ targets
@fastmath function project(Ii::CartesianIndex{4},b)
    I,i,N = front(Ii),last(Ii),size_u(b)[1]
    dj,dk = step(I,i%3+1,N),step(I,(i+1)%3+1,N)
    I,I2,I3,I4 = down(I) .+ (zero(I),dj,dk,dj+dk)
    0.015625f0@inbounds(9b[I,i]+3b[I2,i]+3b[I3,i]+b[I4,i])
end
@fastmath function project(Ii::CartesianIndex{3},b)
    I,i,N = front(Ii),last(Ii),size_u(b)[1]
    d = step(I,i%2+1,N)
    I,I2 = down(I) .+ (zero(I),d)
    0.125f0@inbounds(3b[I,i]+b[I2,i])
end
step(I,j,N,Ij=I.I[j]) = (Ij % 2 == 1 ?    # positive step,
    Ij ÷ 2 == N[j]-2 ? zero(I) : δ(j,I) : # don't step...
    Ij ÷ 2 == 1      ? zero(I) : -δ(j,I)) # past either edge