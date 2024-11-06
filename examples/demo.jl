# initialize with random vorticity array and blank velocity
pow = 9; N = 2+2^pow
u = zeros(Float32,(N,N,N)); # targets are boundaries of this array
ω = rand(Float32,(N,N,N));  # sources are inside this array

# Target indices on all 6 boundary faces
using Base.Iterators
slice(dims::NTuple{N},i,s) where N = CartesianIndices(ntuple( k-> k==i ? (s:s) : (2:dims[k]-1), N))
faces(dims,i) = flatmap(s->slice(dims,i,s),(1,dims[i]))
faces(dims::NTuple{N}) where N = flatmap(i->faces(dims,i),1:N)
targets = collect(faces((N,N,N))) # collected

# Sources indices for each target (always excluding boundary points)
sources(T::CartesianIndex{n},N) where n = CartesianIndices((3:10,3:10,3:3))#ntuple(i->clamp(T.I[i]-1,2:N-1):clamp(T.I[i]+1,2:N-1),n))

# inverse distance weighted source
using StaticArrays,LLVM.Interop
function weighted(T,S,ω)
    r = SVector{3,Float32}((T-S).I...)
    m = √(r'*r)^3
    assume(m>0)
    div(@inbounds(ω[S]*r[1]),m)
end

# Update targets with iterators
serial!(u,ω,N) = for T ∈ faces((N,N,N))
    u[T] = sum(S->weighted(T,S,ω),sources(T,N))
end
u_serial = zeros(Float32,(N,N,N)); # targets are boundaries of this array
serial!(u_serial,ω,N)

using KernelAbstractions: get_backend,@kernel,@index,@Const
@kernel function _partial_kern!(u,@Const(ω),@Const(targets),@Const(N))
    i = @index(Global)
    T = targets[i]
    for S in sources(T,N)
        @inbounds @fastmath u[T] += weighted(T,S,ω)
    end
end
partial!(u,ω,targets,N) = _partial_kern!(get_backend(u))(u,ω,targets,N,ndrange=length(targets))
u_partial = zeros(Float32,(N,N,N)); # targets are boundaries of this array
partial!(u_partial,ω,targets,N)

using Atomix
@kernel function _atomix_kern!(u,@Const(ω),@Const(targets),@Const(N))
    i = @index(Group, Linear)
    j = @index(Local, Linear)
    T = targets[i];
    S = sources(T,N)[j]
    @inbounds @fastmath Atomix.@atomic u[T] += weighted(T,S,ω)
end
atomix!(u,ω,targets,N) = _atomix_kern!(get_backend(u),64)(u,ω,targets,N,ndrange=64*length(targets))
u_atomix = zeros(Float32,(N,N,N)); # targets are boundaries of this array
atomix!(u_atomix,ω,targets,N)
u_partial≈u_atomix≈u_serial

using BenchmarkTools
@btime serial!($u_serial,$ω,$N)
@btime partial!($u_partial,$ω,$targets,$N)
@btime atomix!($u_atomix,$ω,$targets,$N)

using CUDA
ω_GPU = CuArray(ω); targets_GPU = CuArray(targets);

u_partial = CUDA.zeros(Float32,(N,N,N)); # targets are boundaries of this array
partial!(u_partial,ω_GPU,targets_GPU,N)
u_partial≈CuArray(u_serial)

u_atomix = CUDA.zeros(Float32,(N,N,N)); # targets are boundaries of this array
atomix!(u_atomix,ω_GPU,targets_GPU,N)
u_atomix≈CuArray(u_serial)

@kernel function _reduce_kern!(u,@Const(ω),@Const(targets),@Const(dims))
    i = @index(Group, Linear)
    j = @index(Local, Linear)
    T = targets[i];
    S = sources(T,dims)[j]
    @inbounds u[T] = CUDA.reduce_block(+, weighted(T,S,ω), zero(eltype(u)), Val(true))
end
reduce!(u,ω,targets,N) = _reduce_kern!(get_backend(u),64)(u,ω,targets,N,ndrange=64*length(targets))
u_reduce = CUDA.zeros(Float32,(N,N,N)); # targets are boundaries of this array
reduce!(u_reduce,ω_GPU,targets_GPU,N)
u_reduce≈CuArray(u_serial)

@btime @CUDA.sync partial!($u_partial,$ω_GPU,$targets_GPU,$N)
@btime @CUDA.sync atomix!($u_atomix,$ω_GPU,$targets_GPU,$N)
@btime @CUDA.sync reduce!($u_reduce,$ω_GPU,$targets_GPU,$N)
