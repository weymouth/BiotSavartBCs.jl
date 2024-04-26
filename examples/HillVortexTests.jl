using WaterLily,BiotSavartBCs

hill_vortex(N;D=3N/4) = function uλ(i,xyz)
    q = xyz .- (N-2)/2; x,y,z = q; r = √(q'*q); θ = acos(z/r); ϕ = atan(y,x)
    v_r = ifelse(2r<D,-1.5*(1-(2r/D)^2),1-(D/2r)^3)*cos(θ)
    v_θ = ifelse(2r<D,1.5-3(2r/D)^2,-1-0.5*(D/2r)^3)*sin(θ)
    i==1 && return sin(θ)*cos(ϕ)*v_r+cos(θ)*cos(ϕ)*v_θ
    i==2 && return sin(θ)*sin(ϕ)*v_r+cos(θ)*sin(ϕ)*v_θ
    cos(θ)*v_r-sin(θ)*v_θ
end
function fill_hill(N,D;T=Float64,mem=Array)
    u = zeros(T,(N,N,N,3))|>mem; apply!(hill_vortex(N;D),u)
    p = zeros(T,(N,N,N))|>mem;
    ω = ntuple(i->MLArray(p),3); fill_ω!(ω,u)
    return p,ω
end

# Test serial vs multi-threaded vs GPU
using CUDA,BenchmarkTools
CUDA.allowscalar(false)
pow = 6; N,D = 2^pow+2,2^(pow-3)
p,ω = fill_hill(N,D,T=Float32);
pC,ωC = fill_hill(N,D,T=Float32,mem=CuArray);

R = inside(p); p[R[1]] = u_ω(1,R[1],ω)
@btime $p[$R] .= u_ω.(Ref(1),$R,Ref($ω)); # 1600ms 0KiB
@btime @inside $p[I] = u_ω(1,I,$ω); # 220ms 24KiB
@btime CUDA.@sync @inside $pC[I] = u_ω(1,I,$ωC); # 76ms 12KiB (Julia 1.10: 41ms 8KiB!)

# Test time scaling with dist
m_u_ω(i,I,ω,dist) = BiotSavartBCs._u_ω(loc(0,I,Float64),dist,lastindex(ω[1]),inside(ω[1][end]),
        @inline (r,I,l=1) -> WaterLily.permute((j,k)->@inbounds(ω[j][l][I]*r[k]),i)/√(r'*r+eps(Float64))^3)/(4π)

pow = 8; N,D = 2^pow+2,2^(pow-3)
p,ω = fill_hill(N,D);
btime(b) = minimum(b).time
duration = [btime(@benchmark m_u_ω(1,$R[1],$ω,$dist)) for dist ∈ 2 .^ collect(0:pow-1)]

using Plots
plot(2.0.^collect(0:7),duration[end]./duration,xlabel="S",ylabel="speedup",
     legend=false,yaxis=:log,xaxis=:log)
savefig("Hill_speedup_dists.png")

# Check error scaling with dist
@inline sdf(I) = √sum(abs2,loc(0,I) .- (N-2)/2) - D/2; @inline J(I) = I+CartesianIndex(0,N÷2,0)
function hill_error(ω,N,D,U=(0,0,1);dist=4)
    uλ = hill_vortex(N;D)
    @inline ϵ(i,I,ω) = uλ(i,loc(0,I))-U[i]-m_u_ω(i,I,ω,dist)
    p = zeros(Float64,(N,1,N))
    WaterLily.@loop (p[I] = sdf(J(I))>1 ? √(ϵ(1,J(I),ω)^2+ϵ(2,J(I),ω)^2+ϵ(3,J(I),ω)^2) : 0) over I ∈ CartesianIndices(p)
    return p
end

pow = 8; N,D = 2^pow+2,2^(pow-3)
pmap(p) = log10(p+10^-6.5)
dis = range(1,N÷2,length=30)
stats(p,i) = pmap(maximum(p[I] for I in CartesianIndices(p) if dis[i-1]<sdf(J(I))≤dis[i]))

using JLD2
# data = load_object("Hill_error_ref.jld2")
data = []; _,ω = fill_hill(N,D);
for dist ∈ 2 .^ collect(0:pow-3) # skip the last two lines
# for dist ∈ 2 .^ collect(0:pow-1) # this takes hours
    @show dist
    p = hill_error(ω,N,D;dist)
    push!(data,[stats(p,i) for i in 2:lastindex(dis)])
end
save_object("Hill_error.jld2",data) # careful not to overwrite

colors = colormap("Blues",pow+2)
plt = plot(xlabel="d/2R",ylabel="max(|uₑ|/U)",ylims=(10^-6,.1));
for (dist,vec) in enumerate(data)
    plot!(plt,collect(dis)[2:end]./D,10.0.^vec,label="S=$(2 .^(dist-1))",
          c=colors[2+dist],yaxis=:log,legendtitle = "Multilevel")
end
plt
savefig("Hill_error_dists.png")