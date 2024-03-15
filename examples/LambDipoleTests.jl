using WaterLily,BiotSavartBCs
using SpecialFunctions,ForwardDiff,Plots
function lamb_dipole(N;D=3N/4,U=1)
    β = 2.4394π/D
    C = -2U/(β*besselj0(β*D/2))
    function ψ(x,y)
        r = √(x^2+y^2)
        ifelse(r ≥ D/2, U*((D/2r)^2-1)*y, C*besselj1(β*r)*y/r)
    end
    return function uλ(i,xy)
        x,y = xy .- (N-2)/2
        i==1 && return ForwardDiff.derivative(y->ψ(x,y),y)+1+U
        -ForwardDiff.derivative(x->ψ(x,y),x)
    end
end

m_u_ω(i,I,ω,dist) = BiotSavartBCs._u_ω(loc(0,I,Float64),dist,lastindex(ω),inside(ω[end]),
        @inline (r,I,l=1) -> @inbounds(ω[l][I]*r[i%2+1])/(r'*r+eps(Float64)))*(2i-3)/(2π)

using BenchmarkTools
btime(b) = minimum(b).time
function lamb_test(N,D=3N/4,U=(1,0);dist=7)
    uλ = lamb_dipole(N;D)
    u = zeros(Float64,(N,N,2)); apply!(uλ,u)
    ω = MLArray(u[:,:,1]); fill_ω!(ω,u);

    time = btime(@benchmark m_u_ω(1,CartesianIndex(2,2),$ω,$dist))

    sdf(I) = √sum(abs2,loc(0,I) .- (N-2)/2) - D/2
    ϵ(i,I,ω) = uλ(i,loc(0,I))-U[i]-m_u_ω(i,I,ω,dist)

    p = zeros(Float64,(N,N));
    WaterLily.@loop (p[I] = sdf(I)>1 ? √(ϵ(1,I,ω)^2+ϵ(2,I,ω)^2) : 0) over I ∈ inside(p,buff=0)
    return p,time
end
function flood(f::Array;shift=(0.,0.),cfill=:RdBu_11,clims=(),levels=10,kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    Plots.contourf(axes(f,1).+shift[1],axes(f,2).+shift[2],f',
        linewidth=0, levels=levels, color=cfill, clims = clims, 
        aspect_ratio=:equal; kv...)
end

# Check dependancy on dist = size of kernel
pow = 8; N,D = 2^pow+2,2^(pow-3)
pmap(p) = log10(p+10^-6.5)
dis = range(1,N÷2,length=30)
d(I) = √sum(abs2,loc(0,I) .- (N-2)/2) - D/2
stats(p,i) = pmap(maximum(p[I] for I in CartesianIndices(p) if dis[i-1]<d(I)≤dis[i]))

data = []; duration = [];
for dist ∈ 2 .^ collect(0:pow-1)
    @show dist
    p,time = lamb_test(N,D;dist); push!(duration,time)
    flood(pmap.(p),clims=(-6,-1),border=:none,cfill=:Greens)
    savefig("lamb_dipole_error_dist$(dist).png")
    push!(data,[stats(p,i) for i in 2:lastindex(dis)])
end
using JLD2
save_object("error_dists.jld2",data)

colors = colormap("Blues",pow+2)
plt = plot(xlabel="d/D",ylabel="max(log10(|uₑ|/U))");
for (dist,vec) in enumerate(data[1:end])
    plot!(plt,collect(dis)[2:end]./D,vec,label="log₂(size)=$(dist-1)",c=colors[2+dist])
end
plt
savefig("lamb_dipole_error_dists.png")

# duration = [0.667296,1.090,2.389,6.100,16.800,48.600,122.400,230.400]
plot(0:7,log10(duration[1]).-log10.(duration),xlabel="log₂(kernel size)",ylabel="log₁₀(speedup)",legend=false)
savefig("lamb_dipole_speedup_dists.png")
