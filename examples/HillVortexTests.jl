using WaterLily,BiotSavartBCs,Plots

hill_vortex(N;D=3N/4) = function uλ(i,xyz)
    q = xyz .- (N-2)/2; x,y,z = q; r = √(q'*q); θ = acos(z/r); ϕ = atan(y,x)
    v_r = ifelse(2r<D,-1.5*(1-(2r/D)^2),1-(D/2r)^3)*cos(θ)
    v_θ = ifelse(2r<D,1.5-3(2r/D)^2,-1-0.5*(D/2r)^3)*sin(θ)
    i==1 && return sin(θ)*cos(ϕ)*v_r+cos(θ)*cos(ϕ)*v_θ
    i==2 && return sin(θ)*sin(ϕ)*v_r+cos(θ)*sin(ϕ)*v_θ
    cos(θ)*v_r-sin(θ)*v_θ
end

m_u_ω(i,I,ω,dist) = BiotSavartBCs._u_ω(loc(0,I,Float64),dist,lastindex(ω[1]),inside(ω[1][end]),
        @inline (r,I,l=1) -> WaterLily.permute((j,k)->@inbounds(ω[j][l][I]*r[k]),i)/√(r'*r+eps(Float64))^3)/(4π)

using BenchmarkTools
function hill_test(N,D=3N/4,U=(0,0,1);dist=4)
    uλ = hill_vortex(N;D)
    u = zeros(Float64,(N,N,N,3)); apply!(uλ,u)
    ω = ntuple(i->MLArray(u[:,:,:,1]),3); fill_ω!(ω,u)

    @btime m_u_ω(1,CartesianIndex(2,2,2),$ω,$dist)

    @inline sdf(I) = √sum(abs2,loc(0,I) .- (N-2)/2) - D/2
    ϵ(i,I,ω) = uλ(i,loc(0,I))-U[i]-m_u_ω(i,I,ω,dist)

    p = zeros(Float64,(N,1,N)); @inline J(I) = I+CartesianIndex(0,N÷2,0)
    WaterLily.@loop (p[I] = sdf(J(I))>1 ? √(ϵ(1,J(I),ω)^2+ϵ(2,J(I),ω)^2+ϵ(3,J(I),ω)^2) : 0) over I ∈ CartesianIndices(p)
    return p
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
@inline sdf(I) = √sum(abs2,loc(0,I) .- (N-2)/2) - D/2; @inline J(I) = I+CartesianIndex(0,N÷2,0)
stats(p,i) = pmap(maximum(p[I] for I in CartesianIndices(p) if dis[i-1]<sdf(J(I))≤dis[i]))

# data = []
# for dist ∈ 2 .^ collect(0:pow-1) # this takes hours
#     @show dist
#     p = hill_test(N,D;dist)
#     flood(pmap.(p[:,1,:]),clims=(-6,-1),border=:none,cfill=:Greens)
#     savefig("Hill_error_dist$(dist).png")
#     push!(data,[stats(p,i) for i in 2:lastindex(dis)])
# end
using JLD2
save_object("Hill_error.jld2",data)

colors = colormap("Blues",pow+2)
plt = plot(xlabel="d/D",ylabel="max(log10(|uₑ|/U))",ylims=(-6,-1));
for (dist,vec) in enumerate(data)
    plot!(plt,collect(dis)[2:end]./D,vec,label="log₂(size)=$(dist-1)",c=colors[2+dist])
end
plt
savefig("Hill_error_dists.png")

#Data for pow=6 duration = [1,3.400,12.300,56.100,262.600,903.500]
#Data for pow=7 duration = [3.225,8.433,33.600,152.600,775.000,3934,14421]
duration = [3.688,10.400,40.500,196.500,1090,6394,31528,111228]
plot(0:7,log10(duration[1]).-log10.(duration),xlabel="log₂(kernel size)",ylabel="log₁₀(speedup)",legend=false)
savefig("Hill_speedup_dists.png")
