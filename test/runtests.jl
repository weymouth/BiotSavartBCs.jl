using BiotSavartBCs
using Test
using WaterLily

using BiotSavartBCs: @loop,inside_u,restrict!,project!,down,front
@testset "util.jl" begin
    a = zeros(Int,(4,4,6,3))
    @loop a[I] += 1 over I in inside_u(a,buff=2)
    @test sum(a) == 0
    @loop a[I] += 1 over I in inside_u(a)
    @test sum(a) == length(inside_u(a)) == 2*2*4*3

    a = zeros(Int,(10,10,18,3))
    ml=MLArray(a)
    @test length(ml)==3
    @loop a[I] += 1 over I in inside_u(a)
    @test sum(first(ml)) == length(inside_u(a))
    restrict!(ml)
    @test sum(last(ml)) == length(inside_u(a))

    tar = collect_targets(ml)
    @test length(tar[1]) == 4length(tar[2]) == 16length(tar[3])
    Ti = last(tar[2])
    T,i = front(Ti),last(Ti)
    @test CartesianIndex(down(T),i)==last(tar[3])
    
    @loop ml[3][I] += 4 over I in tar[3]
    project!(ml,tar)
    @test ml[2][Ti] == 1

    @test length(flatten_targets(tar)) == sum(length,tar)
    @test flatten_targets(tar)[sum(length,tar[1:2])] == (2,Ti)
end

function hill_vortex(N;D=3N/4)
    return function uλ(i,xyz)
        q = xyz .- (N-2)/2; x,y,z = q; r = √(q'*q); θ = acos(z/r); ϕ = atan(y,x)
        v_r = ifelse(2r<D,-1.5*(1-(2r/D)^2),1-(D/2r)^3)*cos(θ)
        v_θ = ifelse(2r<D,1.5-3(2r/D)^2,-1-0.5*(D/2r)^3)*sin(θ)
        i==1 && return sin(θ)*cos(ϕ)*v_r+cos(θ)*cos(ϕ)*v_θ
        i==2 && return sin(θ)*sin(ϕ)*v_r+cos(θ)*sin(ϕ)*v_θ
        cos(θ)*v_r-sin(θ)*v_θ
    end
end

@testset "vorticity.jl" begin
    # Hill ring vortex in 3D
    N = 2+2^5
    u = Array{Float32}(undef,(N,N,N,3)); apply!(hill_vortex(N),u)
    ω = zeros(Float32,N,N,N,3)

    fill_ω!(ω,u) # Ideally, ω₃=0 & |ωᵩ|N/U≤20, but ω is discontinuous...
    @test all(-0.25 .< extrema(ω[:,:,:,3]) .*N .< 0.25) # roughly 0
    @test 18 < maximum(ω)*N < 20 # roughly |20|
    @test abs(sum(ω)) < 1e-4 # zero total circulation

    # hydrostatic p on an immersed sphere
    p = Array{Float32}(undef,(N,N,N))
    @loop p[I] = -loc(0,I)[1] over I ∈ CartesianIndices(p)
    sdf(x) = √sum(abs2,x .-(N-2)/2)-N/4
    apply!((i,x)->WaterLily.μ₀(sdf(x),1),u) # overwrite u with μ₀
    fill_ω!(ω,u,p)
    @test all(extrema(ω[:,:,:,2]).≈(-0.5,0.5)) # dμ₀/dx for ϵ=1
    @test all(extrema(ω[:,:,:,3]).≈(-0.5,0.5)) # dμ₀/dx for ϵ=1
    @test all(sum(abs2,ω[I,:])<eps() for I ∈ inside(p) if abs(sdf(loc(0,I)))>2.1)  # ω=0 outside smoothing region
end

using BiotSavartBCs: slice,interaction!
@testset "velocity.jl" begin
    N = 2+2^6; U=(0,0,1)
    hill = hill_vortex(N)
    u = Array{Float32}(undef,(N,N,N,3)); apply!(hill,u); u_max = maximum(abs,u)
    ω = MLArray(zeros(Float32,N,N,N,3)); tar = collect_targets(ω); ftar = flatten_targets(tar)

    fill_ω!(ω,u); fill!(u,0f0); biotBC!(u,U,ω,tar,ftar)
    L∞ = L₂ = 0f0
    for i ∈ 1:3, s ∈ (2,N), I ∈ slice(size(u),i,s)
        L₂ += (u[I]-hill(i,loc(I)))^2
        L∞ = max(L∞ ,abs(u[I]-hill(i,loc(I))))
    end
    @test sqrt(L₂/length(tar[1]))/u_max<0.024
    @test L∞/u_max < 0.1
end

@testset "util.jl" begin
    pow = 4; N = 2+2^pow; U=(1,0)
    u,ω = lamb_uω(N); u₀ = copy(u)
    BC!(u,U) # mess up boundaries
    biotBC!(u,U,ω) # fix domain velocities
    @test maximum(abs,(u.-u₀)[2:end,2:end-1,1])<0.003
    @test maximum(abs,(u.-u₀)[2:end-1,2:end,2])<0.003
    pflowBC!(u) # fix ghosts
    @test maximum(abs,(u.-u₀)[2:end-1,1,1])<0.0044 # tangential
    @test maximum(abs,(u.-u₀)[1,2:end-1,2])<0.003 # tangential
    @test maximum(abs,(u.-u₀)[1,2:end-1,1])<0.003 # normal
    @test maximum(abs,(u.-u₀)[2:end-1,1,2])<0.003 # normal

    U=(0,0,1)
    u,ω = hill_uω(N); u₀ = copy(u)
    BC!(u,U) # mess up boundaries
    biotBC!(u,U,ω) # fix domain velocities
    @test maximum(abs,(u.-u₀)[2:end,2:end-1,2:end-1,1])<0.02
    @test maximum(abs,(u.-u₀)[2:end-1,2:end,2:end-1,2])<0.02
    @test maximum(abs,(u.-u₀)[2:end-1,2:end-1,2:end,3])<0.02
    pflowBC!(u) # fix ghosts
    @test maximum(abs,(u.-u₀)[2:end-1,1,2:end-1,3])<0.02 # tangential
    @test maximum(abs,(u.-u₀)[1,2:end-1,2:end-1,3])<0.02 # tangential
    @test maximum(abs,(u.-u₀)[2:end-1,2:end-1,1,3])<0.02 # normal

    r = zeros(Float32,(N,N,N)); @inside r[I] = WaterLily.div(I,u)
    fix_resid!(r)
    @test sum(r)<1e-5
end

@testset "flow.jl" begin
    circ(D,U=1,m=2D) = Simulation((m,m), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=U*D/1e4)
    sim = circ(256); ω = MLArray(sim.flow.σ);
    biot_mom_step!(sim.flow,sim.pois,ω)
    @test abs(maximum(sim.flow.u[:,:,1])-2)<0.02 # circle u_max = 2
    @test abs(maximum(sim.flow.u[:,:,2])-1)<0.02 # circle v_max = 1
    ϕᵢ,ϕₒ=extrema(sim.flow.u[:,end,2])
    @test ϕₒ>0.1                                 # side outflow
    @test abs(ϕₒ+ϕᵢ)<2e-5                        # symmetric in/outflow
    @test minimum(sim.flow.u[1,:,1])-8/10<0.01    # upstream slow down

    sphere(D,U=1,m=2D) = Simulation((m,m,m), (U,0,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=U*D/1e4)
    sim = sphere(128); ω = ntuple(i->MLArray(sim.flow.σ),3);
    biot_mom_step!(sim.flow,sim.pois,ω)
    @test abs(maximum(sim.flow.u[:,:,:,1])-1.5)<0.02    # circle u_max = 3/2
    @test abs(maximum(sim.flow.u[:,:,:,2:3])-0.75)<0.04 # circle v,w_max = 3/4
    @test minimum(sim.flow.u[1,:,:,1])-8/9<0.01       # upstream slow down
end