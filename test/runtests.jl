using BiotSavartBCs
using Test
using WaterLily

using BiotSavartBCs: @loop,inside_u,restrict!,project!,down,front,step
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

using BiotSavartBCs: slice,interaction!
@testset "velocity.jl" begin
    # Hill ring vortex in 3D
    N = 2+2^5
    u = Array{Float32}(undef,(N,N,N,3)); apply!(hill_vortex(N),u)
    ω = zeros(Float32,N,N,N,3)

    fill_ω!(ω,u) # Ideally, ω₃=0 & |ωᵩ|N/U≤20, but ω is discontinuous...
    @test all(-0.25 .< extrema(ω[:,:,:,3]) .*N .< 0.25) # roughly 0
    @test 18 < maximum(ω)*N < 20 # roughly |20|
    @test abs(sum(ω)) < 1e-4 # zero total circulation

    N = 2+3*2^3; U=(0,0,1)
    u = Array{Float32}(undef,(N,N,N,3)); apply!(hill_vortex(N),u); u₀ = copy(u)
    ω = MLArray(zeros(Float32,N,N,N,3)); tar = collect_targets(ω); ftar = flatten_targets(tar);
    fill_ω!(ω,u)
    BC!(u,U) # mess up BCs
    biotBC!(u,U,ω,tar,ftar) # fix face uₙ
    pflowBC!(u) # fix ghosts

    tol = (0.02,0.02,0.048) # Hill vortex has largest uₙ on z faces
    for i in 1:3, s in (2,N)
        @test maximum(I->abs(u[I]-u₀[I]),slice(size(u),i,s)) < tol[i]
    end

    # Tangential ghosts are great
    @test maximum(abs,(u.-u₀)[3:end-1,2:end-1,1,1])<0.02
    @test maximum(abs,(u.-u₀)[3:end-1,2:end-1,end,1])<0.02
    @test maximum(abs,(u.-u₀)[2:end-1,3:end-1,1,2])<0.02
    @test maximum(abs,(u.-u₀)[2:end-1,3:end-1,end,2])<0.02
    @test maximum(abs,(u.-u₀)[1,2:end-1,3:end-1,3])<0.023
    @test maximum(abs,(u.-u₀)[end,2:end-1,3:end-1,3])<0.023

    # Normal ghost has lower accuracy (but it's the least important)
    for i in 1:3
        @test maximum(I->abs(u[I]-u₀[I]),slice(size(u),i,1)) < 0.06
    end
end

@testset "flow.jl" begin
    sphere(D,m=3D÷2) = Simulation((m,m,m), (1,0,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=D/1e4)
    sim = sphere(128); ω = MLArray(sim.flow.f); x₀ = copy(sim.flow.p); tar = collect_targets(ω); ftar = flatten_targets(tar);
    biot_mom_step!(sim.flow,sim.pois,ω,x₀,tar,ftar)
    @test abs(maximum(sim.flow.u[:,:,:,1])-1.5)<0.012    # u_max = 3/2
    @test abs(maximum(sim.flow.u[:,:,:,2:3])-0.75)<0.033 # v,w_max = 3/4
    @test minimum(sim.flow.u[2,:,:,1])-19/27<0.033       # upstream slow down
end