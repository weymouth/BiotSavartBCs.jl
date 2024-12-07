using BiotSavartBCs
using Test
using WaterLily

using BiotSavartBCs: @vecloop,inside_u,restrict!,project!,down,front,step
@testset "util.jl" begin
    a = zeros(Int,(4,4,6,3))
    @vecloop a[I] += 1 over I in inside_u(a,buff=2)
    @test sum(a) == 0
    @vecloop a[I] += 1 over I in inside_u(a)
    @test sum(a) == length(inside_u(a)) == 2*2*4*3

    a = zeros(Int,(10,10,18,3))
    ml=MLArray(a)
    @test length(ml)==3
    @vecloop a[I] += 1 over I in inside_u(a)
    @test sum(first(ml)) == length(inside_u(a))
    restrict!(ml)
    @test sum(last(ml)) == length(inside_u(a))

    tar = collect_targets(ml)
    @test length(tar[1]) == 4length(tar[2]) == 16length(tar[3])
    Ti = last(tar[2])
    T,i = front(Ti),last(Ti)
    @test CartesianIndex(down(T),i)==last(tar[3])
    
    @vecloop ml[3][I] += 16 over I in tar[3]
    project!(ml,tar)
    @test ml[2][Ti] == 4

    @test length(flatten_targets(tar)) == sum(length,tar)
    @test flatten_targets(tar)[sum(length,tar[1:2])] == (2,Ti)

    a = zeros(Int,(34,34,2))
    ml=MLArray(a)
    @test length(ml)==3 # much bigger dis in 2D
    @vecloop a[I] += 1 over I in inside_u(a)
    @test sum(first(ml)) == length(inside_u(a))
    restrict!(ml)
    @test sum(last(ml)) == length(inside_u(a))

    tar = collect_targets(ml)
    @test length(tar[1]) == 2length(tar[2]) == 4length(tar[3])
    Ti = last(tar[2])
    T,i = front(Ti),last(Ti)
    @test CartesianIndex(down(T),i)==last(tar[3])
    
    @vecloop ml[3][I] += 4 over I in tar[3]
    project!(ml,tar)
    @test ml[2][Ti] == 2
end

using SpecialFunctions,ForwardDiff
function lamb_dipole(N;D=3N/4,U=1)
    β = 2.4394π/D
    C = -2U/(β*besselj0(β*D/2))
    function ψ(x,y)
        r = √(x^2+y^2)
        ifelse(r ≥ D/2, U*((D/2r)^2-1)*y, C*besselj1(β*r)*y/r)
    end
    return function uλ(i,xy)
        x,y = xy .- (N-2)/2
        ifelse(i==1,ForwardDiff.derivative(y->ψ(x,y),y)+1+U,-ForwardDiff.derivative(x->ψ(x,y),x))
    end
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

using BiotSavartBCs: slice
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

    # Check domain uₙ using FMM-version of Biot-Savart BCs
    biotBC!(u,U,ω,tar,ftar)
    tol = (0.0222,0.0222,0.05) # Hill vortex has largest uₙ on z faces
    for i in 1:3, s in (2,N)
        mx = maximum(I->abs(u[I]-u₀[I]),slice(size(u),i,s))
        # @show i,s,mx
        @test mx < tol[i]
    end

    # Tangential ghosts are great
    pflowBC!(u) # fix ghosts
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

    # Check domain uₙ using tree-version of Biot-Savart BCs
    BC!(u,U) # mess up BCs
    biotBC!(u,U,ω,tar,ftar,fmm=false) # tree
    tol = (0.004,0.004,0.02) # No target interpolation error!
    for i in 1:3, s in (2,N)
        mx = maximum(I->abs(u[I]-u₀[I]),slice(size(u),i,s))
        # @show i,s,mx
        @test mx < tol[i]
    end

    pow = 5; N = 2+2^pow; U = (1,0)
    u = Array{Float32}(undef,(N,N,2)); apply!(lamb_dipole(N),u); u₀ = copy(u)
    ω = MLArray(zeros(Float32,N,N,2)); tar = collect_targets(ω); ftar = flatten_targets(tar);

    fill_ω!(ω,u)
    @test all(ω[1][:,:,2].==0) # we don't use the second component
    @test all(ω[1][[2,N-1],:,1].==0) # no vorticity outside the bubble
    @test all(@. abs(sum(ω))<12e-5) # zero-sum at every level

    BC!(u,U) # mess up boundaries
    biotBC!(u,U,ω,tar,ftar;fmm=true) # fix domain velocities
    @test maximum(abs,(u.-u₀)[2:end,2:end-1,1])<0.028
    @test maximum(abs,(u.-u₀)[2:end-1,2:end,2])<0.025
    
    BC!(u,U) # mess up boundaries
    biotBC!(u,U,ω,tar,ftar;fmm=false) # fix domain velocities
    @test maximum(abs,(u.-u₀)[2:end,2:end-1,1])<0.0063 # No target interpolation error
    @test maximum(abs,(u.-u₀)[2:end-1,2:end,2])<0.003
    pflowBC!(u) # fix ghosts
    @test maximum(abs,(u.-u₀)[3:end-1,1,1])<0.0044 # tangential
    @test maximum(abs,(u.-u₀)[1,3:end-1,2])<0.003 # tangential
    @test maximum(abs,(u.-u₀)[1,3:end-2,1])<0.0064 # normal
    @test maximum(abs,(u.-u₀)[3:end-2,1,2])<0.003 # normal
end

@testset "flow.jl" begin
    circ(D,U=1,m=2D) = Simulation((m,m), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=U*D/1e4)
    for fmm in (true,false)
        sim = circ(256); ω = MLArray(sim.flow.f); x₀ = copy(sim.flow.p); tar = collect_targets(ω); ftar = flatten_targets(tar);
        biot_mom_step!(sim.flow,sim.pois,ω,x₀,tar,ftar;fmm)
        u_max = maximum(sim.flow.u[:,:,1])
        v_max = maximum(sim.flow.u[:,:,2])
        u_inf = minimum(sim.flow.u[1,:,1])
        @show fmm,u_max,v_max,u_inf
        @test abs(u_max-2)<0.02 # circle u_max = 2
        @test abs(v_max-1)<0.02 # circle v_max = 1
        @test abs(u_inf-0.75)<0.02 # upstream slow down
        @time biot_mom_step!(sim.flow,sim.pois,ω,x₀,tar,ftar;fmm)
        @show sim.pois.n
    end

    sphere(D,m=3D÷2) = Simulation((m,m,m), (1,0,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=D/1e4)
    for fmm in (true,false)
        sim = sphere(128); ω = MLArray(sim.flow.f); x₀ = copy(sim.flow.p); tar = collect_targets(ω); ftar = flatten_targets(tar);
        biot_mom_step!(sim.flow,sim.pois,ω,x₀,tar,ftar;fmm)
        u_max = maximum(sim.flow.u[:,:,:,1])
        v_max = maximum(sim.flow.u[:,:,:,2:3])
        u_inf = minimum(sim.flow.u[2,:,:,1])
        @show fmm,u_max,v_max,u_inf
        @test abs(u_max-1.5)<0.012    # u_max = 3/2
        @test abs(v_max-0.75)<0.035   # v,w_max = 3/4
        @test abs(u_inf-19/27)<0.033  # upstream slow down
        @time biot_mom_step!(sim.flow,sim.pois,ω,x₀,tar,ftar;fmm)
        @show sim.pois.n
    end
end