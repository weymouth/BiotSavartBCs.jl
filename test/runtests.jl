using BiotSavartBCs
using Test
using WaterLily,CUDA,SpecialFunctions,ForwardDiff,StaticArrays
CUDA.allowscalar(false)

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

@testset "vorticity.jl" begin
    pow = 5; N = 2+2^pow
    u = Array{Float32}(undef,(N,N,2)); p = Array{Float32}(undef,(N,N));
    
    # lamb dipole
    apply!(lamb_dipole(N),u)
    ω = MLArray(u[:,:,1])
    @test length(ω)==pow # correct number of levels

    fill_ω!(ω,u)
    @test all(ω[1][[2,N-1],:].==0) # no vorticity outside the bubble
    @test all(@. abs(sum(ω))<1e-5) # zero-sum at every level
    @test allequal(abs.(ω[pow][inside(ω[pow])])) # center = 0

    # hydrostatic p on an immersed circle
    WaterLily.@loop p[I] = -loc(0,I)[2] over I ∈ inside(p,buff=0)
    apply!((i,x)->WaterLily.μ₀(√sum(abs2,x .-(N-2)/2)-N/4,1),u) # overwrite u with μ₀
    fill_ω!(ω,u,p)
    @test all(extrema(ω[1]).≈(-0.5,0.5)) # dμ₀/dx for ϵ=1
    WaterLily.@loop p[I] = √sum(abs2,loc(0,I) .-(N-2)/2)-N/4 over I ∈ inside(p,buff=0) # overwrite p with d
    @test all(abs(ω[1][I])==0 for I ∈ inside(p) if p[I]>2.1 || p[I]<-2.1) # ω=0 outside smoothing region
    
    # Hill ring vortex in 3D
    u = Array{Float32}(undef,(N,N,N,3)); apply!(hill_vortex(N),u)
    ω = ntuple(i->MLArray(u[:,:,:,1]),3)
    @test all(@. length(ω)==pow) # correct number of levels

    fill_ω!(ω,u) # Ideally, ω₃=0 & |ωᵩ|N/U≤20, but ω is discontinuous...
    @test all(x->abs(N/20*x)<0.01,extrema(ω[3][1][inside(ω[3][1])]))  # ω₃ ≈ 0
    @test allequal(round.(Int,abs.(ω[1][pow][inside(ω[1][pow])]))) # center = 0
    @test allequal(round.(Int,abs.(ω[2][pow][inside(ω[2][pow])]))) # center = 0   
end

@testset "velocity.jl" begin
    @test true
end

@testset "util.jl" begin
    @test true
end

@testset "flow.jl" begin
    @test true
end