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
    p = 5; N = 2+2^p
    u = Array{Float32}(undef,(N,N,2)); apply!(lamb_dipole(N),u)
    ω = MLArray(u[:,:,1])
    @test length(ω)==p # correct number of levels

    fill_ω!(ω,u)
    @test all(ω[1][[2,N-1],:].==0) # no vorticity outside the circle
    @test all(@. abs(sum(ω))<1e-5) # zero-sum at every level
    @test allequal(abs.(ω[p][inside(ω[p])])) # center = 0

    u = Array{Float32}(undef,(N,N,N,3)); apply!(hill_vortex(N),u)
    ω = ntuple(i->MLArray(u[:,:,:,1]),3)
    @test all(@. length(ω)==p) # correct number of levels

    fill_ω!(ω,u) # Ideally, ω₃=0 & |ωᵩ|N/U≤20, but ω is discontinuous...
    @test all(x->abs(N/20*x)<0.01,extrema(ω[3][1][inside(ω[3][1])]))  # ω₃ ≈ 0
    @test allequal(round.(Int,abs.(ω[1][p][inside(ω[1][p])]))) # center = 0
    @test allequal(round.(Int,abs.(ω[2][p][inside(ω[2][p])]))) # center = 0

    # need pressure tests
    p = Array{Float32}()
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