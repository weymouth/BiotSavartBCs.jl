using WaterLily,StaticArrays,CUDA,BiotSavartBCs,WriteVTK

function circ(D,n,m;Re=200,U=1,mem=CUDA.CuArray)
    body = AutoBody((x,t)->√sum(abs2,SA[x[1],x[2]].-m*D÷2)-D÷2)
    Simulation((n*D,m*D,8), (U,0,0), D; body,ν=U*D/Re,mem,perdir=(3,))
end

# symmetric BCs
import WaterLily: size_u,slice,div,∂,@loop,CIj
function BiotSavartBCs.biotBC!(u,U,ω)
    N,n = size_u(u)
    # z-direction is periodic
    for i ∈ 1:2, s ∈ (2,N[i]) # Domain faces, biotsavart+background
        @loop u[I,i] = u_ω(i,I,ω)+U[i] over I ∈ slice(N,s,i)
    end
    for i ∈ 3:3, s ∈ (2,N[i]) # Domain faces, biotsavart+background
        @loop u[I,i] = U[i] over I ∈ slice(N,s,i)
    end
    # periodic direction, all components at once
    # for i ∈ 1:n
    #     @loop u[I,i] = u[CIj(3,I,N[3]-1),i] over I ∈ slice(N,1,3)
    #     @loop u[I,i] = u[CIj(3,I,2),i] over I ∈ slice(N,N[3],3)
    # end
end
function BiotSavartBCs.pflowBC!(u)
    N,n = size_u(u)
    for i ∈ 1:3
        for j ∈ 1:3 # Tangential direction ghosts, curl=0
            j==i && continue
            @loop u[I,j] = u[I+δ(i,I),j]-∂(j,CartesianIndex(I+δ(i,I),i),u) over I ∈ slice(N.-1,1,i,2)
            @loop u[I,j] = u[I-δ(i,I),j]+∂(j,CartesianIndex(I,i),u) over I ∈ slice(N.-1,N[i],i,2)
        end
        # Normal direction ghosts, div=0
        @loop u[I,i] += div(I,u) over I ∈ slice(N.-1,1,i,2)
    end
    # periodic direction, all components at once
    # for i ∈ 1:n
    #     @loop u[I,i] = u[CIj(3,I,N[3]-1),i] over I ∈ slice(N,1,3)
    #     @loop u[I,i] = u[CIj(3,I,2),i] over I ∈ slice(N,N[3],3)
    # end
end
function BiotSavartBCs.fix_resid!(r)
    N = size(r); n = length(N); @inline A(i) = 2prod(N.-2)/(N[i]-2)
    # the residuals cannot include the periodic boundaries
    res = sum(r)/sum(A,1:2)
    for i ∈ 1:2 # don't apply the correction z-direction
        @loop r[I] -= res over I ∈ slice(N.-1,2,i,2)
        @loop r[I] -= res over I ∈ slice(N.-1,N[i]-1,i,2)
    end
end

# update domain velocity and residual
function BiotSavartBCs.update_resid!(r,u,u_ϵ,ω_ϵ)
    N,n = size_u(u); inN(I,N) = all(@. 2 ≤ I.I ≤ N-1)
    for i ∈ 1:2 # don't apply the correction z-direction
        @loop (u_ϵ[I]=u_ω(i,I,ω_ϵ); u[I,i]+=u_ϵ[I]; inN(I,N) && (r[I]-=u_ϵ[I])) over I ∈ slice(N,2,i)
        # update the residuals on the bottom left and right boundaries
        @loop (u_ϵ[I]=u_ω(i,I,ω_ϵ); u[I,i]+=u_ϵ[I]; inN(I-δ(i,I),N) && (r[I-δ(i,I)]+=u_ϵ[I])) over I ∈ slice(N,N[i],i)
    end
    fix_resid!(r)
end
# import WaterLily: perBC!,residual!,smooth!,Vcycle!,L₂
# import BiotSavartBCs: ml_restrict!,point,update_resid!,fix_resid!,biotBC!,pflowBC!
# function BiotSavartBCs.biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ω,U;w=1,log=false,tol=1e-6,itmx=32) where n    
#     dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
#     # Apply periodic BCs to the velocity field
#     N,_ = size_u(a.u)
#     for i ∈ 1:n
#         @loop a.u[I,i] = a.u[CIj(3,I,N[3]-1),i] over I ∈ slice(N,1,3)
#         @loop a.u[I,i] = a.u[CIj(3,I,2),i] over I ∈ slice(N,N[3],3)
#     end
#     fill_ω!(ω,a.u,a.μ₀,a.p)       # Compute ω=∇×(u-μ₀∇p)
#     # Apply periodic BCs to the vorticity field at the top level
#     foreach(i->(perBC!(ω[i][1],(3,));ml_restrict!(ω[i])),1:3)
#     biotBC!(a.u,U,ω)            # Apply domain BCs

#     b = ml_b.levels[1]
#     @inside b.z[I] = div(I,a.u)   # Set σ=∇⋅u
#     residual!(b); fix_resid!(b.r) # Set r=Ax-σ, and ensure sum(r)=0

#     r₂ = L₂(b); nᵖ = 0; x₀ = point(ω)
#     while nᵖ<itmx
#         x₀ .= b.x                 # Remember current solution
#         Vcycle!(ml_b); smooth!(b) # Improve solution
#         b.ϵ .= b.x .-x₀; x₀ .= 0  # soln update: ϵ = x-x₀
#         fill_ω!(ω,a.μ₀,b.ϵ)       # vort update: Δω = -∇×μ₀∇ϵ
#         foreach(i->(perBC!(ω[i][1],(3,));ml_restrict!(ω[i])),1:3)
#         update_resid!(b.r,a.u,b.z,ω) # Update domain BC and resid
#         r₂ = L₂(b); nᵖ+=1
#         log && @show nᵖ,r₂
#         r₂<tol && break
#     end
#     push!(ml_b.n,nᵖ)
#     # (nᵖ<2 && length(ml_b.levels)>5) && pop!(ml_b.levels); # remove coarsest level if this was easy
#     # (nᵖ>4 && divisible(ml_b.levels[end])) && push!(ml_b.levels,restrictML(ml_b.levels[end])) # add a level if this was hard
    
#     for i ∈ 1:n   # Project u -= μ₀∇p
#         @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
#     end
#     pflowBC!(a.u) # Update ghost BCs (domain is already correct)
#     a.p ./= dt    # Rescale pressure
# end

# import BiotSavartBCs: inR,up,r,clamp
# # overwrites biot Savart kernel for symmetric BCs, add image vortices
# function BiotSavartBCs._u_ω(x,dis,l,R,biotsavart,u=0f0)
#     # loop levels
#     while l>1
#         # find Region close to x
#         dx = 2f0^(l-1)
#         Rclose = inR(x/dx .-dis,R):inR(x/dx .+dis,R)
#         Rimage = imgR(x/dx,dis,R)

#         # get contributions outside Rclose
#         R ≠ Rclose && for I ∈ R
#             I ∉ Rclose && (u += biotsavart(r(x,I,dx),I,l))
#             # add contribution outside of the image of Rclose, vorticity is mirrored
#             I ∉ Rimage && (u -= biotsavart(r2(x,I,dx,R),I,l))
#         end

#         # move "up" one level within RcloseFi
#         l -= 1
#         R = first(up(first(Rclose))):last(up(last(Rclose)))
#     end

#     # top level contribution
#     Rimage = imgR(x,dis,R)
#     for I ∈ R
#         u += biotsavart(r(x,I),I)
#         I ∈ Rimage && (u -= biotsavart(r2(x,I,1,Rimage),I))
#     end; u
# end
# # mirror the point x in the image of R
# r2(x,I,dx,R) = x-dx*(SA_F32[I.I[1],2last(R)[2]-I.I[2]-1] .- 1.5f0)
# function imgR(x,dis,R)
#     lower = round.(Int,x.-dis .+ 1.5f0)
#     upper = round.(Int,x.+dis .+ 1.5f0)
#     s₁ = clamp(lower[1],R.indices[1])
#     e₁ = clamp(upper[1],R.indices[1])
#     s₂ = upper[2]>last(R)[2] ? 2last(R)[2]-upper[2]-1 : -1 # if the box doesn't overlap in the image, return an empty box
#     e₂ = upper[2]>last(R)[2] ? last(R)[2]-1 : -1
#     return CartesianIndices((s₁:e₁,s₂:e₂))
# end

# make a writer with some attributes, need to output to CPU array to save file (|> Array)
_velocity(a::Simulation) = a.flow.u |> Array;
_pressure(a::Simulation) = a.flow.p |> Array;
vort(a::Simulation) = (@WaterLily.loop a.flow.f[I,:] .= WaterLily.ω(I,a.flow.u) over I ∈ inside(a.flow.p);
                       a.flow.f |> Array)
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); 
                                     a.flow.σ |> Array;)
lamda(a::Simulation) = (@WaterLily.loop a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u) over I ∈ inside(a.flow.p);
                        a.flow.σ |> Array;)

custom_attrib = Dict(
    "u" => _velocity,
    "p" => _pressure,
    "ω" => vort,
    "b" => _body,
    "λ₂" => lamda
)# this maps what to write to the name in the file
# make the writer
writer = vtkWriter("PeriodicCylinder"; attrib=custom_attrib)


CIs = CartesianIndices
R = 16
use_biotsavart = true
sim = circ(2R,4,2;Re=500,U=1,mem=Array)
ω = ntuple(i->MLArray(sim.flow.σ),3)

forces = []
for t in range(0,0.1;step=0.05)#1:6
    while sim_time(sim)<t #sim_step!(sim,t)
        use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : mom_step!(sim.flow,sim.pois)
        f = 2WaterLily.pressure_force(sim)/R
        push!(forces,[sim_time(sim),f[1]])
    end
    write!(writer,sim);
    @show t
    flush(stdout)
end
close(writer)