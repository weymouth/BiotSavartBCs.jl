using StaticArrays

import WaterLily: up
function _u_ω(x,dis,l,R,biotsavart,u=0f0)
    # loop levels
    while l>1
        # find Region close to x
        dx = 2f0^(l-1)
        Rclose = inR(x/dx .-dis,R):inR(x/dx .+dis,R)

        # get contributions outside Rclose
        for I ∈ R
            !(I ∈ Rclose) && (u += biotsavart(r(x,I,dx),I,l))
        end

        # move "up" one level within Rclose
        l -= 1
        R = first(up(first(Rclose))):last(up(last(Rclose)))
    end

    # top level contribution
    for I ∈ R
        u += biotsavart(r(x,I),I)
    end; u
end
r(x,I::CartesianIndex,dx=1) = x-dx*(SA_F32[I.I...] .- 1.5f0) # faster than loc(0,I,Float32)
inR(x,R) = clamp(CartesianIndex(round.(Int,x .+ 1.5f0)...),R)
Base.clamp(I::CartesianIndex,R::CartesianIndices) = CartesianIndex(clamp.(I.I,first(R).I,last(R).I))

u_ω(i,I::CartesianIndex{2},ω) = _u_ω(loc(i,I,Float32),7,lastindex(ω),inside(ω[end]),
    @inline (r,I,l=1) -> @inbounds(ω[l][I]*r[i%2+1])/(r'*r))*(2i-3)/Float32(2π)
u_ω(i,I::CartesianIndex{3},ω) = _u_ω(loc(i,I,Float32),3,lastindex(ω[1]),inside(ω[1][end]),
    @inline (r,I,l=1) -> permute((j,k)->@inbounds(ω[j][l][I]*r[k]),i)/√(r'*r)^3)/Float32(4π)
