# Tree sum over sources for a target
Base.@propagate_inbounds @fastmath function tree(ml,Ti)
    i,T = last(Ti),front(Ti)
    x = shifted(T,i)+SVector{3,Float32}(T.I)

    #Top level
    ω = first(ml)
    val = zero(eltype(ω))
    domain = inside(size_u(ω)[1])
    Router = remaining(T,domain)
    Rinner = inR(Router,inside(size_u(ω)[1],buff=2))
    # Do everything remaining inside buff=2
    for S in Rinner
        val += weighted(x-SVector{3,Float32}(S.I),S,i,ω)
    end
    # @show Rinner, val

    # Loop down levels
    x = x .- 1.5f0 # adjust origin for scaling
    for l in 2:lastindex(ml)
        ω = ml[l]; T = down(T)
        domain = inside(size_u(ω)[1])
        Rinner = close(T,domain)
        Router = l == lastindex(ml) ? domain : remaining(T,domain)
        Rinner ≠ Router && for S in Router
            S ∉ Rinner && (val += weighted(x-(SVector{3,Float32}(S.I) .- 1.5f0)*2^(l-1),S,i,ω))
        end
        # @show T, Rinner, Router, val
    end; val
end

# Biot-Savart BC using the tree sum
treeBC!(u,U,ml,targets) = @vecloop tree_finish!(u,U,ml,Ii) over Ii ∈ targets
Base.@propagate_inbounds @fastmath function tree_finish!(u,U,ml,Ii)
    i,I = last(Ii),front(Ii)
    I.I[i]==1 && (I = I+δ(i,I)) # shift for "left" vector field face
    u[I,i] = U[i]+tree(ml,Ii)/Float32(4π)
end
function treeBC_r!(r,u,U,ml,targets)
    @vecloop tree_resid!(r,u,U,ml,Ii) over Ii ∈ targets
    fix_resid!(r)
end 
Base.@propagate_inbounds @fastmath function tree_resid!(r,u,U,ml,Ii)
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    uI = lower ? Ii+δ(i,Ii) : Ii

    # Update velocity and residual
    uₙ = U[i]+tree(ml,Ii)/Float32(4π)
    uₙ⁰ = u[uI]; u[uI] = uₙ
    sgn = lower ? -1 : 1; r[I-sgn*δ(i,I)] += sgn*(uₙ-uₙ⁰)
end
