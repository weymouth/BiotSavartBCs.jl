function ring(xyz;R=1,Ω=10)
    q = xyz; x,y,z = q; r = √(q'*q); θ = acos(z/r); ϕ = atan(y,x)
    ρ = r/R; A,I₀,I₁ = exp(-(ρ^2-1)*Ω/2)/exp(Ω),besseli(0,ρ*Ω),besseli(1,ρ*Ω)
    @show A,I₀,I₁
    @show ρ,cot(θ),cos(θ)
    v_r,v_θ = A/ρ*cot(θ)*I₁,A*Ω*(ρ*I₁-I₀)
    return v_r,v_θ
end
uλ2(0,[0,0,0.5])
plot(0:0.01:3,x->uλ2(0,[x,0,0.5])[1])
