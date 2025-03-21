using WaterLily,StaticArrays,CUDA,BiotSavartBCs
using JLD2,Plots

function ellipse(D,n,m,Λ=5.0;A₀=1.0,St=0.6,U=1,Re=100,T=Float32,mem=Array,use_biotsavart=false)
    h₀=T(A₀*D/2); ω=T(2π*St*U/D)
    sdf(x,t) = √sum(abs2,SA[x[1]/Λ,x[2]])-D÷2/Λ
    map(x,t) = x .- SA[n*D÷4,m*D÷2-h₀*sin(ω*t)]
    use_biotsavart && BiotSimulation((n*D,m*D), (U,0), D; body=AutoBody(sdf,map), ν=U*D/Re, T, mem)
    Simulation((n*D,m*D), (U,0), D; body=AutoBody(sdf,map), ν=U*D/Re, T, mem)
end
# parameters
params = [(false,0.2,30,20),(false,0.3,30,20),(false,0.4,30,20),(false,0.5,30,20),(false,0.6,30,20),
          (true,0.2,6,4),(true,0.3,6,4),(true,0.4,6,4),(true,0.5,6,4),(true,0.6,6,4),
          (true,0.5,12,8),(true,0.6,12,8),(false,0.6,60,40)]

# run all the cases
for (biot,St,n,m) in params
    sim = ellipse(64,n,m;St,T=Float64,mem=CUDA.CuArray,use_biotsavart=biot)
    R = biot ? inside(sim.flow.p) : CartesianIndices((290:1057,386:897)) # show the domain in 12L, 8L
    n==60 && (R=inside(sim.flow.p))
    forces = []
    anim = @animate for tᵢ in range(0.,100.,step=0.1)
        while sim_time(sim) < tᵢ
            sim_step!(sim;remeasure=true)
            pres,visc = WaterLily.pressure_force(sim),WaterLily.viscous_force(sim)
            push!(forces,[sim_time(sim),pres...,visc...])
        end
        println("tU/L=",round(sim_time(sim),digits=4),
                ", Δt=",round(sim.flow.Δt[end],digits=3))
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ[R]|>Array,clims=(-10,10))
    end
    gif(anim,"airfoil_wake_$(n)L_$(m)L_St$(St)_$(biot).gif")
    jldsave("airfoil_wake_$(n)L_$(m)L_St$(St)_$(biot).jld2"; f=forces)
end

function average_last(data,St,n,L;data_ids=[3,5])
    idx = floor(data[end,1]*St-n).<data[:,1]*St.<floor(data[end,1]*St) # last n cycles
    return sum(data[idx,data_ids])./(sum(idx)*L)
end

# make the figures for the paper
mean_CL = []; mean_CL2=[]; mean_CL_large=[]
blues = colormap("Blues", 8)[3:end] # Biot savart
reds = colormap("Oranges", 8)[3:end] # reflection
for St in [0.2,0.3,0.4,0.5,0.6]
    plt = plot(dpi=300)
    St==0.6 && (plt2 = plot(dpi=300))
    for biot ∈ [false true]
        L=64; (n,m) = biot ? (6,4) : (30,20) # select domain size
        jldopen("airfoil_wake_$(n)L_$(m)L_St$(St)_$(biot).jld2") do file
            BC,ls,color = ifelse(biot,("Biot-Savart  (6Lx4L)",:solid,blues),("Reflection (30Lx20L)",:dash,reds))
            forces = reduce(vcat,file["f"]')
            plot!(plt,forces[:,1]*St,sum(forces[:,[3,5]];dims=2)./L,label=BC;ls,c=color[Int(St*10-1)])
            (St==0.6 && !biot) && plot!(plt2,forces[:,1]*St,sum(forces[:,[3,5]];dims=2)./L,label=BC;ls,c=color[end])
            L_mean = average_last(forces,St,5,L)
            println("Case = ($(n)x$m) St=$St"*ifelse(biot," Biot",""))
            println("▷ CL mean = $L_mean")
            push!(mean_CL,L_mean)
        end
    end
    St==0.6 && jldopen("airfoil_wake_60L_40L_St$(St)_false.jld2") do file
        forces = reduce(vcat,file["f"]'); L=64
        L_mean = average_last(forces,St,5,L)
        push!(mean_CL_large,L_mean)
        println("Case = (60x40) St=$St")
        println("▷ CL mean = $L_mean")
        plot!(plt,forces[:,1]*St,sum(forces[:,[3,5]];dims=2)./L,label="Reflection (60Lx40L)";c=:Red)
        plot!(plt2,forces[:,1]*St,sum(forces[:,[3,5]];dims=2)./L,label="Reflection (60Lx40L)";c=:Red)
        ylims!(plt2,-12,12); xlims!(plt2,50,60); savefig(plt2,"convergence_domain.png")
    end
    if St in [0.5,0.6]
        jldopen("airfoil_wake_12L_8L_St$(St)_true.jld2") do file
            forces = reduce(vcat,file["f"]'); L=64
            L_mean = average_last(forces,St,5,L)
            plot!(plt,forces[:,1]*St,sum(forces[:,[3,5]];dims=2)./L,label="Biot-Savart (12Lx8L)";ls=:dot)
            println("Case = (12x8) St=$St Biot")
            println("▷ CL mean = $L_mean")
            push!(mean_CL2,L_mean)
        end
    end
    xlims!(0*St,100*St); ylims!(-12,12);
    title!("Deflected Wake Lift Forces");xlabel!("Cycles");ylabel!("2F₂/ρU²L")
    savefig("force_deflected_wake_St$St.png")
end

# mean lift convergence
let
    St = [0.2,0.3,0.4,0.5,0.6]
    plt = plot(dpi=300)
    plot!(plt,St,mean_CL[1:2:end],   label="Reflection (30Lx20L)";marker=:rect,ls=:dash,c=reds[3])
    scatter!(plt,[0.6],mean_CL_large,label="Reflection (60Lx40L)";marker=:o,c=reds[6])
    plot!(plt,St,mean_CL[2:2:end],   label="Biot-Savart   (6Lx4L)";marker=:s,ls=:solid,c=blues[3])
    plot!(plt,[0.5,0.6],mean_CL2,   label= "Biot-Savart (12Lx8L)";marker=:x,ls=:solid,c=blues[6])
    xlims!(0.,0.8); ylims!(-3.5,0.5); plot!(legend=:bottomleft)
    xlabel!("Strouhal number"); ylabel!("Mean Lift coefficient")
    savefig("CL_mean_deflected_wake.png")
end
# Poincaré map
using DSP
function mean_filter(data,idx;L=64,r=Lowpass(0.01),m=Butterworth(4))
    return filtfilt(digitalfilter(r, m),sum(data[4:end,idx];dims=2)./L)[:,1]
end

let
    plt = plot(dpi=300)
    for biot ∈ [false true]
        (n,m) = biot ? (6,4) : (30,20) # select domain size
        jldopen("airfoil_wake_$(n)L_$(m)L_St0.6_$(biot).jld2") do file
            BC,ls,color = ifelse(biot,("Biot-Savart",:solid,blues[4]),("Reflection",:dash,reds[4]))
            forces = reduce(vcat,file["f"]')
            plot!(plt,mean_filter(forces,[3,5]),mean_filter(forces,[2,4]),label=:none,lw=0.2,c=color)
        end
        biot==false && jldopen("airfoil_wake_60L_40L_St0.6_false.jld2") do file
            forces = reduce(vcat,file["f"]')
            plot!(plt,mean_filter(forces,[3,5]),mean_filter(forces,[2,4]),label=:none,lw=0.2,c=reds[6])
        end
    end
    # the same for 2x Biot-Savart
    jldopen("airfoil_wake_12L_8L_St0.6_true.jld2") do file
        forces = reduce(vcat,file["f"]')
        plot!(plt,mean_filter(forces,[3,5]),mean_filter(forces,[2,4]),label=:none,lw=0.2,c=blues[6])
    end
    # get legend better
    plot!(plt,[-100,-100],[0.,0.],label="Reflection (30Lx20L)",lw=1.5,color=reds[4])
    plot!(plt,[-100,-100],[0.,0.],label="Reflection (60Lx40L)",lw=1.5,color=reds[6])
    plot!(plt,[-100,-100],[0.,0.],label="Biot-Savart   (6Lx4L)",lw=1.5,color=blues[4])
    plot!(plt,[-100,-100],[0.,0.],label="Biot-Savart (12Lx8L)",lw=1.5,color=blues[6])
    xlims!(-15,15);ylims!(-1.5,1.5);
    ylabel!("Drag coefficient");xlabel!("Lift coefficient")
    savefig("poincare_deflected_wake.png")
end