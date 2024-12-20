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
          (true,0.5,12,8),(true,0.6,12,8)]
# run all the cases
for (biot,St,n,m) in params
    sim = ellipse(64,n,m;St,mem=CUDA.CuArray,use_biotsavart=biot)
    R = biot ? inside(sim.flow.p) : CartesianIndices((290:1057,386:897)) # show the domain in 12L, 8L
    forces = []
    @time anim = @animate for tᵢ in range(0.,100.,step=0.1)
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

# make the figures for the paper
mean_CL = []; mean_CL2=[]
for St in [0.2,0.3,0.4,0.5,0.6]
    (n,m) = biot ? (6,4) : (30,20) # select domain size
    plt = plot(dpi=300)
    for biot ∈ [true false]
        jldopen("airfoil_wake_$(n)L_$(m)L_St$(St)_$(biot).jld2") do file
            BC = ifelse(biot,"Biot-Savart","Reflection")
            ls = ifelse(biot,:solid,:dash)
            forces = reduce(vcat,file["f"]'); L=64
            plot!(plt,forces[:,1]*St,sum(forces[:,[3,5]];dims=2)./L,label="Airfoil St:$St "*BC)
            L_mean = sum(forces[end-20000:end,[3,5]])./20000L
            push!(mean_CL,L_mean)
            plot!([0,maximum(forces[:,1]*St)],[L_mean,L_mean],color=:black,label="Mean force "*BC;ls)
        end
    end
    if St in [0.5,0.6]
        jldopen("airfoil_wake_12L_8L_St$(St)_true.jld2") do file
            forces = reduce(vcat,file["force"]'); L=64
            plot!(plt,forces[:,1]*St,sum(forces[:,[3,5]];dims=2)./L,label="Airfoil St:$St Biot-Savart 2x";ls=:solid)
            L_mean = sum(forces[end-20000:end,[3,5]])./20000L
            push!(mean_CL2,L_mean)
            plot!([0,maximum(forces[:,1]*St)],[L_mean,L_mean],color=:black,label="Mean force Biot-Savart 2x";ls=:solid)
        end
    end
    xlims!(0,100*St);ylims!(-25,25);
    title!("Deflected Wake Lift Forces");xlabel!("Cycles");ylabel!("2Force/ρUR")
    savefig("force_deflected_wake_St$St.png")
end

# mean lift convergence
let
    St = [0.2,0.3,0.4,0.5,0.6]
    plt = plot(dpi=300)
    plot!(plt,St,mean_CL[2:2:end],label="Cₗ reflection";marker=:x,ls=:dash)
    plot!(plt,St,mean_CL[1:2:end],label="Cₗ Biot-Savart";marker=:o,ls=:solid)
    plot!(plt,[0.5,0.6],mean_CL2,label="Cₗ Biot-Savart 2x";marker=:s,ls=:solid)
    xlims!(0.,0.8); ylims!(-0.5,2); plot!(legend=:topleft)
    xlabel!("Strouhal number"); ylabel!("Mean Lift coefficient")
    savefig("CL_mean_deflected_wake.png")
    end
# Poincaré map
let
    using DSP
    responsetype = Lowpass(0.01)
    designmethod = Butterworth(4)
    plt = plot(dpi=300)
    for biot ∈ [true false]
        (n,m) = biot ? (6,4) : (30,20) # select domain size
        St=0.6; L=64
        jldopen("airfoil_wake_$(n)L_$(m)L_St$(St)_$(biot).jld2") do file
            BC = ifelse(biot,"Biot-Savart","Reflection")
            ls = ifelse(biot,:solid,:dash)
            forces = reduce(vcat,file["f"]')
            Lift = filtfilt(digitalfilter(responsetype, designmethod), sum(forces[4:end,[3,5]];dims=2)./L)
            Drag = filtfilt(digitalfilter(responsetype, designmethod), sum(forces[4:end,[2,4]];dims=2)./L)
            plot!(plt,Lift,Drag,label=:none,lw=0.2)
        end
    end
    # the same for 2x Biot-Savart
    jldopen("airfoil_wake_12L_8L_St0.6_$(biot).jld2") do file
        St=0.6; L=64;
        BC = "Biot-Savart"; ls = :solid
        forces = reduce(vcat,file["f"]')
        Lift = filtfilt(digitalfilter(responsetype, designmethod), sum(forces[4:end,[3,5]];dims=2)./L)
        Drag = filtfilt(digitalfilter(responsetype, designmethod), sum(forces[4:end,[2,4]];dims=2)./L)
        plot!(plt,Lift,Drag,label=:none,lw=0.2)
    end
    # get legend better
    plot!(plt,[-100,-100],[0.,0.],label="Airfoil St:$St Biot-Savart",color=plt.series_list[1].plotattributes[:linecolor])
    plot!(plt,[-100,-100],[0.,0.],label="Airfoil St:$St Reflection",color=plt.series_list[2].plotattributes[:linecolor])
    plot!(plt,[-100,-100],[0.,0.],label="Airfoil St:$St Biot-Savart 2x",color=plt.series_list[3].plotattributes[:linecolor])
    xlims!(-15,15);ylims!(-1.5,1.5);
    title!("Poincaré map");ylabel!("Drag coefficient");xlabel!("Lift coefficient")
    savefig("poincare_deflected_wake.png")
end