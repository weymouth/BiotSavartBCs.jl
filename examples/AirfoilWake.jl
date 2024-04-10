using BiotSavartBCs
using WaterLily
using StaticArrays
using CUDA
using JLD2
include("TwoD_plots.jl")
include("Diagnostics.jl")

function ellipse(D,n,m,Λ=5.0;A₀=1.0,St=0.6,U=1,Re=100,T=Float32,mem=Array)
    h₀=T(A₀*D/2); ω=T(2π*St*U/D)
    function sdf(x,t)
        √sum(abs2,SA[x[1]/Λ,x[2]])-D÷2/Λ
    end
    function map(x,t)
        x .- SA[n*D÷4,m*D÷2-h₀*sin(ω*t)]
    end
    Simulation((n*D,m*D), (U,0), D; body=AutoBody(sdf,map), ν=U*D/Re, T, mem)
end
# do we use Biot-Savart?
St=0.5
for use_biotsavart in [true]
    (n,m) = use_biotsavart ? (6,4).*2 : (30,20)
    sim = use_biotsavart ? ellipse(64,n,m,5;St,mem=CUDA.CuArray) : ellipse(64,n,m,5;St,mem=CUDA.CuArray); 
    ω = use_biotsavart ? MLArray(sim.flow.σ) : nothing
    t₀,duration,tstep = round(sim_time(sim)), 100., 0.1
    R = use_biotsavart ? inside(sim.flow.p) : CartesianIndices((386:769, 514:769)) # show the same part
    # R = use_biotsavart ? CartesianIndices((98:481,130:385)) : CartesianIndices((290:1057,386:897)) # show the same part
    forces = []
    anim = @animate for tᵢ in range(t₀,t₀+duration,step=tstep)
        while sim_time(sim) < tᵢ
            measure!(sim)
            use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : mom_step!(sim.flow,sim.pois)
            pres,visc = diagnostics(sim);
            push!(forces,[sim_time(sim),pres...,visc...])
        end
        println("tU/L=",round(sim_time(sim),digits=4),
        ", Δt=",round(sim.flow.Δt[end],digits=3))
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ[R],clims=(-10,10))
        # if use_biotsavart==true
        #     dummy = similar(sim.flow.p,(12sim.L+2,8sim.L+2))
        #     dummy .= 0.0
        #     dummy[R] .= sim.flow.σ[inside(sim.flow.σ)]
        #     flood(dummy,clims=(-10,10))
        #     plot!([R[1][1],R[1,end][1],R[end,1][1],R[end][1],R[1][1]],
        #           [R[1][2],R[1,end][2],R[end][2],R[end,1][2],R[1][2]],
        #           color=:black,lw=0.5,alpha=0.5,ls=:dash,legend=:none)
        # else
        #     flood(sim.flow.σ[R],clims=(-10,10))
        # end
        # xlims!(-1,12sim.L+2); ylims!(-1,8sim.L+2)
    end
    gif(anim, "deflected_wake_$(n)L_$(m)L_St$(St)_$(use_biotsavart)_2x.gif")
    jldopen("deflected_wake_St$(St)_$(use_biotsavart)_2x.jld2","w") do file
        file["force"] = forces
    end
end
using Plots
mean_CL = []; mean_CL2=[]
for St in [0.2,0.3,0.4,0.5,0.6]
    @show St
    plt = plot(dpi=300)
    for use_biotsavart ∈ [true false]
        jldopen("deflected_wake_St$(St)_$(use_biotsavart).jld2","r") do file
            BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
            ls = ifelse(use_biotsavart,:solid,:dash)
            forces = reduce(vcat,file["force"]'); L=64
            plot!(plt,forces[:,1]*St,sum(forces[:,[3,5]];dims=2)./L,label="Airfoil St:$St "*BC)
            L_mean = sum(forces[end-20000:end,[3,5]])./20000L
            push!(mean_CL,L_mean)
            plot!([0,maximum(forces[:,1]*St)],[L_mean,L_mean],color=:black,label="Mean force "*BC;ls)
        end
    end
    if St in [0.5,0.6]
        jldopen("deflected_wake_St$(St)_true_2x.jld2","r") do file
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

# mean lift ocnvergence
St = [0.2,0.3,0.4,0.5,0.6]
plt = plot(dpi=300)
plot!(plt,St,mean_CL[2:2:end],label="Cₗ reflection";marker=:x,ls=:dash)
plot!(plt,St,mean_CL[1:2:end],label="Cₗ Biot-Savart";marker=:o,ls=:solid)
plot!(plt,[0.5,0.6],mean_CL2,label="Cₗ Biot-Savart 2x";marker=:s,ls=:solid)
xlims!(0.,0.8); ylims!(-0.5,2); plot!(legend=:topleft)
xlabel!("Strouhal number"); ylabel!("Mean Lift coefficient")
savefig("CL_mean_deflected_wake.png")

# Poincaré map
using DSP
responsetype = Lowpass(0.01)
designmethod = Butterworth(4)
plt = plot(dpi=300)
for use_biotsavart ∈ [true false]
    jldopen("deflected_wake_St0.6_$use_biotsavart.jld2","r") do file
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        forces = reduce(vcat,file["force"]'); St=0.6; L=64
        Lift = filtfilt(digitalfilter(responsetype, designmethod), sum(forces[4:end,[3,5]];dims=2)./L)
        Drag = filtfilt(digitalfilter(responsetype, designmethod), sum(forces[4:end,[2,4]];dims=2)./L)
        plot!(plt,Lift,Drag,label=:none,lw=0.2)
    end
end
jldopen("deflected_wake_St0.6_true_2x.jld2","r") do file
    BC = "Biot-Savart"; ls = :solid
    forces = reduce(vcat,file["force"]'); St=0.6; L=64
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
