using WaterLily,StaticArrays,CUDA,BiotSavartBCs
using JLD2,Plots
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1) # good idea when accelerating from rest
function make_sim_acc(dims; N=128, R=32, a=0.5, U=1, Re=1e3, thk=2, mem=Array, use_biotsavart=false)
    disk2D(x,t) = (z=x-SA[-R,0].-N/2; Rt=R-thk; y=z.-SA[0,clamp(z[2],-Rt,Rt)]; √sum(abs2,y)-thk)
    disk3D(x,t) = (z=x-SA[-R,0,0].-N/2; Rt=R-thk; r=√(z[2]^2+z[3]^2); √(z[1]^2+(r-min(r,Rt))^2)-thk)
    Ut(i,t::T) where T = i==1 ? convert(T,min(a*t/R,U)) : zero(T) # velocity BC
    body = length(dims)==2 ? AutoBody(disk2D) : AutoBody(disk3D)
    use_biotsavart && return BiotSimulation(dims, Ut, R; U, ν=U*R/Re, body, mem)
    Simulation(dims, Ut, R; U, ν=U*R/Re, body, mem)
end
slice(a::AbstractArray{T,3}) where T = Array(dropdims(a,dims=3))
slice(a::AbstractArray{T,2}) where T = Array(a)

# size of the domain
N = 2^7; R = N/3
for Dim ∈ [2,3], use_biotsavart ∈ [true,false]
    sim = make_sim_acc(ntuple(i->N,Dim);N,R,mem=CUDA.CuArray,use_biotsavart);
    Rslice = Dim==2 ? inside(sim.flow.p) : CartesianIndices((2:N+1,2:N+1,N÷2+1:N÷2+1))
    forces = []; σ = []; p = [];
    for t in 1:6
        @time while sim_time(sim)<t
            sim_step!(sim;remeasure=false)
            f = 2WaterLily.pressure_force(sim)/R^(Dim-1)
            push!(forces,[sim_time(sim),f[1]])
        end
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        push!(p,slice(sim.flow.p[Rslice]))
        push!(σ,slice(sim.flow.σ[Rslice]))
    end
    jldsave("disk_$(Dim)D_$(N)D_$(use_biotsavart).jld2"; f=forces, σ=σ, p=p)
end
let
    # make the figures
    f_f = plot(dpi=600,xlabel="Convective time",ylabel="2F/ρU²RⁿCa")
    for Dim ∈ [2,3], use_biotsavart ∈ [true,false]
        BCs = use_biotsavart ? "biot" : "reflect"
        jldopen("disk_$(Dim)D_$(N)D_$(use_biotsavart).jld2") do file
            df = mapreduce(permutedims, vcat, file["f"])
            t = df[:,1]; idx = t .< 3
            t = t[idx]; f = df[idx,2]
            Ca=Dim==2 ? π : 8/3
            ls = ifelse(Dim==2,:solid,:dash)
            c = ifelse(use_biotsavart,1,2)
            @show -f[1]/Ca
            plot!(f_f,t,-f./Ca,label="$(Dim)D "*(use_biotsavart ? "Biot-Savart" : "Reflection");lw=2,ls,c)
            for t ∈ 1:6
                p = file["p"][t]; σ = file["σ"][t]
                flood(p,clims=(-2,2),cfill=:viridis)
                savefig("Disk_$(Dim)D_"*BCs*"_press_$(t).png")
                flood(σ,clims=(-20,20))
                savefig("Disk_$(Dim)D_"*BCs*"_omega_$(t).png")
                flood(σ,clims=(-20,20))
                contour!(clamp.(p',-2,2),levels=range(-2,2,length=10),color=:black,
                        linewidth=0.5,legend=false)
                savefig("Disk_$(Dim)D_"*BCs*"_press_omega_$(t).png")
            end
        end
    end
    jldopen("disk_high_re_forces.jld2") do file
        df = mapreduce(permutedims, vcat, file["f"])
        t = df[:,1]; idx = t .< 3
        t = t[idx]; f = df[idx,2]
        Ca= 8/3*1.51
        plot!(f_f,t,f./Ca,label="3D High Re";lw=2,ls=:dot)
    end
    xlims!(f_f,0,3); ylims!(f_f,0,10)
    # savefig(f_f,"Disk_force_comparison.png")
end