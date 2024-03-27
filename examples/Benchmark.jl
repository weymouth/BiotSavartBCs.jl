using WaterLily
using BiotSavartBCs
using BenchmarkTools
using CUDA: CuArray, allowscalar
using KernelAbstractions: synchronize, get_backend
using StaticArrays

allowscalar(false)
getf(str) = eval(Symbol(str))
backend_str = Dict(Array => "CPUx$(Threads.nthreads())", CuArray => "GPU")

# the functions we want to benchamrk
circ(;D=64,n=5,m=3,Re=200,U=1,mem=CuArray) = Simulation((n*D,m*D), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x.-m*D÷2)-D÷2),ν=U*D/Re,mem)
function reflect_sim_step!(sim::Simulation,ω,t_end;max_steps=typemax(Int))
    while sim_time(sim) < t_end && length(sim.flow.Δt) <= max_steps
        mom_step!(sim.flow,sim.pois)
    end
end
function biot_sim_step!(sim::Simulation,ω,t_end;max_steps=typemax(Int))
    while sim_time(sim) < t_end && length(sim.flow.Δt) <= max_steps
        biot_mom_step!(sim.flow,sim.pois,ω)
    end
end

macro add_benchmark(args...)
    ex, b, suite, label = args
    return quote
        $suite[$label] = @benchmarkable begin
            $ex
            synchronize($b)
        end
    end |> esc
end

function add_to_suite!(suite, case, domains; s=100, D=64, ft=Float32, backend=Array)
    bstr = backend_str[backend]
    suite[bstr] = BenchmarkGroup([bstr])
    for (n,m) in domains
        sim = circ(D,n,m); ω = MLArray(sim.flow.σ);
        suite[bstr][repr(n)*"x"*repr(m)] = BenchmarkGroup([repr(n)*"x"*repr(m)])
        @add_benchmark $getf($case)($sim, $ω, $typemax($ft); max_steps=$s) $(get_backend(sim.flow.p)) suite[bstr][repr(n)*"x"*repr(m)] case
    end
end

# typical benchamrk
cases=["reflect_sim_step!", "biot_sim_step!"]
max_steps = [1000,1000]
ftype = [Float32,Float32]
backend = CuArray
domains = [[(5,3) (8,5) (10,8) (20,16) (30,24)],[(5,3) (8,5) (10,8)]]

# Generate benchmarks
function benchmark()
    for (case, s, ft, domain) in zip(cases, max_steps, ftype, domains)
        println("Benchmarking: $(case)")
        suite = BenchmarkGroup()
        results = BenchmarkGroup([case, "mom_step!", backend_str[backend], string(VERSION)])
        # generte the stuff we need
        sim = circ(); ω = MLArray(sim.flow.σ);
        getf(case)(sim, ω, typemax(ft); max_steps=1) # warm up
        add_to_suite!(suite, case, domain; s=s, ft=ft, backend=backend) # create benchmark
        results[backend_str[backend]] = run(suite[backend_str[backend]], samples=1, evals=1, seconds=1e6, verbose=true) # run!
        fname = "$(case)_$(n)x$(m)_$(s)_$(ft)_$(backend_str[backend])_$VERSION.json"
        BenchmarkTools.save(fname, results)
    end
end

benchmark()