# Standalone test: spanwise-periodic z-reflection symmetry.
#
# For a z-uniform body, z-invariant inflow, and w(t=0)=0, the spanwise velocity w is odd
# under z -> -z and must stay ~machine zero for all time. Plain WaterLily preserves this
# (max|w| ~ 1e-4, round-off). BiotSimulation must too: the periodic-aware FMM (pinteraction,
# extended-domain clipping + wrapped ω at every level) keeps the induced wall velocity z-uniform,
# so max|w| matches the WaterLily baseline. A regression here (e.g. reintroducing finite-domain
# clipping in the periodic directions) shows up as max|w| ~ 5e-3 (~50x baseline).
#
# Run standalone:  julia --project=. test/spanwise_w_symmetry.jl
using WaterLily, BiotSavartBCs, Test

cyl_body(m,D) = AutoBody((x,t)->√sum(abs2,(x.-m/2)[1:2])-D/2)
maxw(sim,nstep) = (for _ in 1:nstep; sim_step!(sim; remeasure=false); end;
                   maximum(abs, sim.flow.u[:,:,:,3]))

@testset "spanwise z-reflection symmetry (w stays ~0)" begin
    D=24; m=2D; Lz=8; nstep=6
    w_wl   = maxw(Simulation((m,m,Lz),(1,0,0),D; body=cyl_body(m,D), ν=D/1e3, perdir=(3,)), nstep)
    w_biot = maxw(BiotSimulation((m,m,Lz),(1,0,0),D; body=cyl_body(m,D), ν=D/1e3,
                                 fmm=true, perdir=(3,), nimages=4), nstep)
    @info "max|w|" waterlily=w_wl biot=w_biot ratio=w_biot/w_wl

    @test w_wl < 1e-3                       # WaterLily baseline: z-symmetry preserved
    @test w_biot < 5*w_wl                   # Biot-Savart must not exceed the baseline (now ~1x)
end
