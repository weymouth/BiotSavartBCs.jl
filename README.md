# BiotSavartBCs

[![Build Status](https://github.com/weymouth/BiotSavartBCs.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/weymouth/BiotSavartBCs.jl/actions/workflows/CI.yml?query=branch%3Amaster)

![disk](tex/fig/disk_high_re_7.png)

This repository defines an extension to the [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl) flow solver, adding "external flow" boundary conditions based on the [Biot-Savart equation](https://en.wikipedia.org/wiki/Biot%E2%80%93Savart_law#Aerodynamics_applications). This equation is used to update the velocity *on the boundaries* of the simulation domain based on the vorticity *within* the domain. The resulting boundary conditions are an excellent model for external flow, allowing you to use very small domains around any immersed bodies. [See the paper for the detailed methodology and validation.](https://physics.paperswithcode.com/paper/using-biot-savart-to-shrink-eulerian-domains)

### WaterLily.jl Simulations with BiotSavartBCs.jl

Using these new boundary conditions within a `WaterLily` simulation is really straightforward; this requires changing only two (😱) lines of code. The first one is obviously

```julia
using WaterLily,BiotSavartBCs
```
The second line that you have to modify creates the Biot-Savart Simulation structure
```julia
sim = BiotSimulation((4L,2L,2L), Ut, L; body=AutoBody(sdf,map), ν=U*L/Re, T, mem=CUDA.CuArray)
```
You can update and plot the simulation structure exactly the same at with a standard WaterLily `Simulation`
```julia
sim_step!(sim, t_end; remeasure::Bool)
```
Note that this function is type-specialized to use a new `biot_mom_step!()` instead of the classical `mom_step!()`.

There are numerous examples in the `examples` folder of this repository that show how to use these new boundary conditions in practice.

### Method

This package takes a practical approach to avoid the two fundamental issues with applying the Biot-Savart equation to set the boundary conditions of a projection-based Navier-Stokes solver: 
 1. A naive weighted sum over the $N_s$ vorticity sources at every cell for all $N_t$ targets at the domain cell faces would make the boundary condition update take $O(N_s N_t)$ operations, making it *orders of magnitude slower* than the rest of the solver. We accelerate the BC update by clustering the vorticity sources using a tree method (oct-tree in 3D and quad-tree in 2D). This reuses the pooling method in WaterLily's Multigrid pressure solver and reduces the cost to $O(\log(N_s) N_t)$. We can further accelerate the BC update by also clustering the target faces - making this an $O(N_t)$ Fast Multi*grid* Method - a variant of the classic [Fast Multipole Method](https://en.wikipedia.org/wiki/Fast_multipole_method). Finally, we parallelize over all the targets using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) which works on the GPU or multi-threaded CPU.
 2. The pressure projection step depends sensitively on the boundaries conditions, but these *cannot be set* since the unknown pressure generates vorticity on immersed bodies. We solve this problem using a matrix partition method, similar to the approach used for partitioned Fluid-Structure-Interaction (FSI) methods. In practise we see the Multigrid pressure solver actually converges *faster* with `BiotSavartBcs` than with reflection BCs.

The resulting simulation update is very fast, especially with large 3D grids on the GPU - exactly where the ability to use a snug domain is the most important. See the paper for detailed methods, examples, and computational benchmarks. 

### Limitations

Currently, a `BiotSimulation` applies the BiotSavartBC to **all** the domain boundaries - ignoring any request for symmetry or periodic boundary conditions. It should be fairly straightforward to implement the interaction of these boundary conditions and we would be happy to help with cumminity development in this direction.

### Gallery

Here are a few renderings of the cool things you can do with [`WaterLily.jl`](https://github.com/weymouth/WaterLily.jl) and these new Biot-Savart BCs

#### Flow behind a square plate at Re=125,000
[![square1](https://img.youtube.com/vi/CNQqI5rRdug/0.jpg)](https://www.youtube.com/shorts/CNQqI5rRdug)

[![square2](https://img.youtube.com/vi/tbf06uhnAEQ/0.jpg)](https://www.youtube.com/shorts/tbf06uhnAEQ)

#### Flow behind a Doritos at Re=25,000
[![doritos](https://img.youtube.com/vi/spFlx2YW0pg/0.jpg)](https://www.youtube.com/shorts/spFlx2YW0pg)

## Reproducing the results

To reproduce the results presented [here](https://arxiv.org/abs/2404.09034), you will need a working `Julia` kernel, preferably `v.1.10.x`. You will also need a version of the [`WaterLily.jl`](https://github.com/weymouth/WaterLily.jl) flow solver.

The first step before using these new boundary conditions is to clone this repository somewhere on your machine; I will assume that this repository is to be cloned into a `Workspace` folder on your machine. The first thing to do is to go into this directory
```bash
cd ~/Workspace
```
Then, you can clone the `BiotSavartBCs.jl` (and the `WaterLily.jl` solver if you don't already have it, note that you cannot use the official `WaterLily` release from the `Julia` package manager to run these test due to some function incompatibility)
```bash
git clone https://github.com/weymouth/BiotSavartBCs.jl
(git clone https://github.com/weymouth/WaterLily.jl)
```
The next step is to `dev` the `BiotSavartBCs.jl` into the `julia` environment you are using to run the `WaterLily` simulations.

### Activate the `BiotSavartBCs` environment

I will assume you are using `BiotSavartBCs.jl` with `WaterLily` and that you have an `examples` folder on your machine which contains a `Project.toml` file that sets the environment used to run the simulations. 

Start by opening `julia` and activate this environment
```bash
julia --project=/PATH/TO/BiotSavartBCs.jl/examples
```
This will open julia; use the package manager to `dev` the `BiotSavartBCs.jl`
```julia
julia> ]
(examples) pkg> dev /PATH/TO/BiotSavartBCs.jl
...
(examples) pkg> instantiate
...
```
This will install and precompile some of the packages required to use this new package. If you have elected to use the github version of WaterLily, you can also `dev` it in the same way
```julia
(examples) pkg> dev /PATH/TO/WaterLily.jl
...
(examples) pkg> instantiate
...
```
You can then import the `BiotSavartBCs.jl` package and use it within a `WaterLily` simulation.
```julia
using BiotSavartBCs
```

### Citing

We simply ask you to cite the references below in any publication in which you have made use of the `BiotSavartBCs` project. If you are using other `WaterLily` packages, please cite them as indicated in their repositories.

```bibtex
@article{weymouth2024biot,
    title={Using Biot-Savart to shrink Eulerian domains while maintaining or improving external flow accuracy}, 
    author={Gabriel D. Weymouth and Marin Lauber},
    year={2024},
    eprint={2404.09034},
    archivePrefix={arXiv},
    primaryClass={physics.flu-dyn}
}
```
