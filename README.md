# BiotSavartBCs

[![Build Status](https://github.com/weymouth/BiotSavartBCs.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/weymouth/BiotSavartBCs.jl/actions/workflows/CI.yml?query=branch%3Amaster)

![disk](tex/fig/disk_high_re_7.png)

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

### WaterLily.jl Simulations with BiotSavartBCs.jl

Using these new boundary conditions within a `WaterLily` simulation is really straightforward; this requires changing only two (😱) lines of code. The first one is obviously

```julia
using WaterLily,BiotSavartBCs
```
The second line that you have to modify creates the Biot-Savart Simulation structure (which internally takes care of allocating the `tree` or the `fmm` data srtuctures)

```julia
sim = BiotSimulation((4L,2L,2L), Ut, L; body=AutoBody(sdf,map), ν=U*L/Re, T, mem=CUDA.CuArray)
```

all the methods for flow update, such as `sim_step!(sim, args)` are type-specialized to use `biot_mom_step!()` instead of the classical `mom_step!()`, so the user should really only call `sim_step!(sim; remeasure::Bool)`.

There are numerous examples in the `examples` folder of this repository that show how to use these new boundary conditions in practice.

### Gallery

Here are a few renderings of the cool things you can do with [`WaterLily.jl`](https://github.com/weymouth/WaterLily.jl) and these new Biot-Savart BCs

#### Flow behind a square plate at Re=125,000
[![square1](https://img.youtube.com/vi/CNQqI5rRdug/0.jpg)](https://www.youtube.com/shorts/CNQqI5rRdug)

[![square2](https://img.youtube.com/vi/tbf06uhnAEQ/0.jpg)](https://www.youtube.com/shorts/tbf06uhnAEQ)

#### Flow behind a Doritos at Re=25,000
[![doritos](https://img.youtube.com/vi/spFlx2YW0pg/0.jpg)](https://www.youtube.com/shorts/spFlx2YW0pg)


### Limitations

This implementation is limited to flow __without__ symmetries accounted for via reflections, you need to model the full 2D or 3D problem and cannot combine it with symmetric boundary conditions out of the box (you are of course free to implement this yourself).

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
