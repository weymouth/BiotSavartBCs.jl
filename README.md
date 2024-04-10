# BiotSavartBCs

[![Build Status](https://github.com/weymouth/BiotSavartBCs.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/weymouth/BiotSavartBCs.jl/actions/workflows/CI.yml?query=branch%3Amaster)

## Reprodcing the results

To reproduce the results presented in [Article title](https://www.sciencedirect.com/journal/journal-of-computational-physics) you will need a working `Julia` kernel, preferable `v.1.10.x`. You will also need a closed verion of the [`WaterLily.jl`](https://github.com/weymouth/WaterLily.jl) flow solver.

First, we will set up a `Workspace` where we can clone all the repository needed.
```bash
mkdir Workscape
cd Workspace
```
The you can clone the `BiotSavartBCs.jl` and the `WaterLily.jl` solver.
```bash
git clone https://github.com/weymouth/BiotSavartBCs.jl
git clone https://github.com/weymouth/WaterLily.jl
```

### Activate the `BiotSavartBCs` environnement

```bash
julia --project=~/Workspace/BiotSavartBCs
```
This will open julia
```julia
julia> ]
(BiotSavartBCs) pkg> instantiate
```