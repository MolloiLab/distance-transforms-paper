# distance-transforms-paper

## Structure
The `notebooks` folder holds all of the key notebooks for experimentation
- `notebooks/benchmarks.jl` contains all of the julia-based distance transforms comparisons
  - ImageMorphology.jl vs ours: single-threaded vs. multi-threaded vs. CUDA vs. AMDGPU vs. oneAPI vs. Metal
  - Save results to CSV files for analysis
  - [TODO]: add AMDGPU, oneAPI, and Metal to the benchmarks
- `notebooks/python/benchmarks.ipynb` contains all of the python-based distance transforms comparisons
  - Scipy vs. cupoch vs. ... vs. ours: single-threaded vs. multi-threaded vs. CUDA vs. AMDGPU
  - Save results to CSV files for analysis
  - [TODO]: all python based comparisons
- `notebooks/hd_loss_benchmarks.jl` contains the training code for the hybrid hausdorff loss + dice loss benchmarks compared to pure dice loss
  - Save results to CSV files for analysis
  - [TODO]: clean up code
- `notebooks/hd_loss.jl` contains the full training code for the hybrid hausdorff loss + dice loss compared to pure dice loss accuracy
  - Save results to CSV files for analysis
  - [TODO]: clean up code
- `notebooks/watershed.jl` contains the watershed distance transforms benchmarks
  - Compare ImageMorphology.jl vs ours: single-threaded vs. multi-threaded vs. CUDA vs. AMDGPU vs. oneAPI vs. Metal in the watershed algorithm
  - [TODO]: all of it
- `notebooks/analysis.jl` contains the benchmarks, accuracy metrics, and qualitative figures
  - [TODO]: update with new results


This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> hd-loss

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "hd-loss"
```
which auto-activate the project and enable local path handling from DrWatson.
