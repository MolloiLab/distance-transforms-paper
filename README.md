# distance-transforms-paper

## Structure
`analysis.jl`
- Notebook that loads all of the various .csv files from the `/data/*` folder and prepares figures and tables for analysis and publication

`benchmarks.ipynb`
- Google colab notebook that benchmarks our GPU accelerated python-wrapper distance transform against various Python-based distance transforms

`benchmarks.jl`
- Notebook that benchnmarks all of Julia distance transforms against ours (2D and 3D)
  - Maurer (ImageMorphology.jl)
  - Felzenswalb Single-Threaded CPU (DistanceTransforms.jl)
  - Felzenswalb Multi-Threaded CPU (DistanceTransforms.jl)
  - Proposed CUDA (DistanceTransforms.jl)
  - Proposed AMDGPU (DistanceTransforms.jl)
  - Proposed Metal (DistanceTransforms.jl)

`hausdorff_loss_timings.ipynb` [In Progress]
- Google colab notebook that benchmarks our DT in the hausdorff loss vs. other Python-based DTs vs. pure dice loss and benchmarks full training loops of the various hausdorff loss functions

`hausdorff_loss_accuracy.ipynb` [In Progress]
- Google colab notebook that provides full training loop of pure dice vs hausdorff + dice loss accuracy comparisons
- Outputs quantitative date (dice, hausdorff, iou, etc.) and qualitative images showing the different models

`skeletonization.jl` [TODO]
- Notebook that implements a GPU based skeletonization algorithm (using our GPU distance transform) and compares against a pure CPU skeletonization algorithm (using a CPU distance transform)

`/data`
- Contains the various timings and accuracy data as .csv files from the above notebooks

`/plots`
- Contains .png files created from the `analysis.jl` notebook for publications

