### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 3c5fb9b4-d89c-11ed-1f0f-676a16540e9a
md"""
# Timings
## Distance Transforms (2D & 3D)
Compare the pure distance transform algorithms in 2D & 3D. The scipy version is a Julia wrapper around the scipy DT. Compare that against the Julia Felzenszwalb DT on CPU and GPU.
- ✅ ``\text{Scipy}``
- ✅ ``\text{Felzenszwalb}_{\text{CPU}}``
- ✅ ``\text{Felzenszwalb}_{\text{GPU}}``

## Hausdorff Loss (2D & 3D)
Compare the pure Hausdorff loss functions in 2D & 3D on random arrays. The Hausdorff loss function itself will remain identical between all implementations, with only the underlying distance transform algorithm changing. The current gold standard scipy DT version is the reference. Next, show Felzenszwalb DT on CPU in the HD loss and finally Felzenszwalb on GPU in the HD loss, which should be the fastest.
- ✅ Pure HD Loss, Scipy (``\text{HD-Scipy}``)
- ✅ Pure HD Loss, Felzenszwalb on CPU (``\text{HD-Felzenszwalb}_{\text{CPU}}``)
- ✅ Pure HD Loss, Felzenszwalb on GPU (``\text{HD-Felzenszwalb}_{\text{GPU}}``)

## Training Loops (Task02 Heart, 3D Only)
Compare the step and epoch times for various training loops. The baseline will be a pure dice loss. Then compare hybrid forms of the Dice/HD loss functions. Use the scipy implementation since that is the current gold standard. Then use the Felzenszwalb GPU version only, since that is the fastest which is shown in the previous sections above.
- 💬 Pure Dice (``\text{Loss}_{\text{Dice}}``)
- 💬 Hybrid Dice/HD (``\text{Loss}_{\text{Scipy}}``)
- 💬 Hybrid Dice/HD (``\text{Loss}_{\text{Felzenszwalb}}``)

# Accuracy
This section looks at 2 different datasets, and since timings are already acquired, we will only use the fastest hybrid loss function (``\text{Loss}_{\text{Felzenszwalb}}``) on GPU, to compare against the baseline pure dice loss function. Record the Hausdorff Distance and Dice-Sorenson Coefficient for the best model for both loss functions.

## Task02 Heart
- 💬 Pure Dice Metrics (``\text{Loss}_{\text{Dice}}``)
- 💬 Hybrid Dice/HD Metrics (``\text{Loss}_{\text{Felzenszwalb}}``)

## RSNA Challenge
- 💬 Pure Dice Metrics (``\text{Loss}_{\text{Dice}}``)
- 💬 Hybrid Dice/HD Metrics (``\text{Loss}_{\text{Felzenszwalb}}``)

"""

# ╔═╡ Cell order:
# ╠═3c5fb9b4-d89c-11ed-1f0f-676a16540e9a
