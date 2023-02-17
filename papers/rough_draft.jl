### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 706bcf20-ad6a-11ed-1997-ebe169c89f20
# ╠═╡ show_logs = false
using DrWatson; @quickactivate "hd-loss"

# ╔═╡ f7bf8e5b-d4f7-42ae-b911-875f54df49ce
using PlutoUI, Images, CSV, DataFrames

# ╔═╡ 8a852101-15e1-4d00-ae3b-61e94f6236ac
TableOfContents()

# ╔═╡ 0caacaf2-0dd5-4e19-9a60-8fd87953bc15
md"""
# Optimized Hausdorff Loss Function Based on a Novel Distance Transform Operation
"""

# ╔═╡ 06cccce0-3f96-4a92-a7f7-4d16495d94f0
md"""
# Abstract
"""

# ╔═╡ 0d12bc5b-53f6-4c1a-99b6-0dc8ad0af5b2
md"""
# I. Introduction
"""

# ╔═╡ 7596996f-140c-4ffa-9c73-a8ec7746ca9a
md"""
Delineating a region of interest in an image is known as image segmentation. The need to separate the region of interest for additional analysis is a key task in medical image analysis.

Numerous studies have addressed the topic of medical image segmentation [CITE]. Manual segmentations, created by an expert radiologist, are typically regarded as the gold standard. However, the speed and reproducibility of the results can be improved by semi-automatic and fully-automatic segmentation techniques. Fully-automated segmentation techniques remove the inter and intra-observer variability. Furthermore, significant progress has been made in closing the performance gap between automatic and manual segmentation approaches, particularly with the recent emergence of segmentation algorithms based on convolutional neural networks (CNN) [CITE].

Segmentation may be one step in a more involved procedure in some applications. For instance, several multimodal medical image registration techniques rely on segmenting a target organ in one or more pictures [CITE]. The Dice similarity coefficient (DSC), overlap, and Hausdorff Distance (HD) are some popular metrics that are typically computed to assess the segmentation method relative to some ground truth segmentation [CITE]. HD is one of the most useful metrics for quantifying boundary information. As measured by HD, the highest segmentation error can be a strong indicator of how effective the segmentations are. For two sets of points ``(X, Y)``, the HD is defined by [CITE]. Eq. 1 shows the one-sided HD from ``X`` to ``Y``, Eq. 2 shows the one-sided HD from ``Y`` to ``X``, and Eq. 3 shows the bi-directional HD between both sets.

```math
\begin{aligned}
\text{hd}(X, Y) = \text{max}_{x \varepsilon X} \ [\text{min}_{y \varepsilon Y} (||x-y||_{2})]
\end{aligned}
\tag{1}
```

```math
\begin{aligned}
\text{hd}(Y, X) = \text{max}_{y \varepsilon Y} \ [\text{min}_{x \varepsilon X} (||x-y||_{2})]
\end{aligned}
\tag{2}
```

```math
\begin{aligned}
\text{HD}(X, Y) = \text{max}[\text{hd}(X, Y), \text{hd}(Y, X)]
\end{aligned}
\tag{3}
```

Recently, segmentation approaches, specifically deep learning-based methods, have focused on directly minimizing the HD during the training process via a novel loss function, termed the Hausdorff loss (HD loss) [CITE]. The most accurate form of the HD loss utilizes a distance transform operation [CITE] to estimate the HD.

Although the HD loss function improves the segmentation error, especially along the boundaries, this loss function is unstable, particularly in the early part of the training process [CITE]. A weighted combination of loss functions can solve this issue, like the Dice loss and HD loss [CITE]. 

Still, the HD loss is more computationally expensive than the more common Dice loss, which limits the usage of HD loss in deep learning. Previous HD loss implementations utilized central processing unit (CPU)-based distance transform operations, as those methods are readily available in packages like SciPy [CITE]. The downside to this approach is that all practical deep learning training takes place on the graphics processing unit (GPU), which means the training must switch back and forth from CPU to GPU every time the HD loss is computed. Also, for large arrays, like those commonly found in medical imaging, GPU operations are faster than CPU operations.

This study presents a new distance transform operation for the HD loss function. This novel HD loss function improves the speed of operation compared to the previous implementations and provides a GPU-optimized algorithm for deep-learning-based medical image segmentation. All the code is publicly available for use at https://github.com/Dale-Black/DistanceTransforms.jl
"""

# ╔═╡ 0039da08-5d0b-4dfe-be30-bdec06f526e3
md"""
# II. Methods
"""

# ╔═╡ 38c0828f-72a0-44b2-b2e8-9d8c241b1fee
md"""
## 2.1 - Hausdorff Loss Function
"""

# ╔═╡ 5ff24d6d-d385-4efc-807c-c45f0f4a419c
md"""
The HD loss function was previously described by [CITE] and can be seen in Eq. 4 

```math
\begin{aligned}
\text{Loss}_{\text{HD}}(q, p) = \frac{1}{|\Omega|} \sum_{\Omega}((p - q)^{2} \circ (d_{p}^{\alpha} + d_{q}^{\alpha}))
\end{aligned}
\tag{4}
```

where ``p`` is the binary ground-truth segmentation, ``q`` is the binary prediction, ``d_{p}`` is the distance transform of ``p``, ``d_{q}`` is the distance transform of ``q``, ``\Omega`` represents the set of all pixels, and ``\circ`` represents the Hadamard product (i.e., the element-wise matrix product). The parameter ``\alpha`` determines how strongly the larger errors are penalized. Previous reports show that values of ``\alpha`` between 1 and 3 led to good results.

HD loss is a good estimator of HD, and _ et al. show that combining HD loss with the Dice loss function (Eq. 5) helps avoid the instability issue associated with HD loss [CITE]. Thus, the resulting loss function used throughout this study is a combination of ``\text{Loss}_{\text{HD}}`` and ``\text{Loss}_{\text{Dice}}``, shown in Eq. 6

```math
\begin{aligned}
\text{Loss}_{\text{Dice}}(q, p) = 1 - \dfrac{2\sum_{\Omega}(p \circ q)}{\sum_{\Omega}(p^2 +q^2)}
\end{aligned}
\tag{5}
```

```math
\begin{aligned}
\text{Loss}(q, p) = \lambda \Bigg(\dfrac{1}{\lvert\Omega\rvert}\sum_{\Omega} \bigg((p-q)^2\circ (d_p + d_q)\bigg)\Bigg)+ (1-\lambda) \Bigg(1 - \dfrac{2\sum_{\Omega}(p \circ q)}{\sum_{\Omega}(p^2 +q^2)}\Bigg)
\end{aligned}
\tag{6}
```
"""

# ╔═╡ 2c835912-8e15-44da-a512-775730bb5281
md"""
## 2.2 - Distance Transform
"""

# ╔═╡ 585cb69a-98d0-4fe6-8428-4904a4d41dcc
md"""
This study focused on improving CPU and GPU distance transform operations to decrease the time associated with previously proposed HD loss functions. The Julia programming language [CITE] was a logical choice since CPU and GPU code can be written in the same high-level language with speed similar to more common languages like C and Fortran.

The CPU form of the distance transform is a dynamic programming approach adapted from Felzensqalbs et al. [CITE]. Instead of transforming the entire image at once, the image is broken down into multiple one-dimensional transform operations, making this approach effective for arbitrary dimensions. 

The first one-dimensional distance transform operation is unique since the image contains either ``0`` or ``\infty`` corresponding to the background and foreground. This first transform operation is shown in Algorithm 1.

```
Left block:
∞, ..., ∞, 0, ...				=>	 	…, 9, 4, 1, 0, ...
Right block: 
..., 0, ∞, ..., ∞				=>		…, 0, 1, 4, 9, ...
Middle block
..., 0, ∞, ..., ∞, 0, ... 		=> 		…, 0, 1, 4, 9, ..., 9, 4, 1, 0, ...
```
"""

# ╔═╡ bec3fc5a-8b64-46aa-b5c0-500b94d03a8a
md"""
## 2.3 - Timings
"""

# ╔═╡ a019743d-b74e-42f8-a73a-192b87cc4c64
md"""
This study focused on improving the distance transform operation to improve the speed of the HD loss function. To measure this, the current gold standard (CPU) distance transform operation, written in Python (``\text{DT}_{\text{PY\_CPU}}``), was benchmarked against two new distance transform operations written in Julia on the CPU (``\text{DT}_{\text{JL\_CPU}}``) and GPU (``\text{DT}_{\text{JL\_GPU}}``). Various sizes of two and three-dimensional arrays were used as inputs to these three different distance transform operations to determine the fastest distance transform in a size-dependent way.

The current gold standard HD loss (``\text{HD}_{\text{PY\_CPU}}``), which utilizes the Python distance transform (``\text{DT}_{\text{PY\_CPU}}``) was compared to the new HD loss functions, presented in this paper, that utilize the Julia-based distance transforms (``\text{HD}_{\text{JL\_CPU}}`` and ``\text{HD}_{\text{JL\_CPU}}``). These timings were similarly computed on a range of different-sized arrays in two and three-dimensions.

Finally, a medical imaging segmentation task from the segmentation decathlon [CITE] was used to time the difference in speed between the previous HD loss compared to the new HD loss functions presented in this paper. To eliminate uncertainty inherent to benchmarking Python and Julia deep-learning code, the gold standard HD loss was re-written in Julia so that each step and epoch timing would contain the same data and code, with the loss function as the only variable. 
"""

# ╔═╡ 68bc23e7-774d-4378-a9fb-07167ed44411
md"""
## 2.4 - Training
"""

# ╔═╡ 9fe5b400-aaf5-4695-8dc1-8d8d69d06b54
md"""
### 2.4.1 - Model
"""

# ╔═╡ 76b1278d-e5f1-43c3-8f93-3277788301b8
md"""
### 2.4.2 - Loop
"""

# ╔═╡ 28a82a1d-b726-4e58-b164-7b5135273fbd
md"""
### 2.4.3 - Timings
"""

# ╔═╡ 78c8eb22-b8e5-4b3c-bf61-fbac9be5ba7a
md"""
## 2.5 - Hardware
"""

# ╔═╡ fb417691-9c1a-46ec-aa2c-3c9c325c1d73
md"""
# III. Results
"""

# ╔═╡ 9c02f72e-ee10-4a58-a449-16461a475809
md"""
## 3.1 - Distance Transforms Benchmarks
"""

# ╔═╡ d5fd0b05-54fa-43b8-9f35-eea3fd260176
load(plotsdir("dt.png"))

# ╔═╡ 4449f6a2-7727-4bd5-a1b6-394ec1864b81
md"""
## 3.2 - Loss Function Benchmarks
"""

# ╔═╡ ec837694-cfa1-4abe-8de3-9efcf4b46004
load(plotsdir("loss.png"))

# ╔═╡ d899cdf6-4b94-496a-a058-73e394a7ea6a
md"""
## 3.3 - Training Loop Benchmarks
"""

# ╔═╡ c2db5713-3167-46f6-8be0-80d3efc8223d
load(plotsdir("training_step.png"))

# ╔═╡ 2ba9e25b-d46e-4b1e-8455-e30e1ff8c155
load(plotsdir("training_epoch.png"))

# ╔═╡ e548262f-15fc-4446-afa0-e684a0526ad0
md"""
## 3.4 - Accuracy
"""

# ╔═╡ 4d8340f7-e4ca-4403-bb71-66e245036183
load(plotsdir("dice_hd_julia.png"))

# ╔═╡ 43c081f1-14c6-4f72-b112-8fa192d862f1
CSV.read(datadir("analysis", "accuracy.csv"), DataFrame)

# ╔═╡ 8b77b013-f70f-4470-b1bb-5b538edee5e9
md"""
# IV. Discussion
"""

# ╔═╡ afe38064-9194-447c-8f88-23980727bab4
md"""
# V. Conclusion
"""

# ╔═╡ 012af9d3-9095-4420-bffa-b7096665b153
md"""
# References
"""

# ╔═╡ Cell order:
# ╠═706bcf20-ad6a-11ed-1997-ebe169c89f20
# ╠═f7bf8e5b-d4f7-42ae-b911-875f54df49ce
# ╠═8a852101-15e1-4d00-ae3b-61e94f6236ac
# ╟─0caacaf2-0dd5-4e19-9a60-8fd87953bc15
# ╟─06cccce0-3f96-4a92-a7f7-4d16495d94f0
# ╟─0d12bc5b-53f6-4c1a-99b6-0dc8ad0af5b2
# ╟─7596996f-140c-4ffa-9c73-a8ec7746ca9a
# ╟─0039da08-5d0b-4dfe-be30-bdec06f526e3
# ╟─38c0828f-72a0-44b2-b2e8-9d8c241b1fee
# ╟─5ff24d6d-d385-4efc-807c-c45f0f4a419c
# ╟─2c835912-8e15-44da-a512-775730bb5281
# ╟─585cb69a-98d0-4fe6-8428-4904a4d41dcc
# ╟─bec3fc5a-8b64-46aa-b5c0-500b94d03a8a
# ╟─a019743d-b74e-42f8-a73a-192b87cc4c64
# ╟─68bc23e7-774d-4378-a9fb-07167ed44411
# ╟─9fe5b400-aaf5-4695-8dc1-8d8d69d06b54
# ╟─76b1278d-e5f1-43c3-8f93-3277788301b8
# ╟─28a82a1d-b726-4e58-b164-7b5135273fbd
# ╟─78c8eb22-b8e5-4b3c-bf61-fbac9be5ba7a
# ╟─fb417691-9c1a-46ec-aa2c-3c9c325c1d73
# ╟─9c02f72e-ee10-4a58-a449-16461a475809
# ╟─d5fd0b05-54fa-43b8-9f35-eea3fd260176
# ╟─4449f6a2-7727-4bd5-a1b6-394ec1864b81
# ╟─ec837694-cfa1-4abe-8de3-9efcf4b46004
# ╟─d899cdf6-4b94-496a-a058-73e394a7ea6a
# ╟─c2db5713-3167-46f6-8be0-80d3efc8223d
# ╟─2ba9e25b-d46e-4b1e-8455-e30e1ff8c155
# ╟─e548262f-15fc-4446-afa0-e684a0526ad0
# ╟─4d8340f7-e4ca-4403-bb71-66e245036183
# ╟─43c081f1-14c6-4f72-b112-8fa192d862f1
# ╟─8b77b013-f70f-4470-b1bb-5b538edee5e9
# ╟─afe38064-9194-447c-8f88-23980727bab4
# ╟─012af9d3-9095-4420-bffa-b7096665b153
