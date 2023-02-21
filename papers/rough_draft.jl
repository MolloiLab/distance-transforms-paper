### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 706bcf20-ad6a-11ed-1997-ebe169c89f20
# ╠═╡ show_logs = false
using DrWatson; @quickactivate "hd-loss"; using PlutoUI, Images, CSV, DataFrames

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

Segmentation may be one step in a more involved procedure in some applications. For instance, several multimodal medical image registration techniques rely on segmenting a target organ in one or more pictures [CITE]. The Dice similarity coefficient (DSC), overlap, and Hausdorff distance (HD) are some popular metrics that are typically computed to assess the segmentation method relative to some ground truth segmentation [CITE]. HD is one of the most useful metrics for quantifying boundary information. As measured by HD, the highest segmentation error can be a strong indicator of how effective the segmentations are. For two sets of points ``(X, Y)``, the HD is defined by [CITE]. Eq. 1 shows the one-sided HD from ``X`` to ``Y``, Eq. 2 shows the one-sided HD from ``Y`` to ``X``, and Eq. 3 shows the bi-directional HD between both sets.

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

Recently, segmentation approaches, specifically deep learning-based methods, have focused on directly minimizing the HD during the training process via the Hausdorff distance loss function [CITE]. The most accurate form of the HD loss function utilizes a distance transform operation [CITE] to estimate the HD.

Although the HD loss function improves the segmentation error, especially along the boundaries, this loss function is unstable, particularly in the early part of the training process [CITE]. A weighted combination of loss functions can solve this issue, like the Dice loss and HD loss [CITE]. 

Still, the HD loss is more computationally expensive than the more common Dice loss, which limits the usage of HD loss in deep learning. Previous HD loss implementations utilized central processing unit (CPU)-based distance transform operations since those methods are readily available in packages like SciPy [CITE]. The downside to this approach is that all practical deep learning training takes place on the graphics processing unit (GPU), which means the training must switch back and forth from CPU to GPU every time the HD loss is computed. Also, for large arrays, like those commonly found in medical imaging, GPU operations are faster than CPU operations. Previous works have published GPU-optimized distance transform algorithms [CITE + our paper in review], but no reports have taken advantage of these improvements for the HD loss function.

This study presents an HD loss function based on the state-of-the-art distance transform algorithm. This novel HD loss function decreased the computational complexity compared to previous HD loss function algorithms, making it comparable to the gold standard Dice loss function while directly minimizing DSC and HD. All the code is publicly available for use at https://github.com/Dale-Black/DistanceTransforms.jl
"""

# ╔═╡ 0039da08-5d0b-4dfe-be30-bdec06f526e3
md"""
# II. Methods
"""

# ╔═╡ 2c835912-8e15-44da-a512-775730bb5281
md"""
## 2.1 - Distance Transform
"""

# ╔═╡ 585cb69a-98d0-4fe6-8428-4904a4d41dcc
md"""
This study utilized improved distance transform operations to decrease the time associated with previously proposed HD loss functions. The Julia programming language [CITE] was a logical choice since CPU and GPU code can be written in the same high-level language with speed similar to more common languages like C and Fortran.

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

Previous HD loss functions utilized a readily available distance transform algorithm from scipy [CITE]. Another popular library, Tensorflow, provides a CPU and GPU-capable distance transform [CITE]. Each distance transform algorithm was compared to the state-of-the-art distance transform used in this study. We determined each algorithm's speed across various-sized arrays in two and three-dimensions.
"""

# ╔═╡ 38c0828f-72a0-44b2-b2e8-9d8c241b1fee
md"""
## 2.2 - Hausdorff Loss Function
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

Where ``p`` is the binary ground-truth segmentation, ``q`` is the binary prediction, ``d_{p}`` is the distance transform of ``p``, ``d_{q}`` is the distance transform of ``q``, ``\Omega`` represents the set of all pixels, and ``\circ`` represents the Hadamard product (i.e., the element-wise matrix product). The parameter ``\alpha`` determines how strongly the larger errors are penalized. Previous reports show that values of ``\alpha`` between 1 and 3 yielded good results.

HD loss is a good estimator of HD, and _ et al. show that combining HD loss with the Dice loss function (Eq. 5) helps avoid the instability issue associated with HD loss [CITE]. Thus, the resulting loss function used throughout this study directly minimizes DSC and HD and is a combination of ``\text{Loss}_{\text{HD}}`` and ``\text{Loss}_{\text{Dice}}``, shown in Eq. 6

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

Similarly to the pure distance transform timings, various forms of previously published HD loss functions were compared to the HD loss function implemented in this study. Also, the gold standard Dice loss function was included in these timings to provide a baseline operation time - since the HD loss function is typically combined with the Dice loss function to provide stability during training. We determined each loss function's speed across various-sized arrays in two and three-dimensions.
"""

# ╔═╡ 68bc23e7-774d-4378-a9fb-07167ed44411
md"""
## 2.4 - Training
"""

# ╔═╡ fcf6e65d-783d-4839-9b0e-b6f5e9d642eb
md"""
After benchmarking the pure distance transforms and loss functions, we then compared deep-learning training loops. We compared the timings of identical loops, with the only variable coming from the loss function utilized. Since the previously implemented loss functions were written in Python and the loss function in this study is in Julia [CITE], we utilized comparable Julia-based forms of the Python versions. This allowed us to avoid extra variables affecting the timing, from differences in the data loading and processing steps between Python and Julia. In practice, the new loss function implemented in this paper is available for use in Python and Julia via PythonCall.jl [CITE].

The heart dataset from the Medical Segmentation Decathlon was used for training. Along with the timing information, the DSC and HD metrics were calculated on the validation data of the heart dataset and compared between the HD and Dice loss functions.

The Julia programming language was used for the training process, along with key Julia packages [CITE]. A simple UNet model was implemented with {_} channels, {_}
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

# ╔═╡ da0eead6-0a37-4b5c-889f-b2f2ff5dbeca
md"""

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
# ╠═8a852101-15e1-4d00-ae3b-61e94f6236ac
# ╟─0caacaf2-0dd5-4e19-9a60-8fd87953bc15
# ╟─06cccce0-3f96-4a92-a7f7-4d16495d94f0
# ╟─0d12bc5b-53f6-4c1a-99b6-0dc8ad0af5b2
# ╟─7596996f-140c-4ffa-9c73-a8ec7746ca9a
# ╟─0039da08-5d0b-4dfe-be30-bdec06f526e3
# ╟─2c835912-8e15-44da-a512-775730bb5281
# ╟─585cb69a-98d0-4fe6-8428-4904a4d41dcc
# ╟─38c0828f-72a0-44b2-b2e8-9d8c241b1fee
# ╟─5ff24d6d-d385-4efc-807c-c45f0f4a419c
# ╟─68bc23e7-774d-4378-a9fb-07167ed44411
# ╠═fcf6e65d-783d-4839-9b0e-b6f5e9d642eb
# ╟─78c8eb22-b8e5-4b3c-bf61-fbac9be5ba7a
# ╟─fb417691-9c1a-46ec-aa2c-3c9c325c1d73
# ╟─9c02f72e-ee10-4a58-a449-16461a475809
# ╠═da0eead6-0a37-4b5c-889f-b2f2ff5dbeca
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
