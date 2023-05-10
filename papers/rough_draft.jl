### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 706bcf20-ad6a-11ed-1997-ebe169c89f20
# ╠═╡ show_logs = false
using DrWatson

# ╔═╡ 5089782b-fbd4-474f-97b7-ad16698d339b
# ╠═╡ show_logs = false
@quickactivate "hd-loss"

# ╔═╡ 5b3232d5-8842-482e-a3a1-98dd1bb29581
using PlutoUI, Images, CSV, DataFrames

# ╔═╡ 8a852101-15e1-4d00-ae3b-61e94f6236ac
TableOfContents()

# ╔═╡ 0caacaf2-0dd5-4e19-9a60-8fd87953bc15
md"""
# Optimized Hausdorff Loss Function Based on a GPU-Accelerated Distance Transform
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

This study adapted the Felzenszwalb distance transform algorithm [CITE]. The Felzenszwalb algorithm is easily parallelizable, making providing a multi-threaded CPU and GPU-accelerated form simple. Previous HD loss functions utilized a readily available distance transform algorithm from scipy [CITE]. 

The previous distance transform algorithm was compared to the CPU and GPU-accelerated Felzenszwalb distance transform algorithm. We determined each algorithm's speed across various-sized arrays in two and three-dimensions.
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
## 2.3 - Training
"""

# ╔═╡ fcf6e65d-783d-4839-9b0e-b6f5e9d642eb
md"""
After benchmarking the pure distance transforms and loss functions, we then compared deep-learning training loops. We compared the timings of identical loops, with the only changed variable coming from the loss function utilized. The new loss function implemented in this paper is available in Python and Julia via PythonCall.jl [CITE].

The heart dataset from the Medical Segmentation Decathlon was used for training. Along with the timing information, the DSC and HD metrics were calculated on the validation data of the heart dataset and compared between the HD and Dice loss functions.

The Julia programming language was used for the training process, along with crucial Julia packages [CITE]. A simple 3D UNet model was implemented for the training process [CITE]. All loss functions were minimized using the stochastic gradient descent optimizer [CITE], and the learning rate was set to {insert}.

All the training code was implemented in Julia 1.8 and Flux {insert} and run on Windows 10. An NVIDIA {insert} GPU and CUDA {inset} were used for training.
"""

# ╔═╡ fb417691-9c1a-46ec-aa2c-3c9c325c1d73
md"""
# III. Results
"""

# ╔═╡ 13b6d018-9b3c-4eaa-868e-1947c51f0b6f
md"""
## 3.1 - Timings
"""

# ╔═╡ 9c02f72e-ee10-4a58-a449-16461a475809
md"""
### 3.1.1 - Distance Transforms
"""

# ╔═╡ da0eead6-0a37-4b5c-889f-b2f2ff5dbeca
md"""
Two and three-dimensional arrays from size {insert} up to size {insert} were input into the distance transforms. The distance transform utilized in this paper (``\text{DT}_{\text{Felzenszwalb}_\text{GPU}}``) was previously shown to be the fastest implementation for two and three-dimensional arrays within the size range of common medical images, like computed tomography scans [CITE]. For the largest-sized two-dimensional array, ``\text{DT}_{\text{Felzenszwalb}_\text{GPU}}`` was ``11 \times`` faster than ``\text{DT}_{\text{Scipy}}``. For the largest-sized three-dimensional array, ``\text{DT}_{\text{Felzenszwalb}_\text{GPU}}`` was ``[insert] \times`` faster than ``\text{DT}_{\text{Scipy}}``. Figure 1 shows the two, and three-dimensional timings of the various distance transform operations.
"""

# ╔═╡ d5fd0b05-54fa-43b8-9f35-eea3fd260176
load(plotsdir("dt.png"))

# ╔═╡ 4f0ef647-cf78-4f68-be48-2bfffa23f77b
md"""
Figure 1. Distance transform benchmarks. (A) Shows the time to compute the distance transforms (in ms) on various-sized two-dimensional arrays. (B) Shows the time to compute the distance transforms (in ms) on various-sized three-dimensional arrays.
"""

# ╔═╡ 4449f6a2-7727-4bd5-a1b6-394ec1864b81
md"""
### 3.1.2 - Loss Functions
"""

# ╔═╡ 03763d64-afbd-4bac-8a5c-ad700c3200f9
md"""
Two and three-dimensional arrays from size {insert} up to size {insert} were input into the distance transforms. The HD loss function, implemented in this paper (``\text{HD}_{\text{GPU}}^{\text{Felzenszwalb}}``), was faster than any previous implementations. Specifically, ``\text{HD}_{\text{GPU}}^{\text{Felzenszwalb}}`` was ``30 \times`` faster than ``\text{HD}_{\text{CPU}}^{\text{SciPy}}`` on the largest two-dimensional arrays and ``8 \times`` faster on the largest three-dimensional arrays. The input arrays in the training loop were of size ``96 \times 96 \times 96`` (Fig. 2B) and for this size, the ``\text{HD}_{\text{GPU}}^{\text{Felzenszwalb}}`` was ``5 \times`` faster than ``\text{HD}_{\text{CPU}}^{\text{SciPy}}``. Figure 2 shows the two, and three-dimensional timings of the various loss functions.
"""

# ╔═╡ ec837694-cfa1-4abe-8de3-9efcf4b46004
load(plotsdir("loss.png"))

# ╔═╡ a2566ddd-c945-4933-a16d-7f8d1ced2314
md"""
Figure 2. Loss function benchmarks. (A) Shows the time to compute the loss functions (in ms) on various-sized two-dimensional arrays. (B) Shows the time to compute the loss functions (in ms) on various-sized three-dimensional arrays.
"""

# ╔═╡ d899cdf6-4b94-496a-a058-73e394a7ea6a
md"""
### 3.1.3 - Training Loop
"""

# ╔═╡ 96fe5180-b3e9-4994-aa62-fab24592b6cd
md"""
A simplified training loop was run for {insert} epochs, and the average step time and epoch time were examined for each loss function. Based on previous reports [CITE], combining the HD loss function with the Dice loss function is recommended to avoid instability issues. Therefore, three different loss functions were analyzed: 1) pure Dice loss function (``\text{Loss}_{\text{DSC}}``), which serves as the baseline, 2) hybrid HD loss and Dice loss function (``\text{Loss}_{\text{DSC\_HD}}^{\text{CPU}}``), which runs on the CPU and is comparable to the previously reported HD loss function [CITE], and 3) our new hybrid HD loss and Dice loss function (``\text{Loss}_{\text{DSC\_HD}}^{\text{GPU}}``), which runs entirely on the GPU.

The average step time for the previously proposed HD loss function ``\text{Loss}_{\text{DSC\_HD}}^{\text{CPU}}`` was 54.2% slower than the baseline ``\text{Loss}_{\text{DSC}}`` (Fig. 3). Similarly, the average epoch time for the previously proposed HD loss function ``\text{Loss}_{\text{DSC\_HD}}^{\text{CPU}}`` was 53.6% slower than the baseline ``\text{Loss}_{\text{DSC}}`` (Fig. 4). The average step time for our new HD loss function ``\text{Loss}_{\text{DSC\_HD}}^{\text{GPU}}`` was 13.6% slower than the baseline ``\text{Loss}_{\text{DSC}}`` (Fig. 3) and 13.5% slower than the baseline per epoch (Fig. 4). 

Figures 3 and 4 show the average step and epoch time, respectively, for each loss function.
"""

# ╔═╡ c2db5713-3167-46f6-8be0-80d3efc8223d
load(plotsdir("training_step.png"))

# ╔═╡ 8ff25665-1863-4c11-b6d5-d6df4b1c6eac
md"""
Figure 3. Average step time (s) for various loss functions. The baseline, ``\text{Loss}_{\text{DSC}}``, took 0.59s on average per step. The previously proposed hybrid Dice and HD loss function, ``\text{Loss}_{\text{DSC\_HD}}^{\text{CPU}}``, took 0.91s on average per step. Our new loss function, ``\text{Loss}_{\text{DSC\_HD}}^{\text{GPU}}``, took 0.67s on average per step.
"""

# ╔═╡ 2ba9e25b-d46e-4b1e-8455-e30e1ff8c155
load(plotsdir("training_epoch.png"))

# ╔═╡ 311657af-b191-480c-9bd3-4cc769b108d0
md"""
Figure 4. Average epoch time (s) for various loss functions. The baseline, ``\text{Loss}_{\text{DSC}}``, took 2.37s on average per step. The previously proposed hybrid Dice and HD loss function, ``\text{Loss}_{\text{DSC\_HD}}^{\text{CPU}}``, took 3.64s on average per step. Our new loss function, ``\text{Loss}_{\text{DSC\_HD}}^{\text{GPU}}``, took 2.69s on average per step.
"""

# ╔═╡ 458c4a6a-4b0e-4a71-bb07-380be9309c51
# begin
# 	perc_increase(a, b) = ((b - a) / a) * 100
	
# 	dice = 0.59
# 	hd_cpu = 0.91
# 	hd_gpu = 0.67
# 	@info perc_increase(dice, hd_cpu)
# 	@info perc_increase(dice, hd_gpu)
# 	@info perc_increase(hd_gpu, hd_cpu)

# 	dice2 = 2.37
# 	hd_cpu2 = 3.64
# 	hd_gpu2 = 2.69
# 	@info perc_increase(dice2, hd_cpu2)
# 	@info perc_increase(dice2, hd_gpu2)
# 	@info perc_increase(hd_gpu2, hd_cpu2)
# end

# ╔═╡ e548262f-15fc-4446-afa0-e684a0526ad0
md"""
## 3.4 - Accuracy
"""

# ╔═╡ efa8acce-9697-44ae-8518-38faafb27807
md"""
This study focused on solving the computational complexity issue associated with previously proposed HD loss functions. Therefore, benchmarking the loss functions was the main concern. But the purpose of the HD loss function is to improve upon the segmentation compared to the gold standard Dice loss function [CITE]. Therefore, we trained a deep learning model with the Dice loss (``\text{Loss}_{\text{DSC}}``) and our fast hybrid Dice-HD loss function (``\text{Loss}_{\text{DSC\_HD}}^{\text{GPU}}``) and compared the pertinent resulting metrics (DSC and HD) of each model to demonstrate the effectiveness of the HD loss function. Table 1 shows both models' 90th percentile (highest) DSC and 10th percentile (lowest) HD. ``\text{Loss}_{\text{DSC\_HD}}^{\text{GPU}}`` improved upon the baseline model's DSC from 0.82 to 088. Similarly, ``\text{Loss}_{\text{DSC\_HD}}^{\text{GPU}}`` improved upon the baseline model's HD from 9.11 to 5.20. Figure 5 shows the HD and DSC of both models at each epoch. Both the DSC and HD are improved compared to the baseline DSC model, with less variation between each epoch.

Figure 6 shows example slices of the heart dataset with the ground truth segmentation and predicted segmentation overlayed on the slice. The left column shows the predicted mask from the Dice loss function, and the right column shows the predicted mask from the HD-Dice loss.
"""

# ╔═╡ 43c081f1-14c6-4f72-b112-8fa192d862f1
CSV.read(datadir("analysis", "accuracy.csv"), DataFrame)

# ╔═╡ 4d8340f7-e4ca-4403-bb71-66e245036183
load(plotsdir("dice_hd_julia.png"))

# ╔═╡ c39110b1-ba24-45bc-bb8f-372e51eb8d85
md"""
Figure 5. The HD and DSC of both models at each epoch. Both the DSC and HD are improved compared to the baseline DSC model, with less variation between each epoch.
"""

# ╔═╡ 37c90298-c36f-4ced-a3df-af5140d1f23b
load(plotsdir("contours.png"))

# ╔═╡ 4e1069cb-039c-4c62-97b7-96c34c1ed55e
md"""
Figure 6. Selected slices of the heart dataset with ground truth and predicted boundaries overlayed. The left column shows the baseline Dice loss model and the right column shows the hybrid HD-Dice loss proposed in this study.
"""

# ╔═╡ 8b77b013-f70f-4470-b1bb-5b538edee5e9
md"""
# IV. Discussion
"""

# ╔═╡ 09cee8f5-a5f5-4dac-b964-386e55404a8a
md"""
The present study proposes a novel HD loss function based on a state-of-the-art distance transform algorithm. The proposed algorithm decreased the computational complexity compared to previously proposed HD loss functions, making it comparable to the gold standard Dice loss function while directly minimizing the DSC and HD.

The HD is one of the most useful metrics for quantifying boundary information, and it can be a strong indicator of how effective the segmentations are. The proposed method can improve segmentations, particularly along the boundaries, and can be applied to various image segmentation problems, including medical imaging.

Previous works have focused on directly minimizing the HD during the training process via the HD loss function. However, the HD loss function is unstable, particularly in the early part of the training process, which limits its applicability. Although a weighted combination of loss functions, such as the Dice loss and HD loss, can solve this issue. Still, the HD loss is more computationally expensive than the Dice loss.

The proposed method addresses this issue by utilizing a GPU-accelerated distance transform algorithm.

Future work may involve applying the proposed method to other image segmentation tasks, such as object detection and recognition. Also, investigating the performance of the proposed method on larger datasets may be of interest.
"""

# ╔═╡ afe38064-9194-447c-8f88-23980727bab4
md"""
# V. Conclusion
"""

# ╔═╡ e2de2050-059b-43ae-8bf9-237dcff9f2a6
md"""
In conclusion, the proposed method presents an efficient HD loss function based on a GPU-accelerated distance transform algorithm that can improve segmentation accuracy. The method may have applications in various image segmentation tasks and can be implemented using publicly available code.
"""

# ╔═╡ 012af9d3-9095-4420-bffa-b7096665b153
md"""
# References
"""

# ╔═╡ Cell order:
# ╠═706bcf20-ad6a-11ed-1997-ebe169c89f20
# ╠═5089782b-fbd4-474f-97b7-ad16698d339b
# ╠═5b3232d5-8842-482e-a3a1-98dd1bb29581
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
# ╟─fcf6e65d-783d-4839-9b0e-b6f5e9d642eb
# ╟─fb417691-9c1a-46ec-aa2c-3c9c325c1d73
# ╟─13b6d018-9b3c-4eaa-868e-1947c51f0b6f
# ╟─9c02f72e-ee10-4a58-a449-16461a475809
# ╟─da0eead6-0a37-4b5c-889f-b2f2ff5dbeca
# ╟─d5fd0b05-54fa-43b8-9f35-eea3fd260176
# ╟─4f0ef647-cf78-4f68-be48-2bfffa23f77b
# ╟─4449f6a2-7727-4bd5-a1b6-394ec1864b81
# ╟─03763d64-afbd-4bac-8a5c-ad700c3200f9
# ╟─ec837694-cfa1-4abe-8de3-9efcf4b46004
# ╟─a2566ddd-c945-4933-a16d-7f8d1ced2314
# ╟─d899cdf6-4b94-496a-a058-73e394a7ea6a
# ╟─96fe5180-b3e9-4994-aa62-fab24592b6cd
# ╟─c2db5713-3167-46f6-8be0-80d3efc8223d
# ╟─8ff25665-1863-4c11-b6d5-d6df4b1c6eac
# ╟─2ba9e25b-d46e-4b1e-8455-e30e1ff8c155
# ╟─311657af-b191-480c-9bd3-4cc769b108d0
# ╟─458c4a6a-4b0e-4a71-bb07-380be9309c51
# ╟─e548262f-15fc-4446-afa0-e684a0526ad0
# ╟─efa8acce-9697-44ae-8518-38faafb27807
# ╟─43c081f1-14c6-4f72-b112-8fa192d862f1
# ╟─4d8340f7-e4ca-4403-bb71-66e245036183
# ╟─c39110b1-ba24-45bc-bb8f-372e51eb8d85
# ╟─37c90298-c36f-4ced-a3df-af5140d1f23b
# ╟─4e1069cb-039c-4c62-97b7-96c34c1ed55e
# ╟─8b77b013-f70f-4470-b1bb-5b538edee5e9
# ╟─09cee8f5-a5f5-4dac-b964-386e55404a8a
# ╟─afe38064-9194-447c-8f88-23980727bab4
# ╟─e2de2050-059b-43ae-8bf9-237dcff9f2a6
# ╟─012af9d3-9095-4420-bffa-b7096665b153
