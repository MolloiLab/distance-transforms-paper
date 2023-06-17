### A Pluto.jl notebook ###
# v0.19.26

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

Although the HD loss function improves the segmentation error, especially along the boundaries, this loss function is unstable, particularly in the early part of the training process [CITE]. A weighted combination of loss functions can solve this issue, like the combination of Dice loss and HD loss [CITE]. 

Still, the HD loss is more computationally expensive than the more common Dice loss, which limits the usage of HD loss in deep learning. Previous HD loss implementations utilized central processing unit (CPU)-based distance transform operations since those methods are readily available in packages like SciPy [CITE]. The downside to this approach is that all practical deep learning training takes place on the graphics processing unit (GPU), which means the training must switch back and forth from CPU to GPU every time the HD loss is computed. Also, for large arrays, like those commonly found in medical imaging, GPU operations are faster than CPU operations. Previous works have published GPU-optimized distance transform algorithms [CITE], but no reports have taken advantage of these improvements for the HD loss function.

This study presents an HD loss function based on the Felzenszwalb distance transform algorithm. This novel HD loss function decreased the computational complexity compared to previous HD loss function algorithms, making it comparable to the gold standard Dice loss function while directly minimizing DSC and HD. All the code is publicly available for use at https://github.com/Dale-Black/DistanceTransforms.jl
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
To implement a new distance transform operation, the Julia programming language [CITE] was a logical choice since CPU and GPU code can be written in the same high-level language with speed similar to more common languages like C and Fortran.

We adapted the Felzenszwalb distance transform algorithm [CITE]. The Felzenszwalb algorithm is easily parallelizable, which makes it simple to provide a multi-threaded CPU and GPU-accelerated form. Algorithm 1 shows the one-dimensional Felzenszwalb algorithm and code block 1 and 2 shows the direct Julia implementation of both the one-dimensional and three-dimensional distance transforms. 

Previous HD loss functions utilized a readily available distance transform algorithm from Scipy [CITE]. The previous distance transform algorithm was compared to the CPU and GPU-accelerated Felzenszwalb distance transform algorithm. We determined each algorithm's speed across various-sized arrays in two and three-dimensions.

```
\begin{algorithmic}
	\Require A one-dimensional array f[1:n] with n elements
	\Ensure  An array D[1:n] where D[i] represents the transformed distance of f[i]
	\State Define an empty array of parabolas: parabolas[]
	\State Define an array of intersection points: intersection[] with same length as f, initialized to infinity
	\State parabolas[1] $\leftarrow$ 1
	\State k $\leftarrow$ 1
	\For{i = 2 to n}
		\While{k > 0 and f[i] + (i$^2$) < f[parabolas[k]] + (intersection[k]$^2$)}
			\State k $\leftarrow$ k - 1
		\EndWhile
		\If{k $>$ 0}
			\State intersection[k+1] $\leftarrow$ (f[i] - f[parabolas[k]] + i$^2$ - parabolas[k]$^2$) / (2$\times$ i - 2$\times$ parabolas[k])
		\EndIf
		\State parabolas[k+1] $\leftarrow$ i
		\State k $\leftarrow$ k + 1
	\EndFor
	\State k $\leftarrow$ 1
	\For{i = 1 to n}
		\While{intersection[k+1] < i}
			\State k $\leftarrow$ k + 1
		\EndWhile
		\State D[i] $\leftarrow$ (i - parabolas[k])$^2$ + f[parabolas[k]]
	\EndFor
	\State \Return D
	\end{algorithmic}
	\label{algo}
\end{algorithmic}
```
```julia
function transform(f::AbstractVector; 
				   output = similar(f, Float32), 
				   v = ones(Int32, length(f)), 
				   z = ones(Float32, length(f)+1))
	z[1] = -1f10
	z[2] = 1f10
	k = 1 # Index of the rightmost parabola in the lower envelope
	for q = 2:length(f)
		s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
		while s <= z[k]
			k -= 1
			s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
		end
		k += 1
		v[k] = q
		z[k] = s
		z[k+1] = 1f10
	end
	k = 1
	for q in 1:length(f)
		while z[k+1] < q
			k += 1
		end
		output[q] = (q-v[k])^2 + f[v[k]]
	end
	return output
end
```

```julia
function transform(vol::AbstractArray;
            output = similar(vol, Float32), 
			v = ones(Int32, size(vol)), 
			z = ones(Float32, size(vol) .+ 1))
        	
	for k in CartesianIndices(@view(vol[1,:,:]))
		transform(vol[:, k]; output=output[:, k], v=v[:, k], z=z[:, k])
	end
	output2 = similar(output)
	
	for k in CartesianIndices(@view(vol[:,1,:]))
		transform(output[k[1], :, k[2]]; output=output2[k[1], :, k[2]], v=fill!(v[k[1], :, k[2]], 1), z=fill!(z[k[1], :, k[2]], 1))
	end
	
	for k in CartesianIndices(@view(vol[:,:,1]))
		transform(output2[k, :]; output=output[k, :], v=fill!(v[k, :], 1), z=fill!(z[k, :], 1))
	end
	return output
end
```
"""

# ╔═╡ 38c0828f-72a0-44b2-b2e8-9d8c241b1fee
md"""
## 2.2 - Hausdorff Loss Function
"""

# ╔═╡ 5ff24d6d-d385-4efc-807c-c45f0f4a419c
md"""
The HD loss function was previously described by [CITE] and can be seen in Eq. 5 

```math
\begin{aligned}
\text{Loss}_{\text{HD}}(q, p) = \frac{1}{|\Omega|} \sum_{\Omega}((p - q)^{2} \circ (d_{p}^{\alpha} + d_{q}^{\alpha}))
\end{aligned}
\tag{5}
```

Where ``p`` is the binary ground-truth segmentation, ``q`` is the binary prediction, ``d_{p}`` is the distance transform of ``p``, ``d_{q}`` is the distance transform of ``q``, ``\Omega`` represents the set of all pixels, and ``\circ`` represents the Hadamard product (i.e., the element-wise matrix product). The parameter ``\alpha`` determines how strongly the larger errors are penalized. Previous reports show that values of ``\alpha`` between 1 and 3 yielded good results.

HD loss is a good estimator of HD, and _ et al. show that combining HD loss with the Dice loss function (Eq. 6) helps avoid the instability issue associated with HD loss [CITE]. Thus, the resulting loss function used throughout this study directly minimizes DSC and HD and is a combination of ``\text{Loss}_{\text{HD}}`` and ``\text{Loss}_{\text{DSC}}``, shown in Eq. 7

```math
\begin{aligned}
\text{Loss}_{\text{DSC}}(q, p) = 1 - \dfrac{2\sum_{\Omega}(p \circ q)}{\sum_{\Omega}(p^2 +q^2)}
\end{aligned}
\tag{6}
```

```math
\begin{aligned}
\text{Loss}(q, p) = \lambda \Bigg(\dfrac{1}{\lvert\Omega\rvert}\sum_{\Omega} \bigg((p-q)^2\circ (d_p + d_q)\bigg)\Bigg)+ (1-\lambda) \Bigg(1 - \dfrac{2\sum_{\Omega}(p \circ q)}{\sum_{\Omega}(p^2 +q^2)}\Bigg)
\end{aligned}
\tag{7}
```

Similarly to the pure distance transform timings, a previously published HD loss function was compared to the HD loss function implemented in this study. Also, the gold standard Dice loss function was included in these timings to provide a baseline operation time - since the HD loss function is typically combined with the Dice loss function to provide stability during training. We determined each loss function's speed across various-sized arrays in two and three-dimensions.
"""

# ╔═╡ 68bc23e7-774d-4378-a9fb-07167ed44411
md"""
## 2.3 - Training
"""

# ╔═╡ fcf6e65d-783d-4839-9b0e-b6f5e9d642eb
md"""
After benchmarking the pure distance transforms and pure loss functions on various sized arrays, we then benchmarked the average time per epoch on a simple but realistic deep learning training loop. We utilized the heart CT dataset from publicly available  Medical Segmentation Decathlon [CITE]. For the trainig loop bencharms, we used a 3D UNet model [CITE] with [...INSERT UNET INFO HERE]. We ran the training loop for 40 epochs and found the average training time per epoch for all three loss functions (``\text{Loss}_{DSC}``, ``\text{Loss}_{Scipy}``, ``\text{Loss}_{FelzenszwalbGPU}``).

After determining the fastest HD loss function, we then ran a more exhaustive training process to validate the improved accuracy of the hybrid HD loss compared to the baseline DSC loss. The 3D UNet model consisted of [...INSERT UNET INFO HERE] and was run for 500 epochs for each loss function.

The studies loss function is made available in both Python and Julia via juliacall/PythonCall.jl [CITE]. All the training code was implemented in Julia 1.8 and Flux {insert} and run on Windows 10. An NVIDIA {insert} GPU and CUDA {inset} were used for training.
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
Two and three-dimensional arrays with 16 up to 16777216 elements were input into the distance transforms. The distance transform utilized in this paper (``\text{DT}_{FelzenszwalbGPU}``) was shown to be the fastest implementation for two and three-dimensional arrays within the size range of common medical images, like computed tomography scans [CITE]. Specifically, Figure 1A shows a dahsed vertical line of size ``96 \times 96 \times 96`` which corresponds to the size of the images used in this studies training. For the largest-sized two-dimensional array, ``\text{DT}_{FelzenszwalbGPU}`` was ``53 \times`` faster than ``\text{DT}_{Scipy}``. For the largest-sized three-dimensional array, ``\text{DT}_{FelzenszwalbGPU}`` was ``31 \times`` faster than ``\text{DT}_{Scipy}``. Figure 1 shows the two, and three-dimensional timings of the various distance transform algorithms.
"""

# ╔═╡ d5fd0b05-54fa-43b8-9f35-eea3fd260176
load(plotsdir("dt.png"))

# ╔═╡ 4f0ef647-cf78-4f68-be48-2bfffa23f77b
md"""
Figure 1. Distance transform benchmarks. (A) Shows the time to compute the distance transforms (in seconds) on various-sized two-dimensional arrays. (B) Shows the time to compute the distance transforms (in s) on various-sized three-dimensional arrays.
"""

# ╔═╡ 4449f6a2-7727-4bd5-a1b6-394ec1864b81
md"""
### 3.1.2 - Loss Functions
"""

# ╔═╡ 03763d64-afbd-4bac-8a5c-ad700c3200f9
md"""
Two and three-dimensional arrays with 16 up to 16777216 elements were input into the Hausdorff loss functions, with the only variation coming from the type of distance transform algorithm used in the Hausdorff loss function. The Hausdorff loss function, implemented in this paper (``\text{HD}_{FelzenszwalbGPU}``), was faster than any previous implementation. Specifically, ``\text{HD}_{FelzenszwalbGPU}`` was ``38 \times`` faster than ``\text{HD}_{SciPy}`` on the largest two-dimensional arrays and ``[INSERT]`` faster on the largest three-dimensional arrays. The input arrays in the training loop were of size ``96 \times 96 \times 96`` (Fig. 2B) and for this size, the ``\text{HD}_{FelzenszwalbGPU}`` was ``31 \times`` faster than ``\text{HD}_{SciPy}``. Figure 2 shows the two, and three-dimensional timings of the various loss functions.
"""

# ╔═╡ ec837694-cfa1-4abe-8de3-9efcf4b46004
load(plotsdir("loss.png"))

# ╔═╡ a2566ddd-c945-4933-a16d-7f8d1ced2314
md"""
Figure 2. Loss function benchmarks. (A) Shows the time to compute the loss functions (in seconds) on various-sized two-dimensional arrays. (B) Shows the time to compute the loss functions (in seconds) on various-sized three-dimensional arrays.
"""

# ╔═╡ d899cdf6-4b94-496a-a058-73e394a7ea6a
md"""
### 3.1.3 - Training Loop
"""

# ╔═╡ 96fe5180-b3e9-4994-aa62-fab24592b6cd
md"""
A simplified training loop was run for 40 epochs, and the average epoch time was examined for each loss function. Based on previous reports [CITE], combining the HD loss function with the Dice loss function is recommended to avoid instability issues. Therefore, three different loss functions were analyzed: 1) pure Dice loss function (``\text{Loss}_{DSC}``), which serves as the baseline, 2) a hybrid HD loss and Dice loss function which utilized the ``\text{DT}_{Scipy}`` for the distance transform algorithm within the Hausdorff loss function, and corresponds to the previously reported HD loss function [CITE], and 3) our new hybrid HD loss and Dice loss function (``\text{Loss}_{FelzenszwalbGPU}``), which runs entirely on the GPU and uses ``\text{DT}_{FelzenszwalbGPU}`` within the Hausdorff loss function.

The average epoch time for the previously proposed HD loss function ``\text{Loss}_{Scipy}`` was 126% slower than the baseline ``\text{Loss}_{DSC}`` (Fig. 4). The average step time for our new HD loss function ``\text{Loss}_{FelzenszwalbGPU}`` was 11% slower than the baseline per epoch (Fig. 4). 

Figure 4 shows the average epoch time for each loss function.
"""

# ╔═╡ 2ba9e25b-d46e-4b1e-8455-e30e1ff8c155
load(plotsdir("training.png"))

# ╔═╡ 311657af-b191-480c-9bd3-4cc769b108d0
md"""
Figure 4. Average epoch time (s) for various loss functions. The baseline, ``\text{Loss}_{DSC}``, took 5.77s on average per step. The previously proposed hybrid Dice and HD loss function, ``\text{Loss}_{Scipy}``, took 8.38s on average per step. Our new loss function, ``\text{Loss}_{FelzenszwalbGPU}``, took 6.60s on average per step.
"""

# ╔═╡ e548262f-15fc-4446-afa0-e684a0526ad0
md"""
## 3.4 - Accuracy
"""

# ╔═╡ efa8acce-9697-44ae-8518-38faafb27807
md"""
This study focused on solving the computational complexity issue associated with previously proposed Hausdorff loss functions [CITE]. Therefore, benchmarking the Hausdorff loss functions was the main concern. But the purpose of the Hausdorff loss function is to improve upon the segmentation compared to the gold standard Dice loss function [CITE]. Therefore, we trained a deep learning model with the Dice loss (``\text{Loss}_{DSC}``) and our fast hybrid Dice-HD loss function (``\text{Loss}_{FelzenszwalbGPU}``) and compared the pertinent resulting metrics (Dice Similarity Coefficient and Hausdorff Distance) of each model to demonstrate the effectiveness of the Hausdorff loss function. Figure 5 shows both models' DSC and HD metrics at every tenth epoch on the validation set. Table 1 shows both models' best DSC and lowest HD and the 90th and 10th percentile DSC and HD metrics, respectively. ``\text{Loss}_{FelzenszwalbGPU}`` improved upon the baseline model's DSC from 0.918 to 0.925. Similarly, ``\text{Loss}_{FelzenszwalbGPU}`` improved upon the baseline model's HD from 4.58 to 3.46. Both the DSC and HD are improved compared to the baseline DSC model, with less variation between each epoch (Fig. 5).

Figure 6 shows example slices of the heart dataset with the ground truth segmentation and predicted segmentation overlayed on the slice. The left column shows the predicted mask from ``\text{Loss}_{DSC}``, and the right column shows the predicted mask from ``\text{Loss}_{FelzenszwalbGPU}``.
"""

# ╔═╡ 43c081f1-14c6-4f72-b112-8fa192d862f1
CSV.read(datadir("analysis", "metrics_results.csv"), DataFrame)

# ╔═╡ 4d8340f7-e4ca-4403-bb71-66e245036183
load(plotsdir("dice_hd_julia.png"))

# ╔═╡ c39110b1-ba24-45bc-bb8f-372e51eb8d85
md"""
Figure 5. The HD and DSC of both models at every tenth epoch for the validation set. Both the DSC and HD are improved compared to the baseline DSC model, with less variation between each epoch.
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
We introduced a new Hausdorff loss function, based on a GPU accelerated distance transform algorithm. This novel loss function reduces the computational complexity relative to previously proposed Hausdorff loss functions and reduces the time to be within 11% of the gold standard Dice loss function, while concurrently minimizing the DSC and HD better than the pure Dice loss function.

The HD metric is instrumental in quantifying boundary information, serving as an indicator of segmentation efficacy. The method proposed can enahnce segmentations, particularly at the boundaries, and is applicable to a variety of image segmentation challenges, including those found in medical imaging (Zhou et al., 2018)[1].
    
In previous research, Karimi et al. (2019)[2] employed the ``\text{DT}_{Scipy}`` within the Hausdorff loss function (``\text{HD}_{Scipy}``) and combined with the dice loss function to calculate a weighted loss. Our implementation of the Hausdorff loss function (``\text{HD}_{FelzenszwalbGPU}``) was benchmarked using both ``\text{DT}_{Scipy}`` and ``\text{DT}_{FelzenszwalbGPU}`` in the training loop. Given the variations in the dataset, deep learning model, computational language, we opted to reimplement the previous approach directly to minimize confounding variables. Thus, to ensure a robust comparison, we replicated Karimi et al.'s approach under identical conditions to those used for our implementation ``\text{HD}_{FelzenszwalbGPU}``.
    
Our findings align with those of Karimi et al. in that we observed an increase in training when comparing hybrid hausdorff loss function (``\text{Loss}_{FelzenszwalbGPU}``) to pure Dice loss function (``\text{Loss}_{DSC}``). However, our optimized hausdorff loss function only increased the total training time by 11% compared to 126% for ``\text{Loss}_{Scipy}`` (Fig. 4). Direct comparisons to Karimi et al. are not feasible due to the aforementioned confounding variables and the lack of clear stopping criteria during training. As our study is focused on timing optimization, not accuracy, we applied identical stopping criteria to both the dice loss function and the hausdorff loss function training loops.
    
Our results are in broad agreement with previous research (Sudre et al., 2017)[3], which also indicated improved performance when combining hausdorff and dice loss functions during training. However, as the stopping criteria during training were not specified for the various loss functions in the cited studies, variables such as total epochs may differ, precluding direct comparisons. Therefore, while our findings concur with prior research regarding timing results, further investigations are necessary to ensure comparability in terms of accuracy.
    
Our results demonstrate that, on average, ``\text{HD}_{FelzenszwalbCPU}`` is 9.6 times faster than ``\text{HD}_{Scipy}`` on three-dimensional arrays. The Hausdorff loss function with Felzenszwalb's distance transform on the GPU (``\text{HD}_{FelzenszwalbGPU}``) is 31.3 times faster than ``\text{HD}_{Scipy}``. The average epoch time with the hybrid Hausdorff and Dice loss function (``\text{Loss}_{FelzenszwalbGPU}``) is 2.0 times faster than ``\text{Loss}_{Scipy}``.
    
Prior research (Milletari et al., 2016)[4] has concentrated on directly minimizing the HD during the training process via the hausdorff loss function. However, the hausdorff loss function is unstable, particularly in the early part of the training process, which limits its applicability. Although a weighted combination of loss functions, such as the Dice loss and hausdorff loss, can solve this issue. Still, the hausdorff loss is more computationally expensive than the Dice loss and further work needs to be done to investigate the use of more recently proposed distance transform algorithms [CITE] for use in the Hausdorff loss function to further decrease time to compute the Hausdorff loss. Previous studies [CITE] have looked at operations like convolutional kernels and morphological erosion/dilation in order to avoid the computational complexity of distance transforms within the Hausdorff loss function. The results of such operations are lacking in direct minimization of HD compared to true distance transform based Hausdorff loss functions [CITE]. Our results, therefore, introduce an accurate and efficient approach to the Hausdorff loss function.

Our results diverge from previous studies that have focused solely on Dice loss functions for medical image segmentation. In those studies, the emphasis was on achieving good overlap with the ground truth, ignoring the positional accuracy and shape of the segmented region. Our method, however, combines the Dice and Hausdorff loss functions, ensuring not only overlap but also accurate localization and shape preservation of the segmented regions. This can be particularly useful in clinical scenarios where precise delineation of the region of interest can significantly impact the subsequent steps, such as in tumor detection or organ segmentation (Crum et al., 2006)[5]. However, the potential drawback of increased sensitivity to noise and outliers due to the Hausdorff distance might cause issues. Future studies on trickier datasets, like centerline segmentation [CITE], are needed to investigate this. Nevertheless, given the overall improved performance in terms of segmentation precision and robustness to large discrepancies, the combined use of the Dice loss function and our optimized Hausdorff loss function could offer significant advantages (Taha and Hanbury, 2015)[6] while remaining computationally efficient.
    
Prior attempts to implement Hausdorff loss functions were hampered by computational inefficiencies due to the CPU-based distance transform operations. This limitation had, until now, prevented the widespread adoption of Hausdorff loss functions in deep learning for at least two reasons - (1) the time increase of the Hausdorff loss function associated with the distance transform and (2) the complexity associated with handling the back and forth of data between the CPU and GPU within a deep learning training loop. Our approach results in a GPU compatible Hausdorff loss function, which addresses these issues and brings the computational cost closer to that of the pure Dice loss function. This Dice loss function is widely used in the field and our approach can be a drop in replacement for the Dice loss function (Ronneberger et al., 2015)[7].
    
Potential implications of our findings include enhanced segmentation performance in a wide range of medical imaging applications, including MRI and CT scan analysis. By improving the precision and robustness of automatic segmentations, our method could contribute to the acceleration of medical image analysis and the reduction of inter and intra-observer variability, thereby enhancing diagnostic accuracy and patient outcomes (Litjens et al., 2017)[8].
"""

# ╔═╡ 0a319ec1-d4bb-4acd-8da2-ec3a40ecadf9
md"""
(possibly useful references for discussion)

1. Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A nested U-net architecture for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (pp. 3-11). Springer, Cham​1​.

2. Karimi, D., & Salcudean, S. E. (2019). Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks​2​.

3. Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Cardoso, M. J. (2017). Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (pp. 240-248). Springer, Cham​3​.

4. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 Fourth International Conference on 3D Vision (3DV) (pp. 565-571). IEEE​4​.

5. Crum, W. R., Camara, O., & Hill, D. L. (2006). Generalized overlap measures for evaluation and validation in medical image analysis. IEEE transactions on medical imaging, 25(11), 1451-1461​5​.

6. Taha, A. A., & Hanbury, A. (2015). Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC medical imaging, 15(1), 1-28​6​.

7. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham​7​.

8. Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., ... & Sánchez, C. I. (2017). A survey on deep learning in medical image analysis. Medical image analysis, 42, 60-88​8​.
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
# ╠═7596996f-140c-4ffa-9c73-a8ec7746ca9a
# ╟─0039da08-5d0b-4dfe-be30-bdec06f526e3
# ╟─2c835912-8e15-44da-a512-775730bb5281
# ╠═585cb69a-98d0-4fe6-8428-4904a4d41dcc
# ╟─38c0828f-72a0-44b2-b2e8-9d8c241b1fee
# ╠═5ff24d6d-d385-4efc-807c-c45f0f4a419c
# ╟─68bc23e7-774d-4378-a9fb-07167ed44411
# ╠═fcf6e65d-783d-4839-9b0e-b6f5e9d642eb
# ╟─fb417691-9c1a-46ec-aa2c-3c9c325c1d73
# ╟─13b6d018-9b3c-4eaa-868e-1947c51f0b6f
# ╟─9c02f72e-ee10-4a58-a449-16461a475809
# ╠═da0eead6-0a37-4b5c-889f-b2f2ff5dbeca
# ╟─d5fd0b05-54fa-43b8-9f35-eea3fd260176
# ╠═4f0ef647-cf78-4f68-be48-2bfffa23f77b
# ╟─4449f6a2-7727-4bd5-a1b6-394ec1864b81
# ╠═03763d64-afbd-4bac-8a5c-ad700c3200f9
# ╟─ec837694-cfa1-4abe-8de3-9efcf4b46004
# ╠═a2566ddd-c945-4933-a16d-7f8d1ced2314
# ╟─d899cdf6-4b94-496a-a058-73e394a7ea6a
# ╠═96fe5180-b3e9-4994-aa62-fab24592b6cd
# ╟─2ba9e25b-d46e-4b1e-8455-e30e1ff8c155
# ╠═311657af-b191-480c-9bd3-4cc769b108d0
# ╟─e548262f-15fc-4446-afa0-e684a0526ad0
# ╠═efa8acce-9697-44ae-8518-38faafb27807
# ╟─43c081f1-14c6-4f72-b112-8fa192d862f1
# ╟─4d8340f7-e4ca-4403-bb71-66e245036183
# ╟─c39110b1-ba24-45bc-bb8f-372e51eb8d85
# ╟─37c90298-c36f-4ced-a3df-af5140d1f23b
# ╟─4e1069cb-039c-4c62-97b7-96c34c1ed55e
# ╟─8b77b013-f70f-4470-b1bb-5b538edee5e9
# ╠═09cee8f5-a5f5-4dac-b964-386e55404a8a
# ╠═0a319ec1-d4bb-4acd-8da2-ec3a40ecadf9
# ╟─afe38064-9194-447c-8f88-23980727bab4
# ╟─e2de2050-059b-43ae-8bf9-237dcff9f2a6
# ╟─012af9d3-9095-4420-bffa-b7096665b153
