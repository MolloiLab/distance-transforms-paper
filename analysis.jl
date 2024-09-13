### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 2c729da6-40e6-47cd-a14d-c152b8789b17
# ╠═╡ show_logs = false
using Pkg; Pkg.activate("."), Pkg.instantiate();

# ╔═╡ 33e02405-1750-48f9-9776-d1d2d261f63f
using DrWatson: datadir

# ╔═╡ a968bcd8-fc42-45ec-af7c-68e73e8f1cd5
using PlutoUI: TableOfContents

# ╔═╡ 50e24ebe-403a-4d89-b02f-7a1577222838
using CSV: read

# ╔═╡ 50bfb09f-4dbb-4488-9284-7eef837ffe75
using DataFrames: DataFrame

# ╔═╡ d1a12515-a9d0-468b-8978-dbb26a1ee667
using CairoMakie: Figure, Axis, barplot!, Label, PolyElement, Legend, @L_str

# ╔═╡ e39675a9-08c7-4a4a-8eba-021862757a40
using CairoMakie # For the use of `Makie.wong_colors`

# ╔═╡ 30f67101-9626-4d01-a6fd-c260cd5c29b6
# CUDA.set_runtime_version!(v"11.8")

# ╔═╡ 278dfa0e-46e1-4789-9f51-eb3463a9fb00
TableOfContents()

# ╔═╡ 8b786cbc-dd81-4208-9eee-2d7f7bbfa23f
md"""
# Julia DT Benchmarks
"""

# ╔═╡ ad97f6cb-c331-4898-9c6c-485582058e4d
df_metal_2d = read(datadir("dt_2D_Metal.csv"), DataFrame);

# ╔═╡ 83f4fd58-e801-4dda-9ba7-f5eec56722f6
df_cuda_2d = read(datadir("dt_2D_CUDA.csv"), DataFrame);

# ╔═╡ c7c6aa70-6e46-4444-b8df-68895b55d642
df_oneapi_2d = read(datadir("dt_2D_oneAPI.csv"), DataFrame);

# ╔═╡ d86c512c-b3dc-4542-8c2c-27b72019dce1
df_amdgpu_2d = read(datadir("dt_2D_AMDGPU.csv"), DataFrame);

# ╔═╡ eb190959-b90f-4dbb-8ae7-09b964e1a1c2
df_metal_3d = read(datadir("dt_3D_Metal.csv"), DataFrame);

# ╔═╡ 1936dff5-1d17-4773-9009-51ec95eb9411
df_cuda_3d = read(datadir("dt_3D_CUDA.csv"), DataFrame);

# ╔═╡ 2ff50a99-aaf0-4282-a194-6fff6f50dea6
df_oneapi_3d = read(datadir("dt_3D_oneAPI.csv"), DataFrame);

# ╔═╡ facdc420-5c39-4057-853e-bbab8f96fac6
df_amdgpu_3d = read(datadir("dt_3D_AMDGPU.csv"), DataFrame);

# ╔═╡ 492df5fa-e20e-4dcb-8c1f-b7e14d9fc2de
title_2d = "Performance Comparison \nof Julia Distance Transforms (2D)"

# ╔═╡ 7bc02cb0-76e9-4654-b17a-9d95089bf472
dt_names_2d = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (AMDGPU)", "Proposed (Metal)", "Proposed (oneAPI)"]

# ╔═╡ b50a4061-4f49-4578-8671-1746d532c9dc
range_names_2d = [L"(2^3)^2", L"(2^4)^2", L"(2^5)^2", L"(2^6)^2", L"(2^7)^2", L"(2^8)^2", L"(2^9)^2", L"(2^{10})^2", L"(2^{11})^2", L"(2^{12})^2"]

# ╔═╡ 08676d5b-f098-43a9-8bc3-b5cda3282b2a
title_3d = "Performance Comparison \nof Julia Distance Transforms (3D)"

# ╔═╡ f093102d-4796-4d05-943c-c314febe7342
dt_names_3d = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (AMDGPU)", "Proposed (Metal)", "Proposed (oneAPI)"]

# ╔═╡ 0c09ef6c-d05e-4f73-9075-78d9ba986bb9
range_names_3d = [L"(2^0)^3", L"(2^1)^3", L"(2^2)^3", L"(2^3)^3", L"(2^4)^3", L"(2^5)^3", L"(2^6)^3", L"(2^7)^3", L"(2^8)^3"]

# ╔═╡ 335fe4b9-a11f-4cb9-ac81-68d305f73a2d
md"""
## Combined Barplot
"""

# ╔═╡ bad637b7-3449-4481-846f-e5160cdfca40
let
	### ------------------- 2D PLOT ------------------- ###
	title_2d = "Performance Comparison \nof Julia Distance Transforms (2D)"
	dt_names_2d = dt_names_2d
	sizes_2d = df_metal_2d[:, :sizes]
	dt_maurer_2d = df_metal_2d[:, :dt_maurer]
	dt_fenz_2d = df_metal_2d[:, :dt_fenz]
	dt_fenz_multi_2d = df_metal_2d[:, :dt_fenz_multi]
	dt_proposed_cuda_2d = df_cuda_2d[:, :dt_proposed_cuda]
	dt_proposed_amdgpu_2d = df_amdgpu_2d[:, :dt_proposed_amdgpu]
	dt_proposed_metal_2d = df_metal_2d[:, :dt_proposed_metal]
	dt_proposed_oneapi_2d = df_oneapi_2d[:, :dt_proposed_oneapi]
	x_names_2d = range_names_2d
	
	dt_heights_2d = zeros(length(dt_names_2d) * length(sizes_2d))
	
	heights_2d = hcat(
		dt_maurer_2d,
		dt_fenz_2d,
		dt_fenz_multi_2d,
		dt_proposed_cuda_2d,
		dt_proposed_amdgpu_2d,
		dt_proposed_metal_2d,
		dt_proposed_oneapi_2d,
	)

	offset_2d = 1
	for i in eachrow(heights_2d)
		dt_heights_2d[offset_2d:(offset_2d+length(i) - 1)] .= i
		offset_2d += 7
	end

	cat_2d = repeat(1:length(sizes_2d), inner = length(dt_names_2d))
	grp_2d = repeat(1:length(dt_names_2d), length(sizes_2d))
	colors = Makie.wong_colors()

	f = Figure(size = (800, 900))
	ax_2d = Axis(
		f[1:2, 1:2],
		ylabel = "Time (ns)",
		title = title_2d,
		titlesize = 25,
		xticks = (1:length(sizes_2d), x_names_2d),
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
		yscale = log10,
		xgridvisible = false,
		ygridvisible = false
	)

	barplot!(
		cat_2d, dt_heights_2d;
		dodge = grp_2d,
		color = colors[grp_2d],
	)
	
	# X axis label
	Label(f[3, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

	### ------------------- 3D PLOT ------------------- ###
	title_3d = "Performance Comparison \nof Julia Distance Transforms (3D)"
	dt_names_3d = dt_names_3d
	sizes_3d = df_metal_3d[:, :sizes_3D]
	dt_maurer_3d = df_metal_3d[:, :dt_maurer_3D]
	dt_fenz_3d = df_metal_3d[:, :dt_fenz_3D]
	dt_fenz_multi_3d = df_metal_3d[:, :dt_fenz_multi_3D]
	dt_proposed_cuda_3d = df_cuda_3d[:, :dt_proposed_cuda_3D]
	dt_proposed_amdgpu_3d = df_amdgpu_3d[:, :dt_proposed_amdgpu_3D]
	dt_proposed_metal_3d = df_metal_3d[:, :dt_proposed_metal_3D]
	dt_proposed_oneapi_3d = df_oneapi_3d[:, :dt_proposed_oneapi_3D]
	x_names_3d = range_names_3d
	
	dt_heights_3d = zeros(length(dt_names_3d) * length(sizes_3d))
	
	heights_3d = hcat(
		dt_maurer_3d,
		dt_fenz_3d,
		dt_fenz_multi_3d,
		dt_proposed_cuda_3d,
		dt_proposed_amdgpu_3d,
		dt_proposed_metal_3d,
		dt_proposed_oneapi_3d,
	)

	offset_3d = 1
	for i in eachrow(heights_3d)
		dt_heights_3d[offset_3d:(offset_3d+length(i) - 1)] .= i
		offset_3d += 7
	end

	cat_3d = repeat(1:length(sizes_3d), inner = length(dt_names_3d))
	grp_3d = repeat(1:length(dt_names_3d), length(sizes_3d))

	ax_3d = Axis(
		f[4:5, 1:2],
		ylabel = "Time (ns)",
		title = title_3d,
		titlesize = 25,
		xticks = (1:length(sizes_3d), x_names_3d),
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
		yscale = log10,
		xgridvisible = false,
		ygridvisible = false
	)

	barplot!(
		cat_3d, dt_heights_3d;
		dodge = grp_3d,
		color = colors[grp_3d],
	)

	# X axis label
	Label(f[6, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

	# CPU Legend
	rnge = 1:3
    labels = dt_names_2d[rnge]
    elements = [PolyElement(polycolor = colors[i]) for i in rnge]
    title = "Distance Transform \nAlgorithms (CPU)"
    Legend(f[2:3, 3], elements, labels, title)

	# GPU Legend
	rnge = 4:7
    labels = dt_names_2d[rnge]
    elements = [PolyElement(polycolor = colors[i]) for i in rnge]
    title = "Distance Transform \nAlgorithms (GPU)"
    Legend(f[3:4, 3], elements, labels, title)

	save(joinpath(pwd(), "plots/julia_distance_transforms.png"), f)
	f
end

# ╔═╡ 3fc76221-c854-46ce-9c85-8baa43ff7e14
md"""
# Python DT Benchmarks
"""

# ╔═╡ 3a436af8-66d5-4a82-85e1-d860fe52421f
df_py_2d = read(datadir("dt_py_2D_CUDA.csv"), DataFrame);

# ╔═╡ 8460f3a8-8c18-46ef-927e-125520db0db6
df_py_3d = read(datadir("dt_py_3D_CUDA.csv"), DataFrame);

# ╔═╡ 124ab9eb-0959-4a16-9659-f58b01ccf463
title_2d_py = "Performance Comparison \nof Python Distance Transforms (2D)"

# ╔═╡ 5c9a9a74-ad6e-43d3-9dce-226f03dc3535
dt_names_2d_py = ["Scipy", "Tensorflow", "FastGeodis", "OpenCV", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)"]

# ╔═╡ 7760006c-a8cd-4019-a54f-a4256d17f39b
title_3d_py = "Performance Comparison \nof Python Distance Transforms (3D)"

# ╔═╡ ea709965-e6dc-43a2-8472-8169fffb8447
dt_names_3d_py = ["Scipy", "Tensorflow", "FastGeodis", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)"]

# ╔═╡ f0927b59-bc77-4979-8d68-f7ac01773a4b
md"""
## Combined Barplot
"""

# ╔═╡ bb5094df-a79d-42cc-a0f0-103735875482
let
    ### ------------------- 2D PLOT ------------------- ###
    title_2d = title_2d_py
    dt_names_2d = dt_names_2d_py
    sizes_2d = df_py_2d[:, :sizes]
    dt_scipy_2d = df_py_2d[:, :dt_scipy]
    dt_tfa_2d = df_py_2d[:, :dt_tfa]
    dt_fastgeodis_2d = df_py_2d[:, :dt_fastgeodis]
    dt_opencv_2d = df_py_2d[:, :dt_opencv]
    dt_pydt_single_2d = df_py_2d[:, :dt_pydt_single]
    dt_pydt_multi_2d = df_py_2d[:, :dt_pydt_multi]
    dt_pydt_cuda_2d = df_py_2d[:, :dt_pydt_cuda]
    x_names_2d = range_names_2d
    
    dt_heights_2d = zeros(length(dt_names_2d) * length(sizes_2d))
    
    heights_2d = hcat(
        dt_scipy_2d,
        dt_tfa_2d,
        dt_opencv_2d,
        dt_pydt_single_2d,
        dt_pydt_multi_2d,
        dt_fastgeodis_2d,
        dt_pydt_cuda_2d
    )

    offset_2d = 1
    for i in eachrow(heights_2d)
        dt_heights_2d[offset_2d:(offset_2d+length(i) - 1)] .= i
        offset_2d += 7
    end

    cat_2d = repeat(1:length(sizes_2d), inner = length(dt_names_2d))
    grp_2d = repeat(1:length(dt_names_2d), length(sizes_2d))
    colors_2d = Makie.wong_colors()

    f = Figure(size = (800, 900))
    ax_2d = Axis(
        f[1:2, 1:2],
        ylabel = "Time (ns)",
        title = title_2d,
        titlesize = 25,
        xticks = (1:length(sizes_2d), x_names_2d),
        yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
        yscale = log10,
        xgridvisible = false,
        ygridvisible = false
    )

    barplot!(
        cat_2d, dt_heights_2d;
        dodge = grp_2d,
        color = colors_2d[grp_2d],
    )

	# X axis label
    Label(f[3, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

	# CPU Legend
	rnge = [1, 2, 3, 4, 5]
	labels = dt_names_2d[[1, 2, 4, 5, 6]]
	elements = [PolyElement(polycolor = colors_2d[i]) for i in rnge]
	title = "Distance Transform \nAlgorithms (CPU)"
	Legend(f[1, 3], elements, labels, title)

	# GPU Legend
	rnge = [6, 7]
	labels = dt_names_2d[[3, 7]]
	elements = [PolyElement(polycolor = colors_2d[i]) for i in rnge]
	title = "Distance Transform \nAlgorithms (GPU)"
	Legend(f[2, 3], elements, labels, title)

    ### ------------------- 3D PLOT ------------------- ###
    title_3d = title_3d_py
    dt_names_3d = dt_names_3d_py
    sizes_3d = df_py_3d[:, :sizes_3D]
    dt_scipy_3d = df_py_3d[:, :dt_scipy_3D]
    dt_tfa_3d = df_py_3d[:, :dt_tfa_3D]
    dt_fastgeodis_3d = df_py_3d[:, :dt_fastgeodis_3D]
    dt_pydt_single_3d = df_py_3d[:, :dt_pydt_single_3D]
    dt_pydt_multi_3d = df_py_3d[:, :dt_pydt_multi_3D]
    dt_pydt_cuda_3d = df_py_3d[:, :dt_pydt_cuda_3D]
    x_names_3d = range_names_3d
    
    dt_heights_3d = zeros(length(dt_names_3d) * length(sizes_3d))
    
    heights_3d = hcat(
        dt_scipy_3d,
        dt_tfa_3d,
        dt_pydt_single_3d,
        dt_pydt_multi_3d,
        dt_fastgeodis_3d,
        dt_pydt_cuda_3d
    )

    offset_3d = 1
    for i in eachrow(heights_3d)
        dt_heights_3d[offset_3d:(offset_3d+length(i) - 1)] .= i
        offset_3d += 6
    end

    cat_3d = repeat(1:length(sizes_3d), inner = length(dt_names_3d))
    grp_3d = repeat(1:length(dt_names_3d), length(sizes_3d))
    colors_3d = Makie.wong_colors()

    ax_3d = Axis(
        f[4:5, 1:2],
        ylabel = "Time (ns)",
        title = title_3d,
        titlesize = 25,
        xticks = (1:length(sizes_3d), x_names_3d),
        yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
        yscale = log10,
        xgridvisible = false,
        ygridvisible = false
    )

    barplot!(
        cat_3d, dt_heights_3d;
        dodge = grp_3d,
        color = colors_3d[grp_3d],
    )

    # X axis label
    Label(f[6, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

	# CPU Legend
	rnge = [1, 2, 3, 4]
	labels = dt_names_3d[[1, 2, 4, 5]]
	elements = [PolyElement(polycolor = colors_3d[i]) for i in rnge]
	title = "Distance Transform \nAlgorithms (CPU)"
	Legend(f[4, 3], elements, labels, title)
	
	# GPU Legend
	rnge = [5, 6]
	labels = dt_names_3d[[3, 6]]
	elements = [PolyElement(polycolor = colors_3d[i]) for i in rnge]
	title = "Distance Transform \nAlgorithms (GPU)"
	Legend(f[5, 3], elements, labels, title)

	save(joinpath(pwd(), "plots/python_distance_transforms.png"), f)

    f
end

# ╔═╡ c47bd4a9-368e-4288-b0df-9f116574a6b0
md"""
# Hausdorff Loss
"""

# ╔═╡ 1523ddd9-ebe4-4a48-aaaf-0b499fef7b34
df_hd_loss_pure_losses_timings = read(joinpath(pwd(), "data/hd_loss_pure_losses_timings.csv"), DataFrame);

# ╔═╡ 3a26807e-b902-4a02-9af9-f83db03dfe00
df_hd_loss_plain_dice_timing = read(joinpath(pwd(), "data/hd_loss_plain_dice_timing.csv"), DataFrame);

# ╔═╡ d726c7e3-2005-4c2f-a0cc-5d774d1dce88
df_hd_loss_hd_dice_scipy_timing = read(joinpath(pwd(), "data/hd_loss_hd_dice_scipy_timing.csv"), DataFrame);

# ╔═╡ 447c1bad-2a30-4870-9330-e243922f0ec4
df_hd_loss_hd_dice_pydt_timing = read(joinpath(pwd(), "data/hd_loss_hd_dice_pydt_timing.csv"), DataFrame);

# ╔═╡ c49b579f-63f8-430d-8938-cb2ecf35f11c
md"""
## Combined Barplot
"""

# ╔═╡ bfe42c1b-f167-49f0-b05a-fb13e531fb53
let
	df = df_hd_loss_pure_losses_timings
	methods = ["Dice Loss", "HD Loss (Scipy)", "HD Loss (Proposed)"]	
	min_times = df[:, "Minimum Time (s)"]
	std_devs = df[:, "Standard Deviation (s)"]
	
	# Create the barplot
	fig = Figure(size = (800, 800))
	ax = Axis(
		fig[1, 1],
		ylabel = "Time (s)",
		title = "Pure Loss Function Timings",
		xticks = (1:length(methods), methods),
		yticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
		yscale = log10,
		ytickformat = "{:.2e}",  # Format y-axis tick labels as scientific notation with 2 decimal places
		xgridvisible = false,
		ygridvisible = false
	)
	
	colors = [:turquoise3, :mediumorchid3, :mediumseagreen]
	barplot!(
		1:length(methods), min_times;
		color = colors,
		bar_labels = string.(round.(min_times; sigdigits = 3))
	)
	ylims!(ax; high=1e2)

	df1 = df_hd_loss_plain_dice_timing
	df2 = df_hd_loss_hd_dice_scipy_timing
	df3 = df_hd_loss_hd_dice_pydt_timing
	
	methods = ["Dice Loss", "Dice + HD Loss (Scipy)", "Dice + HD Loss (Proposed)"]	
	min_times = [
		df1[:, "Avg Epoch Time (s)"]...,
		df2[:, "Avg Epoch Time (s)"]...,
		df3[:, "Avg Epoch Time (s)"]...
	]
	std_devs = [
		df1[:, "Std Epoch Time (s)"]...,
		df2[:, "Std Epoch Time (s)"]...,
		df3[:, "Std Epoch Time (s)"]...
	]
	
	ax = Axis(
		fig[2, 1],
		ylabel = "Time (s)",
		title = "Average Epoch Timings",
		xticks = (1:length(methods), methods),
		yticks = collect(0:10:50),
		xgridvisible = false,
		ygridvisible = false
	)
	
	barplot!(
		1:length(methods), min_times;
		color = colors,
		bar_labels = string.(round.(min_times; sigdigits = 3))
	)
	ylims!(ax; high=50)

	save(joinpath(pwd(), "plots/hd_loss_timings.png"), fig)
	# Adjust the layout and display the plot
	fig
end

# ╔═╡ 2b054f68-d0c5-4c79-b200-576b1e7c02ee
md"""
## Training/Accuracy Metrics
"""

# ╔═╡ 76b82ae6-38b4-40eb-97a6-286165398e17
df_training_results_dice = read(joinpath(pwd(), "data/training_results_dice.csv"), DataFrame);

# ╔═╡ 6432995d-e296-4cba-810b-b8990cc18f3b
df_training_results_hd_pydt = read(joinpath(pwd(), "data/training_results_hd_pydt.csv"), DataFrame);

# ╔═╡ 7136ad27-fd70-48bf-a7f1-0874c031aa50
df_hd_loss_metrics_dice = read(joinpath(pwd(), "data/hd_loss_metrics_dice.csv"), DataFrame);

# ╔═╡ 3cd22291-ee6a-4032-b05e-36fedb87beac
df_hd_loss_metrics_hd_pydt = read(joinpath(pwd(), "data/hd_loss_metrics_hd_pydt.csv"), DataFrame);

# ╔═╡ 20056bac-bc82-4b28-ad2f-705bd2c4366d
begin
	df_dice = df_hd_loss_metrics_dice
	df_hd_dice = df_hd_loss_metrics_hd_pydt
	
	metrics = ["Dice Score", "IoU Score", "Hausdorff Distance", "95 Percentile Hausdorff Distance", "Total Training Time (s)"]
	
	# Extract the metric values for each model
	dice_values = [df_dice[1, metric] for metric in metrics]
	hd_dice_values = [df_hd_dice[1, metric] for metric in metrics]
end;

# ╔═╡ 4c5b64c2-e979-473f-b10b-071c78e53e93
# Create a DataFrame with the metrics for each model
df_metrics = DataFrame(
	"Metric" => metrics,
	"Dice Loss" => dice_values,
	"HD + Dice Loss" => hd_dice_values
)

# ╔═╡ b928a575-ec2c-4ac2-8bfc-6e646a9bcd99
md"""
# Skeletonization
"""

# ╔═╡ e26dfbb8-75ac-49a3-a3ce-a073fa5d1ef0
df_skeleton = read(datadir("skeleton.csv"), DataFrame);

# ╔═╡ f6c3bcc3-ddca-43e6-8910-38cf547a7596
let
	sizes = df_skeleton[:, :sizes]
	cpu_timings = df_skeleton[:, "cpu timings"]
	gpu_timings = df_skeleton[:, "gpu timings"]
	
	f = Figure()
	ax = Axis(
		f[1, 1],
		ylabel = "Time (ns)",
		title = "Skeletonization Timings",
		yscale = log10,
		xticks = (1:length(sizes), string.(sizes)),
		xlabel = "Array Sizes (Pixels)"
	)
	scatterlines!(cpu_timings; label = "CPU")
	scatterlines!(gpu_timings; label = "Proposed (Metal)")

	axislegend(ax; position = :rb)

	save(joinpath(pwd(), "plots/skeletonization.png"), f)
	
	f
end

# ╔═╡ Cell order:
# ╠═2c729da6-40e6-47cd-a14d-c152b8789b17
# ╠═30f67101-9626-4d01-a6fd-c260cd5c29b6
# ╠═33e02405-1750-48f9-9776-d1d2d261f63f
# ╠═a968bcd8-fc42-45ec-af7c-68e73e8f1cd5
# ╠═50e24ebe-403a-4d89-b02f-7a1577222838
# ╠═50bfb09f-4dbb-4488-9284-7eef837ffe75
# ╠═d1a12515-a9d0-468b-8978-dbb26a1ee667
# ╠═e39675a9-08c7-4a4a-8eba-021862757a40
# ╠═278dfa0e-46e1-4789-9f51-eb3463a9fb00
# ╟─8b786cbc-dd81-4208-9eee-2d7f7bbfa23f
# ╠═ad97f6cb-c331-4898-9c6c-485582058e4d
# ╠═83f4fd58-e801-4dda-9ba7-f5eec56722f6
# ╠═c7c6aa70-6e46-4444-b8df-68895b55d642
# ╠═d86c512c-b3dc-4542-8c2c-27b72019dce1
# ╠═eb190959-b90f-4dbb-8ae7-09b964e1a1c2
# ╠═1936dff5-1d17-4773-9009-51ec95eb9411
# ╠═2ff50a99-aaf0-4282-a194-6fff6f50dea6
# ╠═facdc420-5c39-4057-853e-bbab8f96fac6
# ╠═492df5fa-e20e-4dcb-8c1f-b7e14d9fc2de
# ╠═7bc02cb0-76e9-4654-b17a-9d95089bf472
# ╠═b50a4061-4f49-4578-8671-1746d532c9dc
# ╠═08676d5b-f098-43a9-8bc3-b5cda3282b2a
# ╠═f093102d-4796-4d05-943c-c314febe7342
# ╠═0c09ef6c-d05e-4f73-9075-78d9ba986bb9
# ╟─335fe4b9-a11f-4cb9-ac81-68d305f73a2d
# ╟─bad637b7-3449-4481-846f-e5160cdfca40
# ╟─3fc76221-c854-46ce-9c85-8baa43ff7e14
# ╠═3a436af8-66d5-4a82-85e1-d860fe52421f
# ╠═8460f3a8-8c18-46ef-927e-125520db0db6
# ╠═124ab9eb-0959-4a16-9659-f58b01ccf463
# ╠═5c9a9a74-ad6e-43d3-9dce-226f03dc3535
# ╠═7760006c-a8cd-4019-a54f-a4256d17f39b
# ╠═ea709965-e6dc-43a2-8472-8169fffb8447
# ╟─f0927b59-bc77-4979-8d68-f7ac01773a4b
# ╟─bb5094df-a79d-42cc-a0f0-103735875482
# ╟─c47bd4a9-368e-4288-b0df-9f116574a6b0
# ╠═1523ddd9-ebe4-4a48-aaaf-0b499fef7b34
# ╠═3a26807e-b902-4a02-9af9-f83db03dfe00
# ╠═d726c7e3-2005-4c2f-a0cc-5d774d1dce88
# ╠═447c1bad-2a30-4870-9330-e243922f0ec4
# ╟─c49b579f-63f8-430d-8938-cb2ecf35f11c
# ╟─bfe42c1b-f167-49f0-b05a-fb13e531fb53
# ╟─2b054f68-d0c5-4c79-b200-576b1e7c02ee
# ╠═76b82ae6-38b4-40eb-97a6-286165398e17
# ╠═6432995d-e296-4cba-810b-b8990cc18f3b
# ╠═7136ad27-fd70-48bf-a7f1-0874c031aa50
# ╠═3cd22291-ee6a-4032-b05e-36fedb87beac
# ╠═20056bac-bc82-4b28-ad2f-705bd2c4366d
# ╠═4c5b64c2-e979-473f-b10b-071c78e53e93
# ╟─b928a575-ec2c-4ac2-8bfc-6e646a9bcd99
# ╠═e26dfbb8-75ac-49a3-a3ce-a073fa5d1ef0
# ╟─f6c3bcc3-ddca-43e6-8910-38cf547a7596
