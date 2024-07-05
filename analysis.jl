### A Pluto.jl notebook ###
# v0.19.40

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

# ╔═╡ eb190959-b90f-4dbb-8ae7-09b964e1a1c2
df_metal_3d = read(datadir("dt_3D_Metal.csv"), DataFrame);

# ╔═╡ 1936dff5-1d17-4773-9009-51ec95eb9411
df_cuda_3d = read(datadir("dt_3D_CUDA.csv"), DataFrame);

# ╔═╡ 3175900c-7676-4ab2-b31a-a0ca3b1afde7
function create_barplot(
	title, dt_names;
	sizes = [],
	dt_maurer = [],
	dt_fenz = [],
	dt_fenz_multi = [],
	dt_proposed_cuda = [],
	dt_proposed_metal = [],
	dt_proposed_amdgpu = [],
	x_names = sizes
	)
    dt_heights = zeros(length(dt_names) * length(sizes))
    
    heights = hcat(
        dt_maurer,
        dt_fenz,
        dt_fenz_multi,
        isempty(dt_proposed_cuda) ? zeros(length(dt_maurer)) : dt_proposed_cuda,
        isempty(dt_proposed_metal) ? zeros(length(dt_maurer)) : dt_proposed_metal,
        isempty(dt_proposed_amdgpu) ? zeros(length(dt_maurer)) : dt_proposed_amdgpu
    )

    offset = 1
    for i in eachrow(heights)
        dt_heights[offset:(offset+length(i) - 1)] .= i
        offset += 6
    end

    cat = repeat(1:length(sizes), inner = length(dt_names))
    grp = repeat(1:length(dt_names), length(sizes))
    colors = Makie.wong_colors()

    f = Figure(size = (800, 600))
    ax = Axis(
        f[1, 1:2],
        ylabel = "Time (ns)",
        title = title,
		titlesize = 25,
        xticks = (1:length(sizes), x_names),
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
        yscale = log10
    )

    barplot!(
        cat, dt_heights;
        dodge = grp,
        color = colors[grp],
    )

    # X axis label
    Label(f[2, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

    # Legend
    labels = dt_names
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    title = "Distance Transforms"

    Legend(f[1, 3], elements, labels, title)
    return f
end

# ╔═╡ 97a7f98f-738d-4870-8872-4d0e7ba88c4a
md"""
## 2D
"""

# ╔═╡ 492df5fa-e20e-4dcb-8c1f-b7e14d9fc2de
title_2d = "Performance Comparison \nof Julia Distance Transforms (2D)"

# ╔═╡ 7bc02cb0-76e9-4654-b17a-9d95089bf472
dt_names_2d = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (Metal)", "Proposed (AMDGPU)"]

# ╔═╡ b50a4061-4f49-4578-8671-1746d532c9dc
range_names_2d = [L"(2^3)^2", L"(2^4)^2", L"(2^5)^2", L"(2^6)^2", L"(2^7)^2", L"(2^8)^2", L"(2^9)^2", L"(2^{10})^2", L"(2^{11})^2", L"(2^{12})^2"]

# ╔═╡ 6de86d5d-753c-478e-bf9a-0b7c111192bb
dt_fig_2d = create_barplot(
	title_2d, dt_names_2d;
	sizes = df_metal_2d[:, :sizes],
	dt_maurer = df_metal_2d[:, :dt_maurer],
	dt_fenz = df_metal_2d[:, :dt_fenz],
	dt_fenz_multi = df_metal_2d[:, :dt_fenz_multi],
	dt_proposed_cuda = df_cuda_2d[:, :dt_proposed_cuda],
	dt_proposed_metal = df_metal_2d[:, :dt_proposed_metal],
	x_names = range_names_2d
)

# ╔═╡ 1510781f-cf67-4cd1-852f-4200d4a437a8
save(joinpath(pwd(), "plots", "julia_benchmarks_2d.png"), dt_fig_2d);

# ╔═╡ 54b95ed5-04e1-401d-b240-5a561b2e6713
md"""
## 3D
"""

# ╔═╡ 08676d5b-f098-43a9-8bc3-b5cda3282b2a
title_3d = "Performance Comparison \nof Julia Distance Transforms (3D)"

# ╔═╡ f093102d-4796-4d05-943c-c314febe7342
dt_names_3d = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (Metal)", "Proposed (AMDGPU)"]

# ╔═╡ 0c09ef6c-d05e-4f73-9075-78d9ba986bb9
range_names_3d = [L"(2^0)^3", L"(2^1)^3", L"(2^2)^3", L"(2^3)^3", L"(2^4)^3", L"(2^5)^3", L"(2^6)^3", L"(2^7)^3", L"(2^8)^3"]

# ╔═╡ 7b4e4368-8903-4279-9c16-32cf8b603949
dt_fig_3d = create_barplot(
	title_3d, dt_names_3d;
	sizes = df_metal_3d[:, :sizes_3D],
	dt_maurer = df_metal_3d[:, :dt_maurer_3D],
	dt_fenz = df_metal_3d[:, :dt_fenz_3D],
	dt_fenz_multi = df_metal_3d[:, :dt_fenz_multi_3D],
	dt_proposed_cuda = df_cuda_3d[:, :dt_proposed_cuda_3D],
	dt_proposed_metal = df_metal_3d[:, :dt_proposed_metal_3D],
	x_names = range_names_3d
)

# ╔═╡ b69b8849-2dc7-4eaa-b58e-ae1525e18b80
save(joinpath(pwd(), "plots", "julia_benchmarks_3d.png"), dt_fig_3d);

# ╔═╡ 3fc76221-c854-46ce-9c85-8baa43ff7e14
md"""
# Python DT Benchmarks
"""

# ╔═╡ 3a436af8-66d5-4a82-85e1-d860fe52421f
df_py_2d = read(datadir("dt_py_2D_CUDA.csv"), DataFrame);

# ╔═╡ 8460f3a8-8c18-46ef-927e-125520db0db6
df_py_3d = read(datadir("dt_py_3D_CUDA.csv"), DataFrame);

# ╔═╡ 79530ee6-42bc-4bf4-9e9c-68a14556a7f1
df_py_2d

# ╔═╡ 65d76d1e-6caf-4ad5-acdb-bed143711ee7
md"""
## 2D
"""

# ╔═╡ 1a75f51d-e66c-45d1-8f0a-b72cacbe263e
function create_barplot_py_2d(
	title, dt_names;
	sizes = [],
	dt_scipy = [],
	dt_tfa = [],
	dt_fastgeodis = [],
	dt_opencv = [],
	dt_pydt_single = [],
	dt_pydt_multi = [],
	dt_pydt_cuda = [],
	x_names = sizes
	)
    dt_heights = zeros(length(dt_names) * length(sizes))
    
    heights = hcat(
		dt_scipy,
        dt_tfa,
        dt_fastgeodis,
		dt_opencv,
		dt_pydt_single,
        dt_pydt_multi,
		dt_pydt_cuda
	)

    offset = 1
    for i in eachrow(heights)
        dt_heights[offset:(offset+length(i) - 1)] .= i
        offset += 7
    end

    cat = repeat(1:length(sizes), inner = length(dt_names))
    grp = repeat(1:length(dt_names), length(sizes))
    colors = Makie.wong_colors()

    f = Figure(size = (800, 600))
    ax = Axis(
        f[1, 1:2],
        ylabel = "Time (ns)",
        title = title,
		titlesize = 25,
        xticks = (1:length(sizes), x_names),
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
        yscale = log10
    )

    barplot!(
        cat, dt_heights;
        dodge = grp,
        color = colors[grp],
    )

    # X axis label
    Label(f[2, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

    # Legend
    labels = dt_names
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    title = "Distance Transforms"

    Legend(f[1, 3], elements, labels, title)
    return f
end

# ╔═╡ 124ab9eb-0959-4a16-9659-f58b01ccf463
title_2d_py = "Performance Comparison \nof Python Distance Transforms (2D)"

# ╔═╡ 5c9a9a74-ad6e-43d3-9dce-226f03dc3535
dt_names_2d_py = ["Scipy", "Tensorflow", "FastGeodis", "OpenCV", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)"]

# ╔═╡ d98d995b-09b6-4f3a-b7a7-4bb9effbaf7d
dt_fig_2d_py = create_barplot_py_2d(
	title_2d_py, dt_names_2d_py;
	sizes = df_py_2d[:, :sizes],
	dt_scipy = df_py_2d[:, :dt_scipy],
	dt_tfa = df_py_2d[:, :dt_tfa],
	dt_fastgeodis = df_py_2d[:, :dt_fastgeodis],
	dt_opencv = df_py_2d[:, :dt_opencv],
	dt_pydt_single = df_py_2d[:, :dt_pydt_single],
	dt_pydt_multi = df_py_2d[:, :dt_pydt_multi],
	dt_pydt_cuda = df_py_2d[:, :dt_pydt_cuda],
	x_names = range_names_2d
)

# ╔═╡ 0caab0a0-667f-4862-9e1d-d20e331032fd
save(joinpath(pwd(), "plots", "py_benchmarks_2d.png"), dt_fig_2d_py);

# ╔═╡ 971ab05c-a849-4f44-a484-fec73f8f2326
md"""
## 3D
"""

# ╔═╡ e93f3bd7-9f84-435c-bc57-2aeee49c08f2
function create_barplot_py_3d(
	title, dt_names;
	sizes = [],
	dt_scipy = [],
	dt_tfa = [],
	dt_fastgeodis = [],
	dt_pydt_single = [],
	dt_pydt_multi = [],
	dt_pydt_cuda = [],
	x_names = sizes
	)
    dt_heights = zeros(length(dt_names) * length(sizes))
    
    heights = hcat(
		dt_scipy,
        dt_tfa,
        dt_fastgeodis,
		dt_pydt_single,
        dt_pydt_multi,
		dt_pydt_cuda
	)

    offset = 1
    for i in eachrow(heights)
        dt_heights[offset:(offset+length(i) - 1)] .= i
        offset += 6
    end

    cat = repeat(1:length(sizes), inner = length(dt_names))
    grp = repeat(1:length(dt_names), length(sizes))
    colors = Makie.wong_colors()

    f = Figure(size = (800, 600))
    ax = Axis(
        f[1, 1:2],
        ylabel = "Time (ns)",
        title = title,
		titlesize = 25,
        xticks = (1:length(sizes), x_names),
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
        yscale = log10
    )

    barplot!(
        cat, dt_heights;
        dodge = grp,
        color = colors[grp],
    )

    # X axis label
    Label(f[2, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

    # Legend
    labels = dt_names
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    title = "Distance Transforms"

    Legend(f[1, 3], elements, labels, title)
    return f
end

# ╔═╡ 7760006c-a8cd-4019-a54f-a4256d17f39b
title_3d_py = "Performance Comparison \nof Python Distance Transforms (3D)"

# ╔═╡ ea709965-e6dc-43a2-8472-8169fffb8447
dt_names_3d_py = ["Scipy", "Tensorflow", "FastGeodis", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)"]

# ╔═╡ 520a1602-1d22-4ec5-9410-01124c98feb9
dt_fig_3d_py = create_barplot_py_3d(
	title_3d_py, dt_names_3d_py;
	sizes = df_py_3d[:, :sizes_3D],
	dt_scipy = df_py_3d[:, :dt_scipy_3D],
	dt_tfa = df_py_3d[:, :dt_tfa_3D],
	dt_fastgeodis = df_py_3d[:, :dt_fastgeodis_3D],
	dt_pydt_single = df_py_3d[:, :dt_pydt_single_3D],
	dt_pydt_multi = df_py_3d[:, :dt_pydt_multi_3D],
	dt_pydt_cuda = df_py_3d[:, :dt_pydt_cuda_3D],
	x_names = range_names_3d
)

# ╔═╡ 37087a33-18d1-4e4d-b7e0-229b3b4aa797
save(joinpath(pwd(), "plots", "py_benchmarks_3d.png"), dt_fig_3d_py);

# ╔═╡ c47bd4a9-368e-4288-b0df-9f116574a6b0
md"""
# Hausdorff Loss
"""

# ╔═╡ fbd27c0e-3175-4a59-97f1-89e41fb4afba
md"""
## Pure Loss Timings
"""

# ╔═╡ 1523ddd9-ebe4-4a48-aaaf-0b499fef7b34
df_hd_loss_pure_losses_timings = read(joinpath(pwd(), "data/hd_loss_pure_losses_timings.csv"), DataFrame);

# ╔═╡ 91be7704-515e-4427-88d6-ff4f1530c3f1
let
	df = df_hd_loss_pure_losses_timings
	methods = ["Pure Dice Loss", "HD Loss \n(Scipy)", "HD Loss \n(Proposed)"]	
	min_times = df[:, "Minimum Time (s)"]
	std_devs = df[:, "Standard Deviation (s)"]
	
	# Create the barplot
	fig = Figure(size = (800, 600))
	ax = Axis(
		fig[1, 1],
		ylabel = "Time (s)",
		title = "Pure Loss Function Timings",
		xticks = (1:length(methods), methods),
		yticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
		yscale = log10
	)
	
	colors = [:red, :blue, :green]
	barplot!(1:length(methods), min_times, color = colors)
	
	# Adjust the layout and display the plot
	fig[1, 1] = ax
	fig
end

# ╔═╡ 51825e1d-5eac-40ed-abba-2c1d4dbbc917
md"""
## Training Loop Timings
"""

# ╔═╡ 3a26807e-b902-4a02-9af9-f83db03dfe00
df_hd_loss_plain_dice_timing = read(joinpath(pwd(), "data/hd_loss_plain_dice_timing.csv"), DataFrame);

# ╔═╡ d726c7e3-2005-4c2f-a0cc-5d774d1dce88
df_hd_loss_hd_dice_scipy_timing = read(joinpath(pwd(), "data/hd_loss_hd_dice_scipy_timing.csv"), DataFrame);

# ╔═╡ 447c1bad-2a30-4870-9330-e243922f0ec4
df_hd_loss_hd_dice_pydt_timing = read(joinpath(pwd(), "data/hd_loss_hd_dice_pydt_timing.csv"), DataFrame);

# ╔═╡ a0650b6e-2048-4e84-8143-2c0cdcca326d
let
	df1 = df_hd_loss_plain_dice_timing
	df2 = df_hd_loss_hd_dice_scipy_timing
	df3 = df_hd_loss_hd_dice_pydt_timing
	
	methods = ["Dice Loss", "Dice + HD Loss (Scipy)", "Dice + HD Loss (Proposed)"]	
	min_times = [
		df1[:, "Avg Epoch Time (s)"]...,
		df2[:, "Avg Epoch Time (s)"]...,
		df1[:, "Avg Epoch Time (s)"]...
	]
	std_devs = [
		df1[:, "Std Epoch Time (s)"]...,
		df2[:, "Std Epoch Time (s)"]...,
		df3[:, "Std Epoch Time (s)"]...
	]
	
	# Create the barplot
	fig = Figure(size = (800, 600))
	ax = Axis(
		fig[1, 1],
		ylabel = "Time (s)",
		title = "Average Epoch Time",
		xticks = (1:length(methods), methods),
		yticks = collect(0:10:50)
	)
	
	colors = [:red, :blue, :green]
	barplot!(1:length(methods), min_times, color = colors)
	
	# Adjust the layout and display the plot
	fig[1, 1] = ax
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

# ╔═╡ 32ea92f0-92ef-4a87-bc9b-df52b910a28e
let
	df1 = df_training_results_dice
	df2 = df_training_results_hd_pydt
	
	f = Figure()
	ax1 = Axis(
		f[1:2, 1:4],
		title = "Dice Loss"
	)
	lines!(ax1, df1[:, "Training Loss"], label = "Training Loss")
	lines!(ax1, df1[:, "Validation Loss"], label = "Validation Loss")
	lines!(ax1, df1[:, "Validation Dice Score"], label = "Validation Dice Score")
	ylims!(ax1, low = 0, high = 1.1)
	
	ax2 = Axis(
		f[3:4, 1:4],
		title = "Dice + HD Loss (Proposed)"
	)
	lines!(ax2, df2[:, "Training Loss"], label = "Training Loss")
	lines!(ax2, df2[:, "Validation Loss"], label = "Validation Loss")
	lines!(ax2, df2[:, "Validation Dice Score"], label = "Validation Dice Score")
	ylims!(ax2, low = 0, high = 1.1)

	# Create a combined legend
	Legend(f[2:3, 5:6], ax1)

	f
end


# ╔═╡ 7136ad27-fd70-48bf-a7f1-0874c031aa50
df_hd_loss_metrics_dice = read(joinpath(pwd(), "data/hd_loss_metrics_dice.csv"), DataFrame);

# ╔═╡ 3cd22291-ee6a-4032-b05e-36fedb87beac
df_hd_loss_metrics_hd_pydt = read(joinpath(pwd(), "data/hd_loss_metrics_hd_pydt.csv"), DataFrame);

# ╔═╡ 7b2f0282-db75-4d3c-a832-de19486ddfd8
df_hd_loss_metrics_dice

# ╔═╡ 8f02a8b5-561f-44fc-b954-43f936fa41fc
df_hd_loss_metrics_hd_pydt

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
# ╠═eb190959-b90f-4dbb-8ae7-09b964e1a1c2
# ╠═1936dff5-1d17-4773-9009-51ec95eb9411
# ╟─3175900c-7676-4ab2-b31a-a0ca3b1afde7
# ╟─97a7f98f-738d-4870-8872-4d0e7ba88c4a
# ╠═492df5fa-e20e-4dcb-8c1f-b7e14d9fc2de
# ╠═7bc02cb0-76e9-4654-b17a-9d95089bf472
# ╠═b50a4061-4f49-4578-8671-1746d532c9dc
# ╟─6de86d5d-753c-478e-bf9a-0b7c111192bb
# ╠═1510781f-cf67-4cd1-852f-4200d4a437a8
# ╟─54b95ed5-04e1-401d-b240-5a561b2e6713
# ╠═08676d5b-f098-43a9-8bc3-b5cda3282b2a
# ╠═f093102d-4796-4d05-943c-c314febe7342
# ╠═0c09ef6c-d05e-4f73-9075-78d9ba986bb9
# ╟─7b4e4368-8903-4279-9c16-32cf8b603949
# ╠═b69b8849-2dc7-4eaa-b58e-ae1525e18b80
# ╟─3fc76221-c854-46ce-9c85-8baa43ff7e14
# ╠═3a436af8-66d5-4a82-85e1-d860fe52421f
# ╠═8460f3a8-8c18-46ef-927e-125520db0db6
# ╠═79530ee6-42bc-4bf4-9e9c-68a14556a7f1
# ╟─65d76d1e-6caf-4ad5-acdb-bed143711ee7
# ╟─1a75f51d-e66c-45d1-8f0a-b72cacbe263e
# ╠═124ab9eb-0959-4a16-9659-f58b01ccf463
# ╠═5c9a9a74-ad6e-43d3-9dce-226f03dc3535
# ╟─d98d995b-09b6-4f3a-b7a7-4bb9effbaf7d
# ╠═0caab0a0-667f-4862-9e1d-d20e331032fd
# ╟─971ab05c-a849-4f44-a484-fec73f8f2326
# ╠═e93f3bd7-9f84-435c-bc57-2aeee49c08f2
# ╠═7760006c-a8cd-4019-a54f-a4256d17f39b
# ╠═ea709965-e6dc-43a2-8472-8169fffb8447
# ╟─520a1602-1d22-4ec5-9410-01124c98feb9
# ╠═37087a33-18d1-4e4d-b7e0-229b3b4aa797
# ╟─c47bd4a9-368e-4288-b0df-9f116574a6b0
# ╟─fbd27c0e-3175-4a59-97f1-89e41fb4afba
# ╠═1523ddd9-ebe4-4a48-aaaf-0b499fef7b34
# ╟─91be7704-515e-4427-88d6-ff4f1530c3f1
# ╟─51825e1d-5eac-40ed-abba-2c1d4dbbc917
# ╠═3a26807e-b902-4a02-9af9-f83db03dfe00
# ╠═d726c7e3-2005-4c2f-a0cc-5d774d1dce88
# ╠═447c1bad-2a30-4870-9330-e243922f0ec4
# ╟─a0650b6e-2048-4e84-8143-2c0cdcca326d
# ╟─2b054f68-d0c5-4c79-b200-576b1e7c02ee
# ╠═76b82ae6-38b4-40eb-97a6-286165398e17
# ╠═6432995d-e296-4cba-810b-b8990cc18f3b
# ╟─32ea92f0-92ef-4a87-bc9b-df52b910a28e
# ╠═7136ad27-fd70-48bf-a7f1-0874c031aa50
# ╠═3cd22291-ee6a-4032-b05e-36fedb87beac
# ╠═7b2f0282-db75-4d3c-a832-de19486ddfd8
# ╠═8f02a8b5-561f-44fc-b954-43f936fa41fc
