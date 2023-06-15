### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 5463443f-a69e-4766-bfc2-0b7ca0fd48c9
# ╠═╡ show_logs = false
using DrWatson; 

# ╔═╡ 6495cb7b-67ec-49da-bc5d-f7a63ca9e5e9
# ╠═╡ show_logs = false
@quickactivate "hd-loss"

# ╔═╡ ec608b5e-a6d5-4b12-a3a4-cc1738dbc6ed
# ╠═╡ show_logs = false
using PlutoUI, Statistics, CSV, DataFrames, GLM, CairoMakie, HypothesisTests, Colors, MLJBase, Glob, Flux, NIfTI, Images, ImageMorphology, FastAI, FastVision, StaticArrays, MLDataPattern, Printf, CSVFiles, StatsBase

# ╔═╡ d1a12515-a9d0-468b-8978-dbb26a1ee667
using CairoMakie: Axis, Label

# ╔═╡ f88c7074-8b5c-4dc7-83d5-e65a24b88ab0
include(srcdir("load_model.jl")), include(srcdir("postprocessing.jl")); 

# ╔═╡ f81ac05c-b5f2-4e96-b370-adc3b073d5c9
include(srcdir("load_model.jl"));

# ╔═╡ 20114ef8-cf6b-4a92-89b8-40a85571d6c2
include(srcdir("postprocessing.jl"))

# ╔═╡ 278dfa0e-46e1-4789-9f51-eb3463a9fb00
TableOfContents()

# ╔═╡ 8608a972-73c0-4903-bdd7-9a23f7c57337
medphys_theme = Theme(
    Axis = (
        backgroundcolor = :white,
		xgridcolor = :gray,
		xgridwidth = 0.1,
		xlabelsize = 20,
		xticklabelsize = 20,
		ygridcolor = :gray,
		ygridwidth = 0.1,
		ylabelsize = 20,
		yticklabelsize = 20,
		bottomsplinecolor = :black,
		leftspinecolor = :black,
		titlesize = 30
	)
);

# ╔═╡ 02f182d4-f5ab-46a3-8009-33ed641fcf27
begin
	path_julia = datadir("dt-loss")
	path_py = datadir("dt-loss-py")
end

# ╔═╡ 95a525fe-e10b-42ee-a1b9-9405fe16b50c
md"""
# Distance Transforms
"""

# ╔═╡ 0960ac79-d976-42a7-aaac-4d07ec94927e
df_dt = CSV.read(datadir("results", "dts.csv"), DataFrame);

# ╔═╡ c1659832-9e1a-4af5-9bf6-fd4d6f12589f
function dt()
    f = Figure()

    ##-- A --##
    ax = Axis(
		f[1, 1],
		xlabel = "Array Size (Pixels)",
		ylabel = "Time (s)",
		title = "2D Distance Transform"
	)
    lines!(
		df_dt[!, :sizes], 
		df_dt[!, :dt_scipy] * 1e-9, 
		label=L"\text{DT}_{Scipy}"
	)
	lines!(
		df_dt[!, :sizes], 
		df_dt[!, :dt_fenz] * 1e-9, 
		label=L"\text{DT}_{Felzenszwalb}"
	)
	lines!(
		df_dt[!, :sizes], 
		df_dt[!, :dt_fenz_gpu] * 1e-9, 
		label=L"\text{DT}_{FelzenszwalbGPU}"
	)

	##-- B --##
    ax = Axis(
		f[2, 1],
		xlabel = "Array Size (Pixels)",
		ylabel = "Time (s)",
		title = "3D Distance Transform"
	)
    lines!(
		df_dt[!, :sizes_3D], 
		df_dt[!, :dt_scipy_3D] * 1e-9, 
		label=L"\text{DT}_{Scipy}"
	)
	lines!(
		df_dt[!, :sizes_3D], 
		df_dt[!, :dt_fenz_3D] * 1e-9, 
		label=L"\text{DT}_{Felzenszwalb}"
	)
	lines!(
		df_dt[!, :sizes_3D], 
		df_dt[!, :dt_fenz_gpu_3D] * 1e-9, 
		label=L"\text{DT}_{FelzenszwalbGPU}"
	)

    ##-- LABELS --##
    f[1:2, 2] = Legend(f, ax; framevisible=false)
    for (label, layout) in zip(["A", "B"], [f[1, 1], f[2, 1]])
        Label(
            layout[1, 1, TopLeft()],
            label;
            padding=(0, 0, 40, 0),
            halign=:right,
        )
    end

	save(plotsdir("dt.png"), f)
	
    return f
end

# ╔═╡ aa312920-1c67-46f7-a7b1-dfe42f54c769
with_theme(dt, medphys_theme)

# ╔═╡ 6349d56d-f837-4395-9819-e8e7f34ba01f
(x, y, z) = df_dt[end, :dt_scipy], df_dt[end, :dt_fenz], df_dt[end, :dt_fenz_gpu]

# ╔═╡ a0ea7e69-1249-4d47-9366-ec9f44568236
gpu_vs_scipy = x / z

# ╔═╡ 5059495d-6905-4043-b03e-3ea73f67d341
gpu_vs_cpu = x / y

# ╔═╡ 9e02ae06-a2cd-427c-8e94-205bd8fbc0f3
(x_3D, y_3D, z_3D) = df_dt[end, :dt_scipy_3D], df_dt[end, :dt_fenz_3D], df_dt[end, :dt_fenz_gpu_3D]

# ╔═╡ f2faa757-6a21-40ba-9f36-d3149a37a5fc
gpu_vs_scipy_3D = x_3D / z_3D

# ╔═╡ 609c529b-8dd9-4daa-9cd8-b89d7a4ce792
gpu_vs_cpu_3D = x_3D / y_3D

# ╔═╡ 1852f60f-cdf1-47d5-8d97-052710bed435
md"""
# Losses
"""

# ╔═╡ fbd61bd3-2181-4d24-a417-85bc56a9b51e
df_loss = CSV.read(datadir("results", "losses.csv"), DataFrame);

# ╔═╡ b7236b7b-cf97-4962-b49d-58037b52201d
function loss()
    f = Figure()

	##-- A --##
    ax = Axis(
		f[1, 1],
		xlabel = "Array Size (Pixels)",
		ylabel = "Time (s)",
		title = "2D Hausdorff Loss"
	)
    lines!(
		df_loss[!, :sizes_hd], 
		df_loss[!, :hd_scipy] * 1e-9, 
		label=L"\text{HD}_{Scipy}"
	)
	lines!(
		df_loss[!, :sizes_hd], 
		df_loss[!, :hd_fenz] * 1e-9, 
		label=L"\text{HD}_{Felzenszwalb}"
	)
	lines!(
		df_loss[!, :sizes_hd], 
		df_loss[!, :hd_fenz_gpu] * 1e-9, 
		label=L"\text{HD}_{FelzenszwalbGPU}"
	)

	##-- B --##
    ax = Axis(
		f[2, 1],
		xlabel = "Array Size (Pixels)",
		ylabel = "Time (s)",
		title = "3D Hausdorff Loss"
	)
    lines!(
		df_loss[!, :sizes_hd_3D], 
		df_loss[!, :hd_scipy_3D] * 1e-9, 
		label=L"\text{HD}_{Scipy}"
	)
	lines!(
		df_loss[!, :sizes_hd_3D], 
		df_loss[!, :hd_fenz_3D] * 1e-9, 
		label=L"\text{HD}_{Felzenszwalb}"
	)
	lines!(
		df_loss[!, :sizes_hd_3D], 
		df_loss[!, :hd_fenz_gpu_3D] * 1e-9, 
		label=L"\text{HD}_{FelzenszwalbGPU}"
	)

    ##-- LABELS --##
    f[1:2, 2] = Legend(f, ax; framevisible=false)
    for (label, layout) in zip(["A", "B"], [f[1, 1], f[2, 1]])
        Label(
            layout[1, 1, TopLeft()],
            label;
            padding=(0, 0, 40, 0),
            halign=:right,
        )
    end


	save(plotsdir("loss.png"), f)
    return f
end

# ╔═╡ 7e3ac952-005a-45e6-a395-eea36ba255b5
with_theme(loss, medphys_theme)

# ╔═╡ 94f27ea3-f2f1-4680-9ace-8a31ee66ddfb
(x_hd, y_hd, z_hd) = df_loss[end, :hd_scipy], df_loss[end, :hd_fenz], df_loss[end, :hd_fenz_gpu]

# ╔═╡ 7bebe56e-99d1-4f03-8571-195bffd2fc37
gpu_vs_scipy_hd = x_hd / z_hd

# ╔═╡ 75e00691-8ac8-4962-8df6-e21dff335d3e
gpu_vs_cpu_hd = x_hd / y_hd

# ╔═╡ bd505ac6-1959-4a4e-8a04-894b9a8956a3
(x_3D_hd, y_3D_hd, z_3D_hd) = df_loss[end, :hd_scipy_3D], df_loss[end, :hd_fenz_3D], df_loss[end, :hd_fenz_gpu_3D]

# ╔═╡ b2288c31-8895-48c7-b6b6-d276b25d434e
gpu_vs_scipy_3D_hd = x_3D_hd / z_3D_hd

# ╔═╡ fc6d116f-b04f-4907-9476-237680776689
gpu_vs_cpu_3Dhd = x_3D_hd / y_3D_hd

# ╔═╡ 24e42441-7d63-4c81-a901-5ebfaeb3e7a3
md"""
# Epoch Benchmarks
"""

# ╔═╡ 787cc572-bbd5-4ca5-90c1-cb88bba40f7e
df_epochs = CSV.read(datadir("results", "loop_timing_Task02_Heart_all.csv"), DataFrame);

# ╔═╡ 29e64a2f-1eac-4be1-9d88-b18dcebe0b24
begin
	epoch_julia_dice = mean(df_epochs[2:end-5, :Dice_times]) * 1e-9
	epoch_julia_hd_scipy = mean(df_epochs[2:end-5, :Dice_HD_Scipy_times]) * 1e-9
	epoch_julia_hd_fenz = mean(df_epochs[2:end-5, :Dice_HD_Felzenszwalb_CPU_times]) * 1e-9
end

# ╔═╡ 59f7d51c-2bf7-4ad0-b3b6-d584eed50cf9
labels = [L"\text{Loss}_{DSC}", L"\text{Loss}_{Scipy}", L"\text{Loss}_{FelzenszwalbGPU}"]

# ╔═╡ 2d8a4e37-6dd9-4e46-9a3d-bd8cd71f5f65
function training_step()
    f = Figure()
    colors = Makie.wong_colors()

    ##-- TOP --##
    ax = Axis(
		f[1, 1]; 
		xticks=(1:3, labels), xticklabelrotation=0,
		title = "Average Epoch Time",
		ylabel = "Time (s)"
	)

    table = [1, 2, 3]
	h1 = epoch_julia_dice
	h2 = epoch_julia_hd_scipy
	h3 = epoch_julia_hd_fenz
	heights1 = [h1, h2, h3]
	l1 = @sprintf "%.2f" h1
	l2 = @sprintf "%.2f" h2
	l3 = @sprintf "%.2f" h3
    barplot!(table, heights1; color=colors[1:3], bar_labels=[l1, l2, l3])

	ylims!(ax; low=0, high=10)
	save(plotsdir("training.png"), f)
	
	return f
end

# ╔═╡ a9b48266-a3b5-4734-9d35-f4484aed2e95
with_theme(training_step, medphys_theme)

# ╔═╡ 0eac4ad2-2484-4e2e-aebd-49f3b370555e
md"""
# Metrics
"""

# ╔═╡ 32da60ed-0d18-477b-bbba-51b1d8080539
df_metrics = CSV.read(datadir("results", "metrics_Task02_Heart_lr_is_0.001.csv"), DataFrame);

# ╔═╡ 63f3eda3-5404-4053-bee3-05a6b77b950e
row = 27

# ╔═╡ 812f6634-8509-4588-a4ce-9a06e9fa8c15
epoch_idx = df_metrics[row, :epoch_idx]

# ╔═╡ 9e5416b9-b737-4964-9baf-15d5cbdc9474
begin
	dsc_dicemetric = df_metrics[row, :dice_metric_of_dice_model_valid_set]
	hd_dicemetric = df_metrics[row, :dice_metric_of_HD_Dice_model_valid_set]

	dsc_hausdorffmetric = df_metrics[row, :hd_metric_of_dice_model_valid_set]
	hd_hausdorffmetric = df_metrics[row, :hd_metric_of_HD_Dice_model_valid_set]
end;

# ╔═╡ 475618b0-885a-4fb3-90aa-57a7a649ddcb
df_metrics_results = DataFrame(
	"Loss Function" => [labels[1], labels[3]],
	"Dice Similarity Coefficient" => [dsc_dicemetric, hd_dicemetric],
	"Hausdorff Distance (mm)" => [dsc_hausdorffmetric, hd_hausdorffmetric]
)

# ╔═╡ 7e586270-1111-490a-849c-f41a5f55059c
save(datadir("analysis", "df_metrics_results.csv"), df_metrics_results)

# ╔═╡ f88c873b-469e-4a2f-b5cb-ce0bca21ca9f
md"""
# Visualization
"""

# ╔═╡ b7338313-06bd-484f-adcd-3cb6aa836018
md"""
## Load and prepare data
"""

# ╔═╡ 6a0ae2dd-f584-4bf2-90a0-35cbf90f9015
# const data_dir = "/Users/daleblack/Library/CloudStorage/GoogleDrive-djblack@uci.edu/My Drive/Datasets/Task02_Heart"
const data_dir = joinpath(dirname(dirname(dirname(pwd()))), "datasets/Task02_Heart")

# ╔═╡ da41cb15-042c-4f00-9430-d509c30d381b
# const model_path = "/Users/daleblack/Library/CloudStorage/GoogleDrive-djblack@uci.edu/My Drive/Datasets/hd-loss models"
const model_path = joinpath(dirname(dirname(dirname(pwd()))), "datasets/hd-loss models")

# ╔═╡ c65bb13d-3f2a-4e50-a092-1a7cc2925c7e
task2, model_dsc = loadtaskmodel(joinpath(model_path, "bigger_NN_0.001_Dice_270.jld2"))

# ╔═╡ dd01154e-ceac-4fcd-9b9a-5a9268e7b2ea
_, model_hd = loadtaskmodel(joinpath(model_path, "bigger_NN_0.001_HD_Dice_270.jld2"))

# ╔═╡ a70682fa-01fe-4efa-a944-e328f546fb5b
begin
	model_dsc_gpu = model_dsc |> gpu
	model_hd_gpu = model_hd |> gpu
end;

# ╔═╡ 22a5453f-72e7-4990-86ea-68ef6cc466cb
begin
	images(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
	masks(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
	pre_data = (
	    images(joinpath(data_dir, "imagesTr")),
	    masks(joinpath(data_dir, "labelsTr")),
	)
end

# ╔═╡ 76091037-b59c-4f82-99f4-39202b344180
image_size = (96, 96, 96)

# ╔═╡ f67046b8-1174-4306-8ebf-422203dcfdf7
img_container, mask_container = presize(pre_data)

# ╔═╡ 8fd5c922-cd1b-4143-869b-7f4f6ab3e382
data_resized = (img_container, mask_container);

# ╔═╡ 023d9b3a-5fb7-4543-91b3-d2fbb2350a24
# ╠═╡ show_logs = false
a, b = FastVision.imagedatasetstats(img_container, Gray{N0f8})

# ╔═╡ 3c4ac7bd-cc6f-42e5-abfb-0e6fba855a49
means, stds = SVector{1, Float32}(a[1]), SVector{1, Float32}(b[1])

# ╔═╡ edfb1f5f-01ff-4f31-9898-5a26db1818b7
task = SupervisedTask(
    (FastVision.Image{3}(), Mask{3}(1:2)),
    (
        ProjectiveTransforms((image_size)),
        ImagePreprocessing(means = means, stds = stds, C = Gray{N0f8}),
        FastAI.OneHot()
    )
)

# ╔═╡ 76d13603-7ab9-4c60-8f3a-9b6bce6e317a
train_files, val_files = MLDataPattern.splitobs(data_resized, 0.8);

# ╔═╡ 9afada46-48f4-4e22-86c3-351425e7c3ab
batch_size = 2

# ╔═╡ d816d63d-6875-4a70-8483-e469603660b9
tdl, vdl = FastAI.taskdataloaders(train_files, val_files, task, batch_size);

# ╔═╡ 35af3308-ab47-4c94-ab87-a6d3ca09717f
md"""
## Apply image from dataloader to models
"""

# ╔═╡ 4325537e-47c3-45ae-a162-133eda8d03d4
begin
	(example, ) = vdl
    img, msk = example
end;

# ╔═╡ af635d0c-bd5c-4e93-8295-53795afbe4cf
# ╠═╡ show_logs = false
begin
	pred_dsc = model_dsc_gpu(img |> gpu)
	pred_hd = model_hd_gpu(img |> gpu)
	
	pred_dsc = pred_dsc |> cpu
	pred_hd = pred_hd |> cpu
end;

# ╔═╡ 315e6f9b-447e-4ef4-8aaf-e74f83b31012
img_size = (512, 512, 112)

# ╔═╡ bc3528e9-3115-458a-a928-2e9e23033568
begin
	img1 = imresize(img[:, :, :, 1, 1], img_size)
	img2 = imresize(img[:, :, :, 1, 2], img_size)
	img3 = imresize(img[:, :, :, 1, 3], img_size)
	img4 = imresize(img[:, :, :, 1, 4], img_size)

	msk1 = Bool.(round.(imresize(msk[:, :, :, 2, 1] .> 0, img_size)))
	msk2 = Bool.(round.(imresize(msk[:, :, :, 2, 2] .> 0, img_size)))
	msk3 = Bool.(round.(imresize(msk[:, :, :, 2, 3] .> 0, img_size)))
	msk4 = Bool.(round.(imresize(msk[:, :, :, 2, 4] .> 0, img_size)))
end;

# ╔═╡ d1c00ac4-9ab0-4579-aa0b-b775503a2fb3
begin
	pred_dsc1 = Bool.(round.(imresize(pred_dsc[:, :, :, 2, 1] .> 0, img_size)))
	pred_dsc2 = Bool.(round.(imresize(pred_dsc[:, :, :, 2, 2] .> 0, img_size)))
	pred_dsc3 = Bool.(round.(imresize(pred_dsc[:, :, :, 2, 3] .> 0, img_size)))
	pred_dsc4 = Bool.(round.(imresize(pred_dsc[:, :, :, 2, 4] .> 0, img_size)))

	pred_hd1 = Bool.(round.(imresize(pred_hd[:, :, :, 2, 1] .> 0, img_size)))
	pred_hd2 = Bool.(round.(imresize(pred_hd[:, :, :, 2, 2] .> 0, img_size)))
	pred_hd3 = Bool.(round.(imresize(pred_hd[:, :, :, 2, 3] .> 0, img_size)))
	pred_hd4 = Bool.(round.(imresize(pred_hd[:, :, :, 2, 4] .> 0, img_size)))
end;

# ╔═╡ b7f28aad-ba4a-41a0-9ffe-1f558fd61e24
function find_edge_idxs(mask)
	edge = erode(mask) .⊻ mask
	return Tuple.(findall(isone, edge))
end

# ╔═╡ e1b32b84-998d-41e9-a7a8-a9680147f056
md"""
## Heatmap
"""

# ╔═╡ c1d410ff-c17e-4532-b1ad-777b242b3770
# @bind a1 PlutoUI.Slider(axes(img, 3), default=70, show_value=true)

# ╔═╡ 6c84a364-7fc2-4aeb-9717-34968b722dca
# begin
# 	edges_gt1 = find_edge_idxs(msk1[:, :, a1])
# 	edges_gt2 = find_edge_idxs(msk2[:, :, a1])
# 	edges_gt3 = find_edge_idxs(msk3[:, :, a1])
# 	edges_gt4 = find_edge_idxs(msk4[:, :, a1])
	
# 	edges_dsc1 = find_edge_idxs(pred_dsc1[:, :, a1])
# 	edges_dsc2 = find_edge_idxs(pred_dsc2[:, :, a1])
# 	edges_dsc3 = find_edge_idxs(pred_dsc3[:, :, a1])
# 	edges_dsc4 = find_edge_idxs(pred_dsc4[:, :, a1])
	
# 	edges_hd1 = find_edge_idxs(pred_hd1[:, :, a1])
# 	edges_hd2 = find_edge_idxs(pred_hd2[:, :, a1])
# 	edges_hd3 = find_edge_idxs(pred_hd3[:, :, a1])
# 	edges_hd4 = find_edge_idxs(pred_hd4[:, :, a1])
# end;

# ╔═╡ 470102a3-0aa6-4d34-a5f7-eb081e812edb
# let
# 	f = Figure(; resolution=(1200, 800))
# 	alpha = 1
# 	markersize = 4
	
#     ax = Axis(
# 		f[1, 1],
# 		title = labels[1]
# 	)
#     heatmap!(img4[:, :, a1]; colormap=:grays)
# 	scatter!(edges_gt4; markersize=markersize, color = :red, label="Ground Truth")
# 	scatter!(edges_dsc4; markersize=markersize, color = :blue, label="Predicted")
# 	hidedecorations!(ax)

# 	ax = Axis(
# 		f[1, 2],
# 		title = labels[3]
# 	)
#     heatmap!(img4[:, :, a1]; colormap=:grays)
# 	scatter!(edges_gt4; markersize=markersize, color = :red, label="Ground Truth")
# 	scatter!(edges_hd4; markersize=markersize, color = :blue, label="Predicted")
# 	hidedecorations!(ax)

# 	axislegend(ax)
# 	f
# end

# ╔═╡ 4f43c115-20eb-4041-9f8a-40286a9fd7e5
function countours()
	f = Figure(resolution = (1200, 2000))
	alpha = 1
	markersize = 2

	#------ Row 1 ------#
    ax = Axis(
		f[1, 1],
		title = labels[1],
		ylabel = "Patient 1"
	)
	z = 85
	img = img1[:, :, z]
	edges_gt = find_edge_idxs(msk1[:, :, z])
	edges_dsc = find_edge_idxs(pred_dsc1[:, :, z])
	edges_hd = find_edge_idxs(pred_hd1[:, :, z])
    
	heatmap!(img; colormap=:grays)
	scatter!(edges_gt; markersize=markersize, color = :red, label="Ground Truth")
	scatter!(edges_dsc; markersize=markersize, color = :blue, label="Predicted")
	hidedecorations!(ax, label = false)

	ax = Axis(
		f[1, 2],
		title = labels[3]
	)
    heatmap!(img; colormap=:grays)
	scatter!(edges_gt; markersize=markersize, color = :red, label="Ground Truth")
	scatter!(edges_hd; markersize=markersize, color = :blue, label="Predicted")
	hidedecorations!(ax)
	elem_gt = LineElement(color = :red, linestyle = nothing)
	elem_pred = LineElement(color = :blue, linestyle = nothing)
	
	# axislegend(ax)

	#------ Row 2 ------#
    ax = Axis(
		f[2, 1],
		ylabel = "Patient 2"
	)
	z = 60
	img = img2[:, :, z]
	edges_gt = find_edge_idxs(msk2[:, :, z])
	edges_dsc = find_edge_idxs(pred_dsc2[:, :, z])
	edges_hd = find_edge_idxs(pred_hd2[:, :, z])
    
	heatmap!(img; colormap=:grays)
	scatter!(edges_gt; markersize=markersize, color = :red, label="Ground Truth")
	scatter!(edges_dsc; markersize=markersize, color = :blue, label="Predicted")
	hidedecorations!(ax, label = false)

	ax = Axis(
		f[2, 2],
	)
    heatmap!(img; colormap=:grays)
	scatter!(edges_gt; markersize=markersize, color = :red, label="Ground Truth")
	scatter!(edges_hd; markersize=markersize, color = :blue, label="Predicted")
	hidedecorations!(ax)

	#------ Row 3 ------#
    ax = Axis(
		f[3, 1],
		ylabel = "Patient 3"
	)
	z = 57
	img = img4[:, :, z]
	edges_gt = find_edge_idxs(msk4[:, :, z])
	edges_dsc = find_edge_idxs(pred_dsc4[:, :, z])
	edges_hd = find_edge_idxs(pred_hd4[:, :, z])
    
	heatmap!(img; colormap=:grays)
	scatter!(edges_gt; markersize=markersize, color = :red, label="Ground Truth")
	scatter!(edges_dsc; markersize=markersize, color = :blue, label="Predicted")
	hidedecorations!(ax, label = false)

	ax = Axis(
		f[3, 2],
	)
    heatmap!(img; colormap=:grays)
	scatter!(edges_gt; markersize=markersize, color = :red, label="Ground Truth")
	scatter!(edges_hd; markersize=markersize, color = :blue, label="Predicted")
	hidedecorations!(ax)

	Legend(
		f[2, 3],
    	[elem_gt, elem_pred,],
    	["Ground Truth", "Predicted"],
		framevisible = false
	)

	save(plotsdir("contours.png"), f)
	f
end

# ╔═╡ cb17ad69-3cf6-4dbc-b226-d8a11efb82b1
with_theme(countours, medphys_theme)

# ╔═╡ Cell order:
# ╠═5463443f-a69e-4766-bfc2-0b7ca0fd48c9
# ╠═6495cb7b-67ec-49da-bc5d-f7a63ca9e5e9
# ╠═ec608b5e-a6d5-4b12-a3a4-cc1738dbc6ed
# ╠═d1a12515-a9d0-468b-8978-dbb26a1ee667
# ╠═f88c7074-8b5c-4dc7-83d5-e65a24b88ab0
# ╠═278dfa0e-46e1-4789-9f51-eb3463a9fb00
# ╠═8608a972-73c0-4903-bdd7-9a23f7c57337
# ╠═02f182d4-f5ab-46a3-8009-33ed641fcf27
# ╟─95a525fe-e10b-42ee-a1b9-9405fe16b50c
# ╠═0960ac79-d976-42a7-aaac-4d07ec94927e
# ╟─c1659832-9e1a-4af5-9bf6-fd4d6f12589f
# ╠═aa312920-1c67-46f7-a7b1-dfe42f54c769
# ╠═6349d56d-f837-4395-9819-e8e7f34ba01f
# ╠═a0ea7e69-1249-4d47-9366-ec9f44568236
# ╠═5059495d-6905-4043-b03e-3ea73f67d341
# ╠═9e02ae06-a2cd-427c-8e94-205bd8fbc0f3
# ╠═f2faa757-6a21-40ba-9f36-d3149a37a5fc
# ╠═609c529b-8dd9-4daa-9cd8-b89d7a4ce792
# ╟─1852f60f-cdf1-47d5-8d97-052710bed435
# ╠═fbd61bd3-2181-4d24-a417-85bc56a9b51e
# ╟─b7236b7b-cf97-4962-b49d-58037b52201d
# ╟─7e3ac952-005a-45e6-a395-eea36ba255b5
# ╠═94f27ea3-f2f1-4680-9ace-8a31ee66ddfb
# ╠═7bebe56e-99d1-4f03-8571-195bffd2fc37
# ╠═75e00691-8ac8-4962-8df6-e21dff335d3e
# ╠═bd505ac6-1959-4a4e-8a04-894b9a8956a3
# ╠═b2288c31-8895-48c7-b6b6-d276b25d434e
# ╠═fc6d116f-b04f-4907-9476-237680776689
# ╟─24e42441-7d63-4c81-a901-5ebfaeb3e7a3
# ╠═787cc572-bbd5-4ca5-90c1-cb88bba40f7e
# ╠═29e64a2f-1eac-4be1-9d88-b18dcebe0b24
# ╠═59f7d51c-2bf7-4ad0-b3b6-d584eed50cf9
# ╟─2d8a4e37-6dd9-4e46-9a3d-bd8cd71f5f65
# ╟─a9b48266-a3b5-4734-9d35-f4484aed2e95
# ╟─0eac4ad2-2484-4e2e-aebd-49f3b370555e
# ╠═32da60ed-0d18-477b-bbba-51b1d8080539
# ╠═63f3eda3-5404-4053-bee3-05a6b77b950e
# ╠═812f6634-8509-4588-a4ce-9a06e9fa8c15
# ╠═9e5416b9-b737-4964-9baf-15d5cbdc9474
# ╠═475618b0-885a-4fb3-90aa-57a7a649ddcb
# ╠═7e586270-1111-490a-849c-f41a5f55059c
# ╟─f88c873b-469e-4a2f-b5cb-ce0bca21ca9f
# ╟─b7338313-06bd-484f-adcd-3cb6aa836018
# ╠═f81ac05c-b5f2-4e96-b370-adc3b073d5c9
# ╠═6a0ae2dd-f584-4bf2-90a0-35cbf90f9015
# ╠═da41cb15-042c-4f00-9430-d509c30d381b
# ╠═c65bb13d-3f2a-4e50-a092-1a7cc2925c7e
# ╠═dd01154e-ceac-4fcd-9b9a-5a9268e7b2ea
# ╠═a70682fa-01fe-4efa-a944-e328f546fb5b
# ╠═22a5453f-72e7-4990-86ea-68ef6cc466cb
# ╠═76091037-b59c-4f82-99f4-39202b344180
# ╠═f67046b8-1174-4306-8ebf-422203dcfdf7
# ╠═8fd5c922-cd1b-4143-869b-7f4f6ab3e382
# ╠═023d9b3a-5fb7-4543-91b3-d2fbb2350a24
# ╠═3c4ac7bd-cc6f-42e5-abfb-0e6fba855a49
# ╠═edfb1f5f-01ff-4f31-9898-5a26db1818b7
# ╠═76d13603-7ab9-4c60-8f3a-9b6bce6e317a
# ╠═9afada46-48f4-4e22-86c3-351425e7c3ab
# ╠═d816d63d-6875-4a70-8483-e469603660b9
# ╟─35af3308-ab47-4c94-ab87-a6d3ca09717f
# ╠═20114ef8-cf6b-4a92-89b8-40a85571d6c2
# ╠═4325537e-47c3-45ae-a162-133eda8d03d4
# ╠═af635d0c-bd5c-4e93-8295-53795afbe4cf
# ╠═315e6f9b-447e-4ef4-8aaf-e74f83b31012
# ╠═bc3528e9-3115-458a-a928-2e9e23033568
# ╠═d1c00ac4-9ab0-4579-aa0b-b775503a2fb3
# ╠═b7f28aad-ba4a-41a0-9ffe-1f558fd61e24
# ╟─e1b32b84-998d-41e9-a7a8-a9680147f056
# ╠═c1d410ff-c17e-4532-b1ad-777b242b3770
# ╠═6c84a364-7fc2-4aeb-9717-34968b722dca
# ╠═470102a3-0aa6-4d34-a5f7-eb081e812edb
# ╟─4f43c115-20eb-4041-9f8a-40286a9fd7e5
# ╟─cb17ad69-3cf6-4dbc-b226-d8a11efb82b1
