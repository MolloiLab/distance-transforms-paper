### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

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
df_epochs = CSV.read(datadir("results", "epoch_time_2023_06_09.csv"), DataFrame);

# ╔═╡ 29e64a2f-1eac-4be1-9d88-b18dcebe0b24
begin
	epoch_julia_dice = mean(df_epochs[2:40, :Dice]) * 1e-9
	epoch_julia_hd_scipy = mean(df_epochs[2:40, :Scipy]) * 1e-9
	epoch_julia_hd_fenz = mean(df_epochs[2:40, :Felzenszwalb_gpu]) * 1e-9
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
	save(plotsdir("training_step.png"), f)
	
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

# ╔═╡ 98c45c64-092b-437b-8cb2-99094210e1ad
function training_metrics()
	f = Figure()

	ax = Axis(
		f[1, 1],
		title = "Dice Metric",
		xlabel = "Epoch",
		ylabel = "Dice Similarity Coefficient",
	)
	lines!(df_metrics[!, :epoch_idx], df_metrics[!, :dice_metric_of_dice_model_valid_set], label=labels[1])
	lines!(df_metrics[!, :epoch_idx], df_metrics[!, :dice_metric_of_HD_Dice_model_valid_set], label=labels[3])

	ax = Axis(
		f[2, 1],
		title = "Hausdorff Metric",
		xlabel = "Epoch",
		ylabel = "Hausdorff Distance",
	)
	lines!(df_metrics[!, :epoch_idx], df_metrics[!, :hd_metric_of_dice_model_valid_set], label=labels[1])
	lines!(df_metrics[!, :epoch_idx], df_metrics[!, :hd_metric_of_HD_Dice_model_valid_set], label=labels[3])
	
	f[1:2, 2] = Legend(f, ax; framevisible=false)
    for (label, layout) in zip(["A", "B"], [f[1, 1], f[2, 1]])
        Label(
            layout[1, 1, TopLeft()],
            label;
            padding=(0, 0, 40, 0),
            halign=:right,
        )
    end
	save(plotsdir("dice_hd_julia.png"), f)
	f
end

# ╔═╡ bfb6cd41-9ff2-4f21-8059-2e2771a2b708
with_theme(training_metrics, medphys_theme)

# ╔═╡ 63f3eda3-5404-4053-bee3-05a6b77b950e
row = 27

# ╔═╡ 812f6634-8509-4588-a4ce-9a06e9fa8c15
epoch_idx = df_metrics[row, :epoch_idx]

# ╔═╡ f7bcc6a9-f15d-4e92-aec8-94ae6f022053
df_metrics

# ╔═╡ 9e5416b9-b737-4964-9baf-15d5cbdc9474
begin
	dsc_dicemetric = maximum(df_metrics[:, :dice_metric_of_dice_model_valid_set])
	hd_dicemetric = maximum(df_metrics[:, :dice_metric_of_HD_Dice_model_valid_set])

	dsc_hausdorffmetric = minimum(df_metrics[:, :hd_metric_of_dice_model_valid_set])
	hd_hausdorffmetric = minimum(df_metrics[:, :hd_metric_of_HD_Dice_model_valid_set])

	dsc_dicemetric_perc = percentile(df_metrics[:, :dice_metric_of_dice_model_valid_set], 90)
	hd_dicemetric_perc = percentile(df_metrics[:, :dice_metric_of_HD_Dice_model_valid_set], 90)

	dsc_hausdorffmetric_perc = percentile(df_metrics[:, :hd_metric_of_dice_model_valid_set], 10)
	hd_hausdorffmetric_perc = percentile(df_metrics[:, :hd_metric_of_HD_Dice_model_valid_set], 10)
end

# ╔═╡ 475618b0-885a-4fb3-90aa-57a7a649ddcb
df_metrics_results = DataFrame(
	"Loss Function" => ["Loss_{DSC}", "Loss_{FelzenszwalbGPU}"],
	"Dice Similarity Coefficient (Best)" => [dsc_dicemetric, hd_dicemetric],
	"Dice Similarity Coefficient (90th %)" => [dsc_dicemetric_perc, hd_dicemetric_perc],
	"Hausdorff Distance (Best) (mm)" => [dsc_hausdorffmetric, hd_hausdorffmetric],
	"Hausdorff Distance (10th %) (mm)" => [dsc_hausdorffmetric_perc, hd_hausdorffmetric_perc],
)

# ╔═╡ f83c86f5-45e0-461a-a583-e2995981f33d
save(datadir("analysis", "metrics_results.csv"), df_metrics_results)

# ╔═╡ f88c873b-469e-4a2f-b5cb-ce0bca21ca9f
md"""
# Visualization
"""

# ╔═╡ b7338313-06bd-484f-adcd-3cb6aa836018
md"""
## Load and prepare data
"""

# ╔═╡ 76091037-b59c-4f82-99f4-39202b344180
image_size = (96, 96, 96)

# ╔═╡ 9afada46-48f4-4e22-86c3-351425e7c3ab
batch_size = 1

# ╔═╡ 35af3308-ab47-4c94-ab87-a6d3ca09717f
md"""
## Apply image from dataloader to model
"""

# ╔═╡ 315e6f9b-447e-4ef4-8aaf-e74f83b31012
img_size = (512, 512, 112)

# ╔═╡ e1b32b84-998d-41e9-a7a8-a9680147f056
md"""
## Heatmap
"""

# ╔═╡ 4f43c115-20eb-4041-9f8a-40286a9fd7e5
# let
# 	f = Figure(
# 	)
# 	alpha = 1
# 	markersize = 2

# 	slice_1 = 70
# 	y1_edge, y1_pred_edge, y1_pred_dice_edge = get_mask_edges(y1[:, :, slice_1], y1_pred[:, :, slice_1], y1_pred_dice[:, :, slice_1])
# 	y1_edge, y1_pred_edge, y1_pred_dice_edge = Tuple.(y1_edge), Tuple.(y1_pred_edge), Tuple.(y1_pred_dice_edge)
	
#     ax1 = Axis(
# 		f[1, 1],
# 		title = "Dice Loss"
# 	)
#     heatmap!(x1[:, :, slice_1]; colormap=:grays)
# 	scatter!(y1_edge; markersize=markersize, color = (:red, alpha), label="Ground Truth")
# 	scatter!(y1_pred_dice_edge; markersize=markersize, color = (:blue, alpha), label="Predicted")
# 	hidedecorations!(ax1)
	

# 	ax2 = Axis(
# 		f[1, 2],
# 		title = "HD-Dice Loss"
# 	)
#     heatmap!(x1[:, :, slice_1]; colormap=:grays)
# 	scatter!(y1_edge; markersize=markersize, color = (:red, alpha), label="Ground Truth")
# 	scatter!(y1_pred_edge; markersize=markersize, color = (:blue, alpha), label="Predicted")
# 	hidedecorations!(ax2)


# 	slice_2 = 43
# 	y2_edge, y2_pred_edge, y2_pred_dice_edge = get_mask_edges(y2[:, :, slice_2], y2_pred[:, :, slice_2], y2_pred_dice[:, :, slice_2])
# 	y2_edge, y2_pred_edge, y2_pred_dice_edge = Tuple.(y2_edge), Tuple.(y2_pred_edge), Tuple.(y2_pred_dice_edge)

# 	ax3 = Axis(
# 		f[2, 1]
# 	)
#     heatmap!(x2[:, :, slice_2]; colormap=:grays)
# 	scatter!(y2_edge; markersize=markersize, color = (:red, alpha), label="Ground Truth")
# 	scatter!(y2_pred_dice_edge; markersize=markersize, color = (:blue, alpha), label="Predicted")
# 	hidedecorations!(ax3)
	
# 	ax4 = Axis(
# 		f[2, 2]
# 	)
#     heatmap!(x2[:, :, slice_2]; colormap=:grays)
# 	scatter!(y2_edge; markersize=markersize, color = (:red, alpha), label="Ground Truth")
# 	scatter!(y2_pred_edge; markersize=markersize, color = (:blue, alpha), label="Predicted")
# 	hidedecorations!(ax4)

# 	Legend(f[1:2, 3],
#     [LineElement(color = :red, linestyle = nothing), LineElement(color = :blue, linestyle = nothing)],
#     ["Ground Truth", "Predicted"]; framevisible=false)

# 	save(plotsdir("contours.png"), f)
	
#     f
# end

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
# ╟─aa312920-1c67-46f7-a7b1-dfe42f54c769
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
# ╟─98c45c64-092b-437b-8cb2-99094210e1ad
# ╠═bfb6cd41-9ff2-4f21-8059-2e2771a2b708
# ╠═63f3eda3-5404-4053-bee3-05a6b77b950e
# ╠═812f6634-8509-4588-a4ce-9a06e9fa8c15
# ╠═f7bcc6a9-f15d-4e92-aec8-94ae6f022053
# ╠═9e5416b9-b737-4964-9baf-15d5cbdc9474
# ╠═475618b0-885a-4fb3-90aa-57a7a649ddcb
# ╠═f83c86f5-45e0-461a-a583-e2995981f33d
# ╟─f88c873b-469e-4a2f-b5cb-ce0bca21ca9f
# ╟─b7338313-06bd-484f-adcd-3cb6aa836018
# ╠═f81ac05c-b5f2-4e96-b370-adc3b073d5c9
# ╠═76091037-b59c-4f82-99f4-39202b344180
# ╠═9afada46-48f4-4e22-86c3-351425e7c3ab
# ╟─35af3308-ab47-4c94-ab87-a6d3ca09717f
# ╠═20114ef8-cf6b-4a92-89b8-40a85571d6c2
# ╠═315e6f9b-447e-4ef4-8aaf-e74f83b31012
# ╟─e1b32b84-998d-41e9-a7a8-a9680147f056
# ╠═4f43c115-20eb-4041-9f8a-40286a9fd7e5
