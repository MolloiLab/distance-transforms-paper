### A Pluto.jl notebook ###
# v0.19.22

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
using DrWatson; 

# ╔═╡ 6495cb7b-67ec-49da-bc5d-f7a63ca9e5e9
# ╠═╡ show_logs = false
@quickactivate "hd-loss"

# ╔═╡ ec608b5e-a6d5-4b12-a3a4-cc1738dbc6ed
# ╠═╡ show_logs = false
using PlutoUI, Statistics, CSV, DataFrames, GLM, CairoMakie, HypothesisTests, Colors, MLJBase, Glob, Flux, NIfTI, Images, ImageMorphology, FastAI, FastVision, StaticArrays, MLDataPattern, Printf, CSVFiles

# ╔═╡ d1a12515-a9d0-468b-8978-dbb26a1ee667
using CairoMakie: Axis, Label

# ╔═╡ f88c7074-8b5c-4dc7-83d5-e65a24b88ab0
include(srcdir("load_model.jl")), include(srcdir("postprocessing.jl")); 

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

# ╔═╡ 95a525fe-e10b-42ee-a1b9-9405fe16b50c
md"""
# Distance Transforms
"""

# ╔═╡ 82134f7b-2cb1-4d9a-982f-f8b2799c9518
df_dt = CSV.read(datadir("results", "dts.csv"), DataFrame);

# ╔═╡ 6349d56d-f837-4395-9819-e8e7f34ba01f
(x, y, z) = df_dt[end, :dt_scipy], df_dt[end, :dt_fenz], df_dt[end, :dt_fenz_gpu]

# ╔═╡ a0ea7e69-1249-4d47-9366-ec9f44568236
fenzGPU_vs_scipy = x / z

# ╔═╡ 5059495d-6905-4043-b03e-3ea73f67d341
wenbo_gpu_vs_wenbo_cpu = x / y

# ╔═╡ 9c2b898e-0295-44da-8e3f-bab93524eec3
(x_3D, y_3D, z_3D) = df_dt[end, :dt_scipy_3D], df_dt[end, :dt_fenz_3D], df_dt[end, :dt_fenz_gpu_3D]

# ╔═╡ 7b34de9e-747b-4a80-870a-6f26178a378d
fenzGPU_vs_scipy_3D = x_3D / z_3D

# ╔═╡ 1852f60f-cdf1-47d5-8d97-052710bed435
md"""
# Losses
"""

# ╔═╡ fbda132c-1f19-4103-84f0-34215ce4e5b7
df_loss = CSV.read(datadir("results", "losses.csv"), DataFrame);

# ╔═╡ bac3b79b-8b61-4bec-9478-fca3b4c901c5
df_loss[!, :sizes_hd], df_loss[!, :sizes_hd_3D]

# ╔═╡ 892bf5ef-012c-49c2-bb2d-80a006965c28
df_loss

# ╔═╡ 6a77f8f4-5880-4267-ae44-0de8077bce1e
function dt()
    f = Figure()

    ##-- A --##
    ax1 = Axis(
		f[1, 1],
		xlabel = "Array Size (Pixels)",
		ylabel = "Time (ms)",
		title = "2D Distance Transform"
	)
    lines!(df_dt[!, :sizes], df_dt[!, :dt_scipy] * 1e-6, label=L"\text{DT}_{\text{Scipy}}")
	lines!(df_dt[!, :sizes], df_dt[!, :dt_fenz] * 1e-6, label=L"\text{DT}_{\text{Felzenszwalb}}")
    lines!(df_dt[!, :sizes], df_dt[!, :dt_fenz_gpu] * 1e-6, label=L"\text{DT}_{\text{Felzenszwalb}_\text{GPU}}")

	##-- B --##
    ax2 = Axis(
		f[2, 1],
		xlabel = "Array Size (Pixels)",
		ylabel = "Time (ms)",
		title = "3D Distance Transform"
	)
    lines!(df_loss[!, :sizes_hd_3D], df_dt[!, :dt_scipy_3D] * 1e-6, label=L"\text{DT}_{\text{Scipy}}")
	lines!(df_loss[!, :sizes_hd_3D], df_dt[!, :dt_fenz_3D] * 1e-6, label=L"\text{DT}_{\text{Felzenszwalb}}")
    lines!(df_loss[!, :sizes_hd_3D], df_dt[!, :dt_fenz_gpu_3D] * 1e-6, label=L"\text{DT}_{\text{Felzenszwalb}_\text{GPU}}")

    ##-- LABELS --##
    f[1:2, 2] = Legend(f, ax1; framevisible=false, labelsize=25)
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

# ╔═╡ bb891719-2676-49c9-8b23-dfdfc6f4fd3d
with_theme(dt, medphys_theme)

# ╔═╡ 8a054ee3-b388-4ea6-9b1a-3dbcc668714d
function loss()
    f = Figure()

	##-- A --##
    ax = Axis(
		f[1, 1:2],
		xlabel = "Array Size (Pixels)",
		ylabel = "Time (ms)",
		title = "2D Loss",
		xticklabelrotation = pi/4
	)
	lines!(df_loss[!, :sizes_hd], df_loss[!, :hd_scipy] * 1e-6, label=L"\text{HD}_{\text{Scipy}}")
	lines!(df_loss[!, :sizes_hd], df_loss[!, :hd_fenz] * 1e-6, label=L"\text{HD}_{\text{Felzenszwalb}}")
	lines!(df_loss[!, :sizes_hd], df_loss[!, :hd_fenz_gpu] * 1e-6, label=L"\text{HD}_{\text{Felzenszwalb}_\text{GPU}}")

	##-- B --##
    ax = Axis(
		f[2, 1:2],
		xlabel = "Array Size (Voxels)",
		ylabel = "Time (ms)",
		yticks = [0, 200, 400, 600],
		title = "3D Loss",
		xticklabelrotation = pi/4
	)
	lines!(df_loss[!, :sizes_hd_3D], df_loss[!, :hd_scipy_3D] * 1e-6, label=L"\text{HD}_{\text{Scipy}}")
	lines!(df_loss[!, :sizes_hd_3D], df_loss[!, :hd_fenz_3D] * 1e-6, label=L"\text{HD}_{\text{Felzenszwalb}}")
	lines!(df_loss[!, :sizes_hd_3D], df_loss[!, :hd_fenz_gpu_3D] * 1e-6, label=L"\text{HD}_{\text{Felzenszwalb}_\text{GPU}}")

   	##-- LABELS --##
    f[1:2, end+1] = Legend(f, ax; framevisible=false, labelsize=25)
    for (label, layout) in zip(["A", "B"], [f[1, 1], f[2, 1]])
        Label(
            layout[1, 1, TopLeft()],
            label;
            # fontsize=25,
            padding=(0, 0, 40, 0),
            halign=:right,
        )
    end

	save(plotsdir("loss.png"), f)
    return f
end

# ╔═╡ ca48acf9-7546-40ab-ad69-c5a6ed3092b9
with_theme(loss, medphys_theme)

# ╔═╡ 24e42441-7d63-4c81-a901-5ebfaeb3e7a3
md"""
# Training Loops
"""

# ╔═╡ 622443e7-b90c-491a-beaf-d7aa668474ca
begin
	path_julia_loop = datadir("loop")
	path_py_loop = datadir("loop-py")
	
end

# ╔═╡ 810674a3-6be0-47c1-8b3f-ff9b385ccf77
md"""
## Step
"""

# ╔═╡ 93afb259-43b6-4796-9fce-3d715538a47a
begin
	df_step_julia = CSV.read(joinpath(path_julia_loop, "Step_Time.csv"), DataFrame)
	df_step_julia_manuer = CSV.read(joinpath(path_julia_loop, "Step_Time_Manuer.csv"), DataFrame)
end;

# ╔═╡ 29e64a2f-1eac-4be1-9d88-b18dcebe0b24
begin
	step_julia_dice = mean(df_step_julia[!, :step_times_dice]) * 1e-9
	step_julia_hd_cpu = mean(df_step_julia_manuer[!, :step_times_dice_hd]) * 1e-9
	step_julia_hd_gpu = mean(df_step_julia[!, :step_times_dice_hd]) * 1e-9
end

# ╔═╡ 2d8a4e37-6dd9-4e46-9a3d-bd8cd71f5f65
function training_step()
    f = Figure()
    colors = Makie.wong_colors()

    ##-- TOP --##
    ax = Axis(
		f[1, 1]; 
		xticks=(1:3, [L"\text{Loss}_{\text{DSC}}", L"\text{Loss}_{\text{DSC/HD}}^{\text{CPU}}", L"\text{Loss}_{\text{DSC/HD}}^{\text{GPU}}"]), xticklabelrotation=0,
		title = "Average Step Time",
		ylabel = "Time (s)"
	)

    table = [1, 2, 3]
	h1 = step_julia_dice
	h2 = step_julia_hd_cpu
	h3 = step_julia_hd_gpu
	heights1 = [h1, h2, h3]
	l1 = @sprintf "%.2f" h1
	l2 = @sprintf "%.2f" h2
	l3 = @sprintf "%.2f" h3
    barplot!(table, heights1; color=colors[1:3], bar_labels=[l1, l2, l3])

	ylims!(ax; low=0, high=1)
	save(plotsdir("training_step.png"), f)
	
	return f
end

# ╔═╡ a9b48266-a3b5-4734-9d35-f4484aed2e95
with_theme(training_step, medphys_theme)

# ╔═╡ 5691a8ac-cb96-435f-aaef-48e4f72f77e5
md"""
## Epoch
"""

# ╔═╡ 87dfe577-9ffa-4e89-9c2b-d2cde95fcbae
begin
	df_epoch_julia = CSV.read(joinpath(path_julia_loop, "Epoch_Time.csv"), DataFrame)
	df_epoch_julia_manuer = CSV.read(joinpath(path_julia_loop, "Epoch_Time_Manuer.csv"), DataFrame)
end;

# ╔═╡ 23a06ef1-aff1-43b0-b328-e3e01e95efcf
begin
	epoch_julia_dice = mean(df_epoch_julia[!, :epoch_times_dice]) * 1e-9
	epoch_julia_hd_cpu = mean(df_epoch_julia_manuer[!, :epoch_times_dice_hd]) * 1e-9
	epoch_julia_hd_gpu = mean(df_epoch_julia[!, :epoch_times_dice_hd]) * 1e-9
end

# ╔═╡ 7d3aab45-0e21-4be4-8034-6f702025f1b6
function training_epoch()
	f = Figure()
    colors = Makie.wong_colors()
	##-- Bottom --##
     ax = Axis(
		 f[1, 1]; 
		xticks=(1:3, [L"\text{Loss}_{\text{DSC}}", L"\text{Loss}_{\text{DSC/HD}}^{\text{CPU}}", L"\text{Loss}_{\text{DSC/HD}}^{\text{GPU}}"]), xticklabelrotation=0,
		 title = "Average Epoch Time",
		 ylabel = "Time (s)"
	 )

    table = [1, 2, 3]
	h1 = epoch_julia_dice
	h2 = epoch_julia_hd_cpu
	h3 = epoch_julia_hd_gpu
	heights1 = [h1, h2, h3]
	l1 = @sprintf "%.2f" h1
	l2 = @sprintf "%.2f" h2
	l3 = @sprintf "%.2f" h3
    barplot!(table, heights1; color=colors[1:3], bar_labels=[l1, l2, l3])
	
	ylims!(ax; low=0, high=5)

	save(plotsdir("training_epoch.png"), f)

    return f
end

# ╔═╡ fa56d7f3-4e00-43d2-84bc-913c77c0fbe4
with_theme(training_epoch, medphys_theme)

# ╔═╡ 29fbb7d4-e992-4b2b-a0af-5a31bcf6ba00
md"""
# Accuracy
"""

# ╔═╡ 2898cc85-d85e-46fe-988f-3843972fe7f1
path_acc = datadir("model-data")

# ╔═╡ 0eac4ad2-2484-4e2e-aebd-49f3b370555e
md"""
## Metrics
"""

# ╔═╡ bdbae682-29c5-4920-8983-0d445a153b31
begin
	df_dice_metrics = CSV.read(joinpath(path_acc, "Julia_Loop_Dice_Metric_best_model_jan_12.csv"), DataFrame)
	df_hd_metrics = CSV.read(joinpath(path_acc, "Julia_Loop_HD_Metric_best_model_jan_12.csv"), DataFrame)
end;

# ╔═╡ 15316f17-c282-4618-967a-7539afe497fa
begin
	diceloss_best_dicemetric = percentile(filter(!iszero, df_dice_metrics[!, :valid_dice_metric_dice]), 90)
	hdloss_best_dicemetric = percentile(filter(!iszero, df_dice_metrics[!, :valid_dice_metric_dice_hd]), 90)

	diceloss_best_hdmetric = percentile(filter(!isinf, df_hd_metrics[!, :valid_hd_metric_dice]), 10)
	hdloss_best_hdmetric = percentile(filter(!isinf, df_hd_metrics[!, :valid_hd_metric_dice_hd]), 10)
end

# ╔═╡ 40fbef10-8a8a-4f80-964c-882894742bbf
function hd_dice_julia()
    f = Figure()

    ##-- A --##
    ax = Axis(
		f[1, 1],
		xlabel = "Epoch",
		ylabel = "Hausdorff Metric (mm)",
		title = "HD"
	)
    lines!(filter(!isinf, df_hd_metrics[!, :valid_hd_metric_dice]), label=L"\text{Loss}_{\text{DSC}}")
    lines!(filter(!isinf, df_hd_metrics[!, :valid_hd_metric_dice_hd]), label=L"\text{Loss}_{\text{DSC/HD}}^{\text{GPU}}")
	ylims!(ax, low=-5, high=100)

	#-- B --##
    ax = Axis(
		f[2, 1],
		xlabel = "Epoch",
		ylabel = "Dice Metric",
		title = "DSC"
	)
    lines!(filter(!isinf, df_dice_metrics[!, :valid_dice_metric_dice]), label=L"\text{Loss}_{\text{DSC}}")
    lines!(filter(!isinf, df_dice_metrics[!, :valid_dice_metric_dice_hd]), label=L"\text{Loss}_{\text{DSC/HD}}^{\text{GPU}}")
	ylims!(ax, low=0, high=1)

    ##-- LABELS --##
    f[1:2, 2] = Legend(f, ax; framevisible=false)
    for (label, layout) in zip(["A", "B"], [f[1, 1], f[2, 1]])
        Label(
            layout[1, 1, TopLeft()],
            label;
            fontsize=25,
            padding=(0, 0, 40, 0),
            halign=:right,
        )
    end

	save(plotsdir("dice_hd_julia.png"), f)

    return f
end

# ╔═╡ 9f20d955-a8cd-47bb-aed4-b0b50e915df5
with_theme(hd_dice_julia, medphys_theme)

# ╔═╡ 475618b0-885a-4fb3-90aa-57a7a649ddcb
df_accuracy_julia = DataFrame(
	"Loss Function" => ["Dice Loss", "HD-Dice Loss"
	],
	"Dice Metric (90%)" => [diceloss_best_dicemetric, hdloss_best_dicemetric],
	"Hausdorff Metric (10%, mm)" => [diceloss_best_hdmetric, hdloss_best_hdmetric]
)

# ╔═╡ 7e586270-1111-490a-849c-f41a5f55059c
save(datadir("analysis", "accuracy.csv"), df_accuracy_julia)

# ╔═╡ 0e6fe7e8-0f10-4fef-86e2-4673011acf08
md"""
## Loss
"""

# ╔═╡ 0ef58a86-2901-40f0-9c62-9cbda7fbf03a
df_train_loss = CSV.read(joinpath(path_acc, "Julia_Loop_Loss_best_model_jan_12.csv"), DataFrame);

# ╔═╡ b1622073-ffcd-41bd-b452-8abbbc4e0a33
function training_loss()
    f = Figure()

    ##-- A --##
    ax = Axis(
		f[1, 1],
		xlabel = "Epoch",
		ylabel = "Loss",
		title = "Training Losses (Dice Loss)"
	)
    lines!(filter(!isinf, df_train_loss[!, :train_losses_dice]), label="Train Loss")
    lines!(filter(!isinf, df_train_loss[!, :valid_losses_dice]), label="Validation Loss")

	##-- B --##
    ax = Axis(
		f[2, 1],
		xlabel = "Epoch",
		ylabel = "Loss",
		title = "Training Losses (HD-Dice Loss)"
	)
    lines!(filter(!isinf, df_train_loss[!, :train_losses_dice_hd]), label="Train Loss")
    lines!(filter(!isinf, df_train_loss[!, :valid_losses_dice_hd]), label="Validation Loss")

	##-- LABELS --##
    f[1:2, 2] = Legend(f, ax; framevisible=false)
    for (label, layout) in zip(["A", "B"], [f[1, 1], f[2, 1]])
        Label(
            layout[1, 1, TopLeft()],
            label;
            fontsize=25,
            padding=(0, 0, 40, 0),
            halign=:right,
        )
    end

    # save(joinpath(dirname(pwd()),"figures", "training.png"), f)
    return f
end

# ╔═╡ 9a4223ed-f075-4f15-b2a8-f505e83fa253
with_theme(training_loss, medphys_theme)



# ╔═╡ 0614b6e9-9853-4f09-b8de-b3e8d1775aba
# md"""
# # Contour Visualizations
# """



# ╔═╡ 4ace7533-de03-4a31-847e-69f30d7ce0b7
# md"""
# ## Load model
# """



# ╔═╡ c356a52f-88e1-4659-a122-56dc2fe9315f
path_to_dataset = "/Users/daleblack/Library/CloudStorage/GoogleDrive-djblack@uci.edu/My Drive/Datasets/Task02_Heart";

# ╔═╡ ece12891-9445-4a43-8201-9b99f20e2352
begin
	path_to_DICE_HD_model = datadir("models", "Dice_HD_125.jld2")
	path_to_DICE_model = datadir("models", "Dice_125.jld2")
end;

# ╔═╡ 1a4f5225-b905-4b87-af71-997817141460
tdl, vdl, model_dice, model_dice_hd = load_data_and_model(path_to_dataset, path_to_DICE_model, path_to_DICE_HD_model);

# ╔═╡ 35af3308-ab47-4c94-ab87-a6d3ca09717f
md"""
### Apply image from dataloader to model
"""

# ╔═╡ 4325537e-47c3-45ae-a162-133eda8d03d4
begin
	(example,) = vdl
    xs, ys = example
end;

# ╔═╡ 98cebb61-5fa4-4800-a734-ca6190cca5f0
begin
    y_pred = model_dice_hd(xs)
    y_pred = keep_largest_component(argmax_2ch(y_pred))
end;

# ╔═╡ 9ffc5a7b-97fc-46cc-9b2b-6aa89f060fee
begin
    y_pred_dice = model_dice(xs)
    y_pred_dice = keep_largest_component(argmax_2ch(y_pred_dice))
end;

# ╔═╡ 525e2404-9887-4347-843b-ef22c809c077
begin
	x1, x2, x3, x4 = xs[:, :, :, :, 1], xs[:, :, :, :, 2], xs[:, :, :, :, 3], xs[:, :, :, :, 4]
	x1, x2, x3, x4 = reshape(x1, size(x1)[1:end-1]), reshape(x2, size(x2)[1:end-1]), reshape(x3, size(x3)[1:end-1]), reshape(x4, size(x4)[1:end-1])
	y1, y2, y3, y4 = ys[:, :, :, 2, 1], ys[:, :, :, 2, 2], ys[:, :, :, 2, 3], ys[:, :, :, 2, 4]
	y1_pred, y2_pred, y3_pred, y4_pred = y_pred[:, :, :, 1], y_pred[:, :, :, 2], y_pred[:, :, :, 3], y_pred[:, :, :, 4]
	y1_pred_dice, y2_pred_dice, y3_pred_dice, y4_pred_dice = y_pred_dice[:, :, :, 1], y_pred_dice[:, :, :, 2], y_pred_dice[:, :, :, 3], y_pred_dice[:, :, :, 4]

	img_size = (512, 512, 112)
	x1, x2 = imresize(x1, img_size), imresize(x3, img_size)
	y1, y2 = round.(imresize(y1, img_size)), round.(imresize(y3, img_size))
	y1_pred, y2_pred = round.(imresize(y1_pred, img_size)), round.(imresize(y3_pred, img_size))
	y1_pred_dice, y2_pred_dice = round.(imresize(y1_pred_dice, img_size)), round.(imresize(y3_pred_dice, img_size))
end;

# ╔═╡ 91c3eddf-57b0-419d-a37c-c799b3cc6352
size(x1), size(y1), size(y1_pred), size(y1_pred_dice)

# ╔═╡ 59aafc4c-6b89-484d-a45f-8c94c577365b
md"""
## Visualize
"""

# ╔═╡ c1d410ff-c17e-4532-b1ad-777b242b3770
@bind a PlutoUI.Slider(axes(x2, 3), default=70, show_value=true)

# ╔═╡ f6a97124-dc51-437d-b352-94f46f3f4c22
let
	f = Figure(;resolution=(800, 600))
	alpha = 1
	markersize = 2
	
	y2_edge, y2_pred_edge, y2_pred_dice_edge = get_mask_edges(y2[:, :, a], y2_pred[:, :, a], y2_pred_dice[:, :, a])
	y2_edge, y2_pred_edge, y2_pred_dice_edge = Tuple.(y2_edge), Tuple.(y2_pred_edge), Tuple.(y2_pred_dice_edge)
	
    ax = Axis(
		f[1, 1],
		title = "Dice Loss"
	)
    heatmap!(x2[:, :, a]; colormap=:grays)
	scatter!(y2_edge; markersize=markersize, color = (:red, alpha), label="Ground Truth")
	scatter!(y2_pred_dice_edge; markersize=markersize, color = (:blue, alpha), label="Predicted")
	hidedecorations!(ax)

	ax = Axis(
		f[1, 2],
		title = "HD-Dice Loss"
	)
    heatmap!(x2[:, :, a]; colormap=:grays)
	scatter!(y2_edge; markersize=markersize, color = (:red, alpha), label="Ground Truth")
	scatter!(y2_pred_edge; markersize=markersize, color = (:blue, alpha), label="Predicted")
	hidedecorations!(ax)


	f
end

# ╔═╡ 4f43c115-20eb-4041-9f8a-40286a9fd7e5
let
	f = Figure(
	)
	alpha = 1
	markersize = 2

	slice_1 = 70
	y1_edge, y1_pred_edge, y1_pred_dice_edge = get_mask_edges(y1[:, :, slice_1], y1_pred[:, :, slice_1], y1_pred_dice[:, :, slice_1])
	y1_edge, y1_pred_edge, y1_pred_dice_edge = Tuple.(y1_edge), Tuple.(y1_pred_edge), Tuple.(y1_pred_dice_edge)
	
    ax1 = Axis(
		f[1, 1],
		title = "Dice Loss"
	)
    heatmap!(x1[:, :, slice_1]; colormap=:grays)
	scatter!(y1_edge; markersize=markersize, color = (:red, alpha), label="Ground Truth")
	scatter!(y1_pred_dice_edge; markersize=markersize, color = (:blue, alpha), label="Predicted")
	hidedecorations!(ax1)
	

	ax2 = Axis(
		f[1, 2],
		title = "HD-Dice Loss"
	)
    heatmap!(x1[:, :, slice_1]; colormap=:grays)
	scatter!(y1_edge; markersize=markersize, color = (:red, alpha), label="Ground Truth")
	scatter!(y1_pred_edge; markersize=markersize, color = (:blue, alpha), label="Predicted")
	hidedecorations!(ax2)


	slice_2 = 43
	y2_edge, y2_pred_edge, y2_pred_dice_edge = get_mask_edges(y2[:, :, slice_2], y2_pred[:, :, slice_2], y2_pred_dice[:, :, slice_2])
	y2_edge, y2_pred_edge, y2_pred_dice_edge = Tuple.(y2_edge), Tuple.(y2_pred_edge), Tuple.(y2_pred_dice_edge)

	ax3 = Axis(
		f[2, 1]
	)
    heatmap!(x2[:, :, slice_2]; colormap=:grays)
	scatter!(y2_edge; markersize=markersize, color = (:red, alpha), label="Ground Truth")
	scatter!(y2_pred_dice_edge; markersize=markersize, color = (:blue, alpha), label="Predicted")
	hidedecorations!(ax3)
	
	ax4 = Axis(
		f[2, 2]
	)
    heatmap!(x2[:, :, slice_2]; colormap=:grays)
	scatter!(y2_edge; markersize=markersize, color = (:red, alpha), label="Ground Truth")
	scatter!(y2_pred_edge; markersize=markersize, color = (:blue, alpha), label="Predicted")
	hidedecorations!(ax4)

	Legend(f[1:2, 3],
    [LineElement(color = :red, linestyle = nothing), LineElement(color = :blue, linestyle = nothing)],
    ["Ground Truth", "Predicted"]; framevisible=false)

	save(plotsdir("contours.png"), f)
	
    f
end

# ╔═╡ Cell order:
# ╠═5463443f-a69e-4766-bfc2-0b7ca0fd48c9
# ╠═6495cb7b-67ec-49da-bc5d-f7a63ca9e5e9
# ╠═ec608b5e-a6d5-4b12-a3a4-cc1738dbc6ed
# ╠═d1a12515-a9d0-468b-8978-dbb26a1ee667
# ╠═f88c7074-8b5c-4dc7-83d5-e65a24b88ab0
# ╠═278dfa0e-46e1-4789-9f51-eb3463a9fb00
# ╠═8608a972-73c0-4903-bdd7-9a23f7c57337
# ╟─95a525fe-e10b-42ee-a1b9-9405fe16b50c
# ╠═82134f7b-2cb1-4d9a-982f-f8b2799c9518
# ╠═bac3b79b-8b61-4bec-9478-fca3b4c901c5
# ╠═892bf5ef-012c-49c2-bb2d-80a006965c28
# ╠═6a77f8f4-5880-4267-ae44-0de8077bce1e
# ╠═bb891719-2676-49c9-8b23-dfdfc6f4fd3d
# ╠═6349d56d-f837-4395-9819-e8e7f34ba01f
# ╠═a0ea7e69-1249-4d47-9366-ec9f44568236
# ╠═5059495d-6905-4043-b03e-3ea73f67d341
# ╠═9c2b898e-0295-44da-8e3f-bab93524eec3
# ╠═7b34de9e-747b-4a80-870a-6f26178a378d
# ╟─1852f60f-cdf1-47d5-8d97-052710bed435
# ╠═fbda132c-1f19-4103-84f0-34215ce4e5b7
# ╟─8a054ee3-b388-4ea6-9b1a-3dbcc668714d
# ╠═ca48acf9-7546-40ab-ad69-c5a6ed3092b9
# ╟─24e42441-7d63-4c81-a901-5ebfaeb3e7a3
# ╠═622443e7-b90c-491a-beaf-d7aa668474ca
# ╟─810674a3-6be0-47c1-8b3f-ff9b385ccf77
# ╠═93afb259-43b6-4796-9fce-3d715538a47a
# ╠═29e64a2f-1eac-4be1-9d88-b18dcebe0b24
# ╟─2d8a4e37-6dd9-4e46-9a3d-bd8cd71f5f65
# ╟─a9b48266-a3b5-4734-9d35-f4484aed2e95
# ╟─5691a8ac-cb96-435f-aaef-48e4f72f77e5
# ╠═87dfe577-9ffa-4e89-9c2b-d2cde95fcbae
# ╠═23a06ef1-aff1-43b0-b328-e3e01e95efcf
# ╠═7d3aab45-0e21-4be4-8034-6f702025f1b6
# ╠═fa56d7f3-4e00-43d2-84bc-913c77c0fbe4
# ╟─29fbb7d4-e992-4b2b-a0af-5a31bcf6ba00
# ╠═2898cc85-d85e-46fe-988f-3843972fe7f1
# ╟─0eac4ad2-2484-4e2e-aebd-49f3b370555e
# ╠═bdbae682-29c5-4920-8983-0d445a153b31
# ╠═15316f17-c282-4618-967a-7539afe497fa
# ╠═40fbef10-8a8a-4f80-964c-882894742bbf
# ╟─9f20d955-a8cd-47bb-aed4-b0b50e915df5
# ╟─475618b0-885a-4fb3-90aa-57a7a649ddcb
# ╠═7e586270-1111-490a-849c-f41a5f55059c
# ╟─0e6fe7e8-0f10-4fef-86e2-4673011acf08
# ╠═0ef58a86-2901-40f0-9c62-9cbda7fbf03a
# ╟─b1622073-ffcd-41bd-b452-8abbbc4e0a33
# ╠═9a4223ed-f075-4f15-b2a8-f505e83fa253
# ╟─0614b6e9-9853-4f09-b8de-b3e8d1775aba
# ╟─4ace7533-de03-4a31-847e-69f30d7ce0b7
# ╠═c356a52f-88e1-4659-a122-56dc2fe9315f
# ╠═ece12891-9445-4a43-8201-9b99f20e2352
# ╠═1a4f5225-b905-4b87-af71-997817141460
# ╟─35af3308-ab47-4c94-ab87-a6d3ca09717f
# ╠═4325537e-47c3-45ae-a162-133eda8d03d4
# ╠═98cebb61-5fa4-4800-a734-ca6190cca5f0
# ╠═9ffc5a7b-97fc-46cc-9b2b-6aa89f060fee
# ╠═525e2404-9887-4347-843b-ef22c809c077
# ╠═91c3eddf-57b0-419d-a37c-c799b3cc6352
# ╟─59aafc4c-6b89-484d-a45f-8c94c577365b
# ╟─c1d410ff-c17e-4532-b1ad-777b242b3770
# ╠═f6a97124-dc51-437d-b352-94f46f3f4c22
# ╠═4f43c115-20eb-4041-9f8a-40286a9fd7e5
