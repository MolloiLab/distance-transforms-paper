### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 87ecdac4-0742-401c-b44e-25ab9660de68
# ╠═╡ show_logs = false
using DrWatson; @quickactivate "hd-loss"

# ╔═╡ 058e8772-e82d-4102-a05a-b1c64c4839b3
begin

    using PlutoUI, Statistics, CSV, DataFrames, GLM, CairoMakie, HypothesisTests, Colors, MLJBase, Glob, Flux, NIfTI, Images, ImageMorphology, FastAI, FastVision, StaticArrays, MLDataPattern, Printf
	using StatsBase: quantile!, rmsd, percentile
	using CairoMakie: Axis, Label
end

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

# ╔═╡ 9445d9fe-2155-4cd6-9f2e-d20d8719747a
begin
	df_dt_julia_2D = CSV.read(joinpath(path_julia, "dt_2D_min.csv"), DataFrame)
	df_dt_julia_3D = CSV.read(joinpath(path_julia, "dt_3D_min.csv"), DataFrame)

	df_dt_py_2D = CSV.read(joinpath(path_py, "purePython_DT_2D_Dec29.csv"), DataFrame)
	df_dt_py_2D = df_dt_py_2D[:, 2:end]
	df_dt_py_3D = CSV.read(joinpath(path_py, "purePython_DT_3D_Dec29.csv"), DataFrame)
	df_dt_py_3D = df_dt_py_3D[:, 2:end]
end;

# ╔═╡ c1659832-9e1a-4af5-9bf6-fd4d6f12589f
function dt()
    f = Figure()

    ##-- A --##
    ax1 = Axis(
		f[1, 1],
		xlabel = "Array Size(Pixels)",
		ylabel = "Time (ms)",
		title = "2D Distance Transform"
	)
    lines!(df_dt_julia_2D[!, :sizes_2D], df_dt_py_2D[!, :dt_min_cpu_2D] * 1e-6, label="DT Python (CPU)")
	lines!(df_dt_julia_2D[!, :sizes_2D], df_dt_julia_2D[!, :wenbo_threaded_minimum_2D] * 1e-6, label="DT Julia (CPU)")
    lines!(df_dt_julia_2D[!, :sizes_2D], df_dt_julia_2D[!, :wenbo_gpu_minimum_2D] * 1e-6, label="DT Julia (GPU)")

	##-- B --##
    ax2 = Axis(
		f[2, 1],
		xlabel = "Array Size(Pixels)",
		ylabel = "Time (s)",
		title = "3D Distance Transform"
	)
    lines!(df_dt_julia_3D[!, :sizes_3D], df_dt_py_3D[!, :dt_min_cpu_3D] * 1e-9, label="dt_py")
	lines!(df_dt_julia_3D[!, :sizes_3D], df_dt_julia_3D[!, :wenbo_threaded_minimum_3D] * 1e-9, label="dt_julia")
    lines!(df_dt_julia_3D[!, :sizes_3D], df_dt_julia_3D[!, :wenbo_gpu_minimum_3D] * 1e-9, label="dt_julia_gpu")

    ##-- LABELS --##
    f[1:2, 2] = Legend(f, ax1; framevisible=false)
    for (label, layout) in zip(["A", "B"], [f[1, 1], f[2, 1]])
        Label(
            layout[1, 1, TopLeft()],
            label;
            fontsize=25,
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
(x, y, z) = df_dt_py_2D[end, :dt_min_cpu_2D], df_dt_julia_2D[end, :wenbo_threaded_minimum_2D], df_dt_julia_2D[end, :wenbo_gpu_minimum_2D]

# ╔═╡ a0ea7e69-1249-4d47-9366-ec9f44568236
wenbo_gpu_vs_py = x / z

# ╔═╡ 5059495d-6905-4043-b03e-3ea73f67d341
wenbo_gpu_vs_wenbo_cpu = x / y

# ╔═╡ 1852f60f-cdf1-47d5-8d97-052710bed435
md"""
# Losses
"""

# ╔═╡ 96adfd06-9131-4f96-b49e-adac4e89bde4
begin
	df_loss_julia_2D = CSV.read(joinpath(path_julia, "HD_2D_Not_Loop_Jan_11.csv"), DataFrame)
	df_loss_julia_2D_loop = CSV.read(joinpath(path_julia, "HD_2D_In_Loop_Jan_11.csv"), DataFrame)
	
	df_loss_julia_3D = CSV.read(joinpath(path_julia, "HD_3D_Not_Loop_Jan_11.csv"), DataFrame)
	df_loss_julia_3D_loop = CSV.read(joinpath(path_julia, "HD_3D_In_Loop_Jan_11.csv"), DataFrame)

	df_loss_py_2D = CSV.read(joinpath(path_py, "HD_2D_Not_Loop_Jan_12.csv"), DataFrame)
	df_loss_py_3D = CSV.read(joinpath(path_py, "HD_3D_Not_Loop_Jan_12.csv"), DataFrame)
end;

# ╔═╡ b7236b7b-cf97-4962-b49d-58037b52201d
function loss()
    f = Figure()

	full_label1 = "HD Loss Julia CPU (DT: Mauer CPU, Loss: HD GPU)"
	full_label2 = "HD Loss Julia GPU (DT: Wenbo GPU, Loss: HD GPU)"

	##-- A --##
    ax1 = Axis(
		f[1, 1:2],
		xlabel = "Array Size (Pixels)",
		ylabel = "Time (ms)",
		title = "2D Loss",
		xticklabelrotation = pi/4
	)
	lines!(df_loss_julia_2D[!, :size], df_loss_py_2D[!, :Scipy_cpu_hd_gpu_min] * 1e-6, label="HD Loss Python (CPU)")
	lines!(df_loss_julia_2D[!, :size], df_loss_julia_2D[!, :Maurer_cpu_hd_gpu_min] * 1e-6, label="HD Loss Julia (CPU)")
	lines!(df_loss_julia_2D[!, :size], df_loss_julia_2D[!, :Wenbo_gpu_hd_gpu_min] * 1e-6, label="HD Loss Julia (GPU)")

	##-- B --##
    ax2 = Axis(
		f[2, 1:2],
		xlabel = "Array Size (Voxels)",
		ylabel = "Time (ms)",
		title = "3D Loss",
		xticklabelrotation = pi/4
	)
	lines!(df_loss_julia_3D[!, :size], df_loss_py_3D[!, :Scipy_cpu_hd_gpu_min] * 1e-6, label="HD Loss Python (CPU)")
	lines!(df_loss_julia_3D[!, :size], df_loss_julia_3D[!, :Maurer_cpu_hd_gpu_min] * 1e-6, label="HD Loss Julia (CPU)")
	lines!(df_loss_julia_3D[!, :size], df_loss_julia_3D[!, :Wenbo_gpu_hd_gpu_min] * 1e-6, label="HD Loss Julia (GPU)")
	vlines!(96^3:96^3; linestyle=:dash, label="Training Array Input Size")

   	##-- LABELS --##
    f[1:2, end+1] = Legend(f, ax2; framevisible=false, orientation=:vertical)
    for (label, layout) in zip(["A", "B"], [f[1, 1], f[2, 1]])
        Label(
            layout[1, 1, TopLeft()],
            label;
            fontsize=25,
            padding=(0, 0, 40, 0),
            halign=:right,
        )
    end

	save(plotsdir("loss.png"), f)
    return f
end

# ╔═╡ 7e3ac952-005a-45e6-a395-eea36ba255b5
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
    ax1 = Axis(f[1, 1]; xticks=(1:3, ["Dice Loss", "HD-Dice Loss (CPU)", "HD-Dice Loss (GPU)"]), xticklabelrotation=pi/4)

    table = [1, 2, 3]
	h1 = step_julia_dice
	h2 = step_julia_hd_cpu
	h3 = step_julia_hd_gpu
	heights1 = [h1, h2, h3]
	l1 = @sprintf "%.2f" h1
	l2 = @sprintf "%.2f" h2
	l3 = @sprintf "%.2f" h3
    barplot!(table, heights1; color=colors[1:3], bar_labels=[l1, l2, l3])

    ax1.title = "Average Step Time"
    ax1.ylabel = "Time (s)"
	ylims!(ax1; low=0, high=1)
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
     ax1 = Axis(f[1, 1]; xticks=(1:3, ["Dice Loss", "HD-Dice Loss (CPU)", "HD-Dice Loss (GPU)"]), xticklabelrotation=pi/4)

    table = [1, 2, 3]
	h1 = epoch_julia_dice
	h2 = epoch_julia_hd_cpu
	h3 = epoch_julia_hd_gpu
	heights1 = [h1, h2, h3]
	l1 = @sprintf "%.2f" h1
	l2 = @sprintf "%.2f" h2
	l3 = @sprintf "%.2f" h3
    barplot!(table, heights1; color=colors[1:3], bar_labels=[l1, l2, l3])


    ax1.title = "Average Epoch Time"
    ax1.ylabel = "Time (s)"
	ylims!(ax1; low=0, high=5)

	save(plotsdir("training_epoch.png"))

    return f
end

# ╔═╡ fa56d7f3-4e00-43d2-84bc-913c77c0fbe4
with_theme(training_epoch, medphys_theme)

# ╔═╡ 29fbb7d4-e992-4b2b-a0af-5a31bcf6ba00
md"""
# Accuracy
"""

# ╔═╡ 2898cc85-d85e-46fe-988f-3843972fe7f1


# ╔═╡ 0eac4ad2-2484-4e2e-aebd-49f3b370555e


# ╔═╡ bdbae682-29c5-4920-8983-0d445a153b31


# ╔═╡ 15316f17-c282-4618-967a-7539afe497fa


# ╔═╡ 40fbef10-8a8a-4f80-964c-882894742bbf


# ╔═╡ 9f20d955-a8cd-47bb-aed4-b0b50e915df5


# ╔═╡ 475618b0-885a-4fb3-90aa-57a7a649ddcb


# ╔═╡ b8ea2ac2-2945-4ad1-80c4-07c5d6f4004a


# ╔═╡ 0e6fe7e8-0f10-4fef-86e2-4673011acf08


# ╔═╡ 0ef58a86-2901-40f0-9c62-9cbda7fbf03a


# ╔═╡ b1622073-ffcd-41bd-b452-8abbbc4e0a33


# ╔═╡ 9a4223ed-f075-4f15-b2a8-f505e83fa253


# ╔═╡ 0614b6e9-9853-4f09-b8de-b3e8d1775aba


# ╔═╡ 3398ce62-907e-4807-9149-8488e9f8c3c2


# ╔═╡ 5b2f3378-1a2a-4a39-918b-9aea79ae1940


# ╔═╡ 4ace7533-de03-4a31-847e-69f30d7ce0b7


# ╔═╡ ece12891-9445-4a43-8201-9b99f20e2352


# ╔═╡ 35af3308-ab47-4c94-ab87-a6d3ca09717f


# ╔═╡ 2a97f5a9-e236-4fc7-a39a-998b92ef066e


# ╔═╡ 7ea58ac8-344f-481c-9864-4f9879698581


# ╔═╡ 4325537e-47c3-45ae-a162-133eda8d03d4


# ╔═╡ 98cebb61-5fa4-4800-a734-ca6190cca5f0


# ╔═╡ 9ffc5a7b-97fc-46cc-9b2b-6aa89f060fee


# ╔═╡ 525e2404-9887-4347-843b-ef22c809c077


# ╔═╡ 91c3eddf-57b0-419d-a37c-c799b3cc6352


# ╔═╡ 9178756e-a277-41ae-bcf0-fd08a5f62059


# ╔═╡ 975ce131-e5b8-4ce4-84e6-bf259c750796


# ╔═╡ 59aafc4c-6b89-484d-a45f-8c94c577365b


# ╔═╡ 4f43c115-20eb-4041-9f8a-40286a9fd7e5


# ╔═╡ 1d43bb6c-26e8-4524-95f8-4fb9bd06fe7f


# ╔═╡ Cell order:
# ╠═87ecdac4-0742-401c-b44e-25ab9660de68
# ╠═058e8772-e82d-4102-a05a-b1c64c4839b3
# ╠═278dfa0e-46e1-4789-9f51-eb3463a9fb00
# ╠═8608a972-73c0-4903-bdd7-9a23f7c57337
# ╠═02f182d4-f5ab-46a3-8009-33ed641fcf27
# ╟─95a525fe-e10b-42ee-a1b9-9405fe16b50c
# ╠═9445d9fe-2155-4cd6-9f2e-d20d8719747a
# ╟─c1659832-9e1a-4af5-9bf6-fd4d6f12589f
# ╠═aa312920-1c67-46f7-a7b1-dfe42f54c769
# ╠═6349d56d-f837-4395-9819-e8e7f34ba01f
# ╠═a0ea7e69-1249-4d47-9366-ec9f44568236
# ╠═5059495d-6905-4043-b03e-3ea73f67d341
# ╟─1852f60f-cdf1-47d5-8d97-052710bed435
# ╠═96adfd06-9131-4f96-b49e-adac4e89bde4
# ╟─b7236b7b-cf97-4962-b49d-58037b52201d
# ╠═7e3ac952-005a-45e6-a395-eea36ba255b5
# ╟─24e42441-7d63-4c81-a901-5ebfaeb3e7a3
# ╠═622443e7-b90c-491a-beaf-d7aa668474ca
# ╟─810674a3-6be0-47c1-8b3f-ff9b385ccf77
# ╠═93afb259-43b6-4796-9fce-3d715538a47a
# ╠═29e64a2f-1eac-4be1-9d88-b18dcebe0b24
# ╟─2d8a4e37-6dd9-4e46-9a3d-bd8cd71f5f65
# ╠═a9b48266-a3b5-4734-9d35-f4484aed2e95
# ╟─5691a8ac-cb96-435f-aaef-48e4f72f77e5
# ╠═87dfe577-9ffa-4e89-9c2b-d2cde95fcbae
# ╠═23a06ef1-aff1-43b0-b328-e3e01e95efcf
# ╟─7d3aab45-0e21-4be4-8034-6f702025f1b6
# ╠═fa56d7f3-4e00-43d2-84bc-913c77c0fbe4
# ╟─29fbb7d4-e992-4b2b-a0af-5a31bcf6ba00
# ╠═2898cc85-d85e-46fe-988f-3843972fe7f1
# ╠═0eac4ad2-2484-4e2e-aebd-49f3b370555e
# ╠═bdbae682-29c5-4920-8983-0d445a153b31
# ╠═15316f17-c282-4618-967a-7539afe497fa
# ╠═40fbef10-8a8a-4f80-964c-882894742bbf
# ╠═9f20d955-a8cd-47bb-aed4-b0b50e915df5
# ╠═475618b0-885a-4fb3-90aa-57a7a649ddcb
# ╠═b8ea2ac2-2945-4ad1-80c4-07c5d6f4004a
# ╠═0e6fe7e8-0f10-4fef-86e2-4673011acf08
# ╠═0ef58a86-2901-40f0-9c62-9cbda7fbf03a
# ╠═b1622073-ffcd-41bd-b452-8abbbc4e0a33
# ╠═9a4223ed-f075-4f15-b2a8-f505e83fa253
# ╠═0614b6e9-9853-4f09-b8de-b3e8d1775aba
# ╠═3398ce62-907e-4807-9149-8488e9f8c3c2
# ╠═5b2f3378-1a2a-4a39-918b-9aea79ae1940
# ╠═4ace7533-de03-4a31-847e-69f30d7ce0b7
# ╠═ece12891-9445-4a43-8201-9b99f20e2352
# ╠═35af3308-ab47-4c94-ab87-a6d3ca09717f
# ╠═2a97f5a9-e236-4fc7-a39a-998b92ef066e
# ╠═7ea58ac8-344f-481c-9864-4f9879698581
# ╠═4325537e-47c3-45ae-a162-133eda8d03d4
# ╠═98cebb61-5fa4-4800-a734-ca6190cca5f0
# ╠═9ffc5a7b-97fc-46cc-9b2b-6aa89f060fee
# ╠═525e2404-9887-4347-843b-ef22c809c077
# ╠═91c3eddf-57b0-419d-a37c-c799b3cc6352
# ╠═9178756e-a277-41ae-bcf0-fd08a5f62059
# ╠═975ce131-e5b8-4ce4-84e6-bf259c750796
# ╠═59aafc4c-6b89-484d-a45f-8c94c577365b
# ╠═4f43c115-20eb-4041-9f8a-40286a9fd7e5
# ╠═1d43bb6c-26e8-4524-95f8-4fb9bd06fe7f
