### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 0ff50fcc-a71a-403d-9e62-458125ac6dc4
using Pkg; Pkg.activate(".."), Pkg.instantiate()

# ╔═╡ 5abcd30b-bc3e-4d15-97be-6d53f5cc6481
# ╠═╡ show_logs = false
begin
	using CUDA
	# CUDA.set_runtime_version!(v"11.8")
end

# ╔═╡ 6972d6e6-8a28-11ee-223b-8fe3cb023e45
using DrWatson: datadir

# ╔═╡ 5eaabc58-aa01-419a-bfa1-7cc3eb96d8d1
# ╠═╡ show_logs = false
using PlutoUI, BenchmarkTools, CairoMakie, DataFrames, CSV, Statistics

# ╔═╡ 07dc4f9b-7e39-4d76-88e6-3c32068c99b0
# ╠═╡ show_logs = false
using Lux, LuxCUDA, Random, Optimisers, Zygote, MLUtils

# ╔═╡ 05d00e65-b22d-4ba9-83e7-3f53647080e7
using ChainRulesCore: ignore_derivatives

# ╔═╡ 5dc05f88-8672-427b-ab52-d92358f4151c
using Losers: dice_loss, hausdorff_loss

# ╔═╡ 514a75f3-decc-46d0-b9b4-df971631f564
using ImageMorphology: distance_transform, feature_transform

# ╔═╡ 85ae8235-f118-4c78-a442-257dfa49e729
using DistanceTransforms: transform, boolean_indicator

# ╔═╡ d7959ac3-4366-41f4-9487-a2b9bc1f65d2
TableOfContents()

# ╔═╡ e1189b71-b938-414a-bb44-8d8d472ffe1b
md"""
# Hausdorff Loss Benchmarks
"""

# ╔═╡ 58b5e833-0754-4b1c-816f-cf54e8735f50
md"""
## 2D
"""

# ╔═╡ b6e1f4aa-c732-41b1-a296-fd8b3ab29443
begin
	sizes_hd = Float64[]

	dice = Float64[]
	dice_std = Float64[]

	hd_maurer = Float64[]
	hd_maurer_std = Float64[]

	hd_fenz = Float64[]
	hd_fenz_std = Float64[]

	hd_proposed = Float64[]
	hd_proposed_std = Float64[]
	
	for _n in range_size_2D
		n = round(Int, _n)
		@info n
		push!(sizes_hd, n^2)
		f = Float32.(rand([0, 1], n, n))
		f2 = Float32.(rand([0, 1], n, n))

		# Dice Loss (Losers.jl)
		loss = @benchmark dice_loss(
			$f, 
			$f2
		)
		push!(dice, BenchmarkTools.minimum(loss).time)
		push!(dice_std, BenchmarkTools.std(loss).time)

		# Maurer Hausdorff Loss (ImageMorphology.jl)
		f_bool, f2_bool = Bool.(f), Bool.(f2)
		loss = @benchmark hausdorff_loss(
			$f_bool, 
			$f2_bool, 
			$distance_transform($feature_transform($f_bool)),
			$distance_transform($feature_transform($f2_bool)),
		)
		push!(hd_maurer, BenchmarkTools.minimum(loss).time)
		push!(hd_maurer_std, BenchmarkTools.std(loss).time)
		
		# Felzenszwalb Hausdorff Loss (DistanceTransforms.jl)
		loss = @benchmark hausdorff_loss(
			$f, 
			$f2, 
			$transform($boolean_indicator($f)), 
			$transform($boolean_indicator($f))
		)
		push!(hd_fenz, BenchmarkTools.minimum(loss).time)
		push!(hd_fenz_std, BenchmarkTools.std(loss).time)

		# Proposed Hausdorff Loss (DistanceTransforms.jl)
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			f2_cuda = CuArray(f2)
			loss = @benchmark hausdorff_loss(
				$f_cuda, 
				$f2_cuda, 
				$transform($boolean_indicator($f_cuda)), 
				$transform($boolean_indicator($f2_cuda))
			)
			push!(hd_proposed, BenchmarkTools.minimum(loss).time)
			push!(hd_proposed_std, BenchmarkTools.std(loss).time)
		end
	end
end

# ╔═╡ 2f2b9c3c-85c0-406e-88fd-774244956147
md"""
## 3D
"""

# ╔═╡ 53c3ea64-a4e4-4dd4-af87-9a62bdebf5e7
begin
	sizes_hd_3D = Float64[]

	dice_3D = Float64[]
	dice_std_3D = Float64[]
	
	hd_maurer_3D = Float64[]
	hd_maurer_std_3D = Float64[]

	hd_fenz_3D = Float64[]
	hd_fenz_std_3D = Float64[]

	hd_proposed_3D = Float64[]
	hd_proposed_std_3D = Float64[]
	
	for _n in range_size_3D
		n = round(Int, _n)
		@info n
		push!(sizes_hd_3D, n^2)
		f = Float32.(rand([0, 1], n, n, n))
		f2 = Float32.(rand([0, 1], n, n, n))

		# Dice Loss (Losers.jl)
		loss = @benchmark dice_loss(
			$f, 
			$f2
		)
		push!(dice_3D, BenchmarkTools.minimum(loss).time)
		push!(dice_std_3D, BenchmarkTools.std(loss).time)

		# Maurer Hausdorff Loss (ImageMorphology.jl)
		f_bool, f2_bool = Bool.(f), Bool.(f2)
		loss = @benchmark hausdorff_loss(
			$f_bool, 
			$f2_bool, 
			$distance_transform($feature_transform($f_bool)),
			$distance_transform($feature_transform($f2_bool)),
		)
		push!(hd_maurer_3D, BenchmarkTools.minimum(loss).time)
		push!(hd_maurer_std_3D, BenchmarkTools.std(loss).time)
		
		# Felzenszwalb Hausdorff Loss (DistanceTransforms.jl)
		loss = @benchmark hausdorff_loss(
			$f, 
			$f2, 
			$transform($boolean_indicator($f)), 
			$transform($boolean_indicator($f))
		)
		push!(hd_fenz_3D, BenchmarkTools.minimum(loss).time)
		push!(hd_fenz_std_3D, BenchmarkTools.std(loss).time)

		# Felzenszwalb GPU 
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			f2_cuda = CuArray(f2)
			loss = @benchmark hausdorff_loss(
				$f_cuda, 
				$f2_cuda, 
				$transform($boolean_indicator($f_cuda)), 
				$transform($boolean_indicator($f2_cuda))
			)
			push!(hd_proposed_3D, BenchmarkTools.minimum(loss).time)
			push!(hd_proposed_std_3D, BenchmarkTools.std(loss).time)
		end
	end
end

# ╔═╡ 818224dc-7527-4179-a202-4c5669f8ab91
let
	f = CairoMakie.Figure()
	ax = Axis(
		f[1, 1],
		title="2D"
	)
	scatterlines!(dice, label="Dice Loss")
	scatterlines!(hd_maurer, label="Maurer Hausdorff Loss")
	scatterlines!(hd_fenz, label="Felzenszwalb Hausdorff Loss")
	scatterlines!(hd_proposed, label="Proposed Hausdorff Loss")
	

	ax = Axis(
		f[2, 1],
		title="3D"
	)
	scatterlines!(dice_3D, label="Dice Loss")
	scatterlines!(hd_maurer_3D, label="Maurer Hausdorff Loss")
	scatterlines!(hd_fenz_3D, label="Felzenszwalb Hausdorff Loss")
	scatterlines!(hd_proposed_3D, label="Proposed Hausdorff Loss")

	f[1:2, 2] = Legend(f, ax, "Loss Functions", framevisible = false)

	Label(f[0, 1:2]; text="Loss Functions", fontsize=30)
	
	f
end

# ╔═╡ be13f136-eeb7-49cc-82f1-95dd00ba6641
begin
	global df_loss
	if has_cuda_gpu()
		df_loss = DataFrame(
			sizes_hd = sizes_hd,
			dice = dice,
			dice_std = dice_std,
			hd_maurer = hd_maurer,
			hd_maurer_std = hd_maurer_std,
			hd_fenz = hd_fenz,
			hd_fenz_std = hd_fenz_std,
			hd_proposed = hd_proposed,
			hd_proposed_std = hd_proposed_std,
		
			sizes_hd_3D = sizes_hd_3D,
			dice_3D = dice_3D,
			dice_std_3D = dice_std_3D,
			hd_maurer_3D = hd_maurer_3D,
			hd_maurer_std_3D = hd_maurer_std_3D,
			hd_fenz_3D = hd_fenz_3D,
			hd_fenz_std_3D = hd_fenz_std_3D,
			hd_proposed_3D = hd_proposed_3D,
			hd_proposed_std_3D = hd_proposed_std_3D,
		)
	else
		df_loss = DataFrame(
			sizes_hd = sizes_hd,
			dice = dice,
			dice_std = dice_std,
			hd_maurer = hd_maurer,
			hd_maurer_std = hd_maurer_std,
			hd_fenz = hd_fenz,
			hd_fenz_std = hd_fenz_std,
		
			sizes_hd_3D = sizes_hd_3D,
			dice_3D = dice_3D,
			dice_std_3D = dice_std_3D,
			hd_maurer_3D = hd_maurer_3D,
			hd_maurer_std_3D = hd_maurer_std_3D,
			hd_fenz_3D = hd_fenz_3D,
			hd_fenz_std_3D = hd_fenz_std_3D,
		)
	end
end

# ╔═╡ e3f95955-9ccb-42af-b2c5-94a3525fb627
save(datadir("results", "losses.csv"), df_loss)

# ╔═╡ 4ccc2224-a361-4647-9dd6-bc598cc86bb0
md"""
# Training Loop Setup
"""

# ╔═╡ 1b758e7b-df5c-4fc0-b534-ef3f76b5a59a
md"""
## Randomness
Initialize the random number generator
"""

# ╔═╡ c6dec210-71d4-4f07-b5d1-93af2dc2d6e3
begin
	rng = MersenneTwister()
	Random.seed!(rng, 12345)
end;

# ╔═╡ 99abc39a-6f4d-4f99-8380-682a98d56b8e
md"""
## Model
"""

# ╔═╡ 6b0c097a-2025-4c72-b051-4fb1053cb02d
function conv_layer(
	k, in_channels, out_channels;
	pad=2, stride=1, activation=relu)
	
    return Chain(
        Conv((k, k, k), in_channels => out_channels, pad=pad, stride=stride),
        BatchNorm(out_channels),
        WrappedFunction(activation)
    )
end

# ╔═╡ 9843b780-883c-40a0-883e-e11aa8d0422d
function contract_block(
	in_channels, mid_channels, out_channels;
	k=5, stride=2, activation=relu)
	
    return Chain(
        conv_layer(k, in_channels, mid_channels),
        conv_layer(k, mid_channels, out_channels),
        Chain(
            Conv((2, 2, 2), out_channels => out_channels, stride=stride),
            BatchNorm(out_channels),
            WrappedFunction(activation)
        )
    )
end

# ╔═╡ f1653109-b8be-465e-8119-639bba2bd0de
function expand_block(
	in_channels, mid_channels, out_channels;
	k=5, stride=2, activation=relu)
	
    return Chain(
        conv_layer(k, in_channels, mid_channels),
        conv_layer(k, mid_channels, out_channels),
        Chain(
            ConvTranspose((2, 2, 2), out_channels => out_channels, stride=stride),
            BatchNorm(out_channels),
            WrappedFunction(activation)
        )
    )
end

# ╔═╡ 9f5c2d53-1054-4b23-ade4-3bcabe906cad
function Unet(in_channels::Int = 1, out_channels::Int = in_channels)
    return Chain(
        # Initial Convolution Layer
        conv_layer(5, in_channels, 4),

        # Contracting Path
        contract_block(4, 8, 8),
        contract_block(8, 16, 16),
        contract_block(16, 32, 32),
        contract_block(32, 64, 64),

        # Bottleneck Layer
        conv_layer(5, 64, 128),

        # Expanding Path
        expand_block(128, 64, 64),
        expand_block(64, 32, 32),
        expand_block(32, 16, 16),
        expand_block(16, 8, 8),

        # Final Convolution Layer
        Conv((1, 1, 1), 8 => out_channels)
    )
end

# ╔═╡ 09fca34b-44a3-4918-8f6b-5a405b26ec33
model = Unet(1, 2)

# ╔═╡ fa520196-ea57-4c25-9445-bdabbeab9a82
ps, st = Lux.setup(rng, model)

# ╔═╡ 4c0d4d36-dccb-420f-a383-c39a64e1b42d
md"""
## Optimizer
"""

# ╔═╡ 43748430-84c2-4915-a7af-3df50b1cc48a
function create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
    return Optimisers.setup(opt, ps)
end

# ╔═╡ f71d9fac-345e-4031-8443-5c4cf54bcaf1
md"""
## Loss Functions
"""

# ╔═╡ 8bf8ed10-3170-4404-8916-ba129efba390
function compute_loss_dice(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
	local loss
	for b in axes(y, 5)
		_y_pred = copy(round.(y_pred[:, :, :, 1, b]))
		_y = copy(y[:, :, :, 1, b])
		loss = dice_loss(_y_pred, _y)
	end
    return loss, y_pred, st
end

# ╔═╡ 5cb8b16e-9d0d-46a7-813c-344a29a8c4b6
function compute_loss_hd_proposed(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
	local loss
	for b in axes(y, 5)
		_y_pred = copy(round.(y_pred[:, :, :, 1, b]))
		_y = copy(y[:, :, :, 1, b])

		local _y_dtm, _y_pred_dtm
		ignore_derivatives() do
			_y_dtm = transform(boolean_indicator(_y))
			_y_pred_dtm = transform(boolean_indicator(_y_pred))
		end
		hd = hausdorff_loss(_y_pred, _y, _y_pred_dtm, _y_dtm)
		
		dsc = dice_loss(_y_pred, _y)
		
		loss = hd + dsc
	end
    return loss, y_pred, st
end

# ╔═╡ a4c06559-0bc8-4547-afc5-a5d320a981be
function compute_loss_hd_fenz(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
	
	y = y |> cpu_device()
	y_pred = y_pred |> cpu_device()
	
	local loss
	for b in axes(y, 5)
		_y_pred = copy(round.(y_pred[:, :, :, 1, b]))
		_y = copy(y[:, :, :, 1, b])

		local _y_dtm, _y_pred_dtm
		ignore_derivatives() do
			_y_dtm = transform(boolean_indicator(_y))
			_y_pred_dtm = transform(boolean_indicator(_y_pred))
		end
		hd = hausdorff_loss(_y_pred, _y, _y_pred_dtm, _y_dtm)
		
		dsc = dice_loss(_y_pred, _y)
		
		loss = hd + dsc
	end
	
	y = y |> gpu_device()
	y_pred = y_pred |> gpu_device()
	
    return loss, y_pred, st
end

# ╔═╡ f3697f64-3d5a-4ace-80c8-54d646cc57e4
function compute_loss_hd_maurer(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
	
	y = y |> cpu_device()
	y_pred = y_pred |> cpu_device()
	
	local loss
	for b in axes(y, 5)
		_y_pred = copy(round.(y_pred[:, :, :, 1, b]))
		_y = copy(y[:, :, :, 1, b])

		local _y_dtm, _y_pred_dtm
		ignore_derivatives() do
			_y_dtm = distance_transform(feature_transform(Bool.(_y)))
			_y_pred_dtm = distance_transform(feature_transform(Bool.(_y_pred)))
		end
		hd = hausdorff_loss(_y_pred, _y, _y_pred_dtm, _y_dtm)
		
		dsc = dice_loss(_y_pred, _y)
		
		loss = hd + dsc
	end
	
	y = y |> gpu_device()
	y_pred = y_pred |> gpu_device()
	
    return loss, y_pred, st
end

# ╔═╡ dbde99bc-c062-4c13-b556-1f2dbee1cc41
arr = rand(10, 10) .* 2

# ╔═╡ 96fb1c88-7397-4e1c-80c5-4ce796a85514
md"""
# Training Benchmarks
"""

# ╔═╡ 93f973db-f038-4c72-857a-a50df78a1e7e
function run_epoch!(train_loader, dev, compute_loss, model, ps, st, opt_state)
	# Train the model
	for (x, y) in train_loader
		x = x |> dev
		y = y |> dev
		(loss, y_pred, st), back = pullback(
			compute_loss, x, y, model, ps, st
		)
		gs = back((one(loss), nothing, nothing))[4]
		opt_state, ps = Optimisers.update(opt_state, ps, gs)
	end
end

# ╔═╡ 4753fc5d-616d-4f43-9d3c-032d882dc86f
function time_epoch!(model, ps, st, train_loader, compute_loss)
    dev = gpu_device()
    ps = ps |> dev
    st = st |> dev

    # Create the optimiser
    opt_state = create_optimiser(ps)
	bench = @benchmark run_epoch!($train_loader, $dev, $compute_loss, $model, $ps, $st, $opt_state)
	return bench
end

# ╔═╡ eab39035-c7f0-43fb-a6a3-ab5b004d3153
md"""
## Benchmarks
"""

# ╔═╡ 9bc6158c-0d78-445d-a4d0-ee2133ea7252
md"""
## Dataset
"""

# ╔═╡ 7664bf01-e5dd-49d5-9b64-56f4b8efc594
img_sizes = [64, 96, 128, 256]
# img_sizes = [64, 96, 128]

# ╔═╡ bf00f5b9-311d-4f83-ac97-e5e5fe55b346
begin
	channels = 1
	num_imgs = 5
	batch_size = 4
end

# ╔═╡ 0efd5aad-baac-4654-a7f9-216c2d5493e0
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	sizes = Float64[]

	ep_dice = Float64[]
	ep_dice_std = Float64[]

	ep_maurer = Float64[]
	ep_maurer_std = Float64[]

	ep_fenz = Float64[]
	ep_fenz_std = Float64[]

	ep_proposed = Float64[]
	ep_proposed_std = Float64[]
	
	for _n in img_sizes
		n = round(Int, _n)
		@info n
		push!(sizes, n^3)

		# Data creation
		img_size = (n, n, n)
		img_data = rand(Float32, img_size..., channels, num_imgs)
		lbl_data = rand([0f0, 1f0], img_size..., channels, num_imgs)
		(img_train, lbl_train), (img_val, lbl_val) = splitobs((img_data, lbl_data); at=0.8, shuffle=true)
		train_loader = DataLoader(collect.((img_train, lbl_train)); batchsize=batch_size, shuffle=true)

		# Dice
		ep = time_epoch!(model, ps, st, train_loader, compute_loss_dice)
		push!(ep_dice, BenchmarkTools.minimum(ep).time)
		push!(ep_dice_std, BenchmarkTools.std(ep).time)

		# Maurer (ImageMorphology.jl)
		ep = time_epoch!(
			model, 
			ps, 
			st, 
			train_loader, 
			compute_loss_hd_maurer
		)
		push!(ep_maurer, BenchmarkTools.minimum(ep).time)
		push!(ep_maurer_std, BenchmarkTools.std(ep).time)
		
		# Felzenszwalb (DistanceTransforms.jl)
		ep = time_epoch!(
			model, 
			ps, 
			st, 
			train_loader, 
			compute_loss_hd_fenz
		)
		push!(ep_fenz, BenchmarkTools.minimum(ep).time)
		push!(ep_fenz_std, BenchmarkTools.std(ep).time)

		# Proposed-GPU (DistanceTransforms.jl)
		ep = time_epoch!(
			model, 
			ps, 
			st, 
			train_loader, 
			compute_loss_hd_proposed
		)
		push!(ep_proposed, BenchmarkTools.minimum(ep).time)
		push!(ep_proposed_std, BenchmarkTools.std(ep).time)
	end
end
  ╠═╡ =#

# ╔═╡ 4558cee9-c9bf-4f28-a7a6-bd3179edab3b
#=╠═╡
let
	f = CairoMakie.Figure()
	ax = Axis(
		f[1, 1],
		title="2D"
	)

	scatterlines!(ep_dice, label="Dice Loss")
	scatterlines!(ep_maurer, label="Hausdorff Loss Maurer")
	scatterlines!(ep_fenz, label="Hausdorff Loss Felzenswalb")
	scatterlines!(ep_proposed, label="Hausdorff Loss Proposed")

	f[1, 2] = Legend(f, ax, "Loss Function", framevisible = false)

	Label(f[0, 1]; text="Minimum Epoch Time", fontsize=30)
	f
end
  ╠═╡ =#

# ╔═╡ 83aa0bf6-ed72-4aad-9976-45a89dd24a05
#=╠═╡
df_ep = DataFrame(
	batch_size = batch_size,
	sizes = sizes,
	ep_dice = ep_dice,
	ep_dice_std = ep_dice_std,
	ep_maurer = ep_maurer,
	ep_maurer_std = ep_maurer_std,
	ep_fenz = ep_fenz,
	ep_fenz_std = ep_fenz_std,
	ep_proposed = ep_proposed,
	ep_proposed_std = ep_proposed_std,
)
  ╠═╡ =#

# ╔═╡ b0ad91db-660e-4cda-b229-a40fa987589c
#=╠═╡
save(datadir("results", "epochs.csv"), df_ep)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═0ff50fcc-a71a-403d-9e62-458125ac6dc4
# ╠═5abcd30b-bc3e-4d15-97be-6d53f5cc6481
# ╠═6972d6e6-8a28-11ee-223b-8fe3cb023e45
# ╠═5eaabc58-aa01-419a-bfa1-7cc3eb96d8d1
# ╠═07dc4f9b-7e39-4d76-88e6-3c32068c99b0
# ╠═05d00e65-b22d-4ba9-83e7-3f53647080e7
# ╠═5dc05f88-8672-427b-ab52-d92358f4151c
# ╠═514a75f3-decc-46d0-b9b4-df971631f564
# ╠═85ae8235-f118-4c78-a442-257dfa49e729
# ╠═d7959ac3-4366-41f4-9487-a2b9bc1f65d2
# ╟─e1189b71-b938-414a-bb44-8d8d472ffe1b
# ╟─58b5e833-0754-4b1c-816f-cf54e8735f50
# ╠═b6e1f4aa-c732-41b1-a296-fd8b3ab29443
# ╟─2f2b9c3c-85c0-406e-88fd-774244956147
# ╠═53c3ea64-a4e4-4dd4-af87-9a62bdebf5e7
# ╠═818224dc-7527-4179-a202-4c5669f8ab91
# ╠═be13f136-eeb7-49cc-82f1-95dd00ba6641
# ╠═e3f95955-9ccb-42af-b2c5-94a3525fb627
# ╟─4ccc2224-a361-4647-9dd6-bc598cc86bb0
# ╟─1b758e7b-df5c-4fc0-b534-ef3f76b5a59a
# ╠═c6dec210-71d4-4f07-b5d1-93af2dc2d6e3
# ╟─99abc39a-6f4d-4f99-8380-682a98d56b8e
# ╠═6b0c097a-2025-4c72-b051-4fb1053cb02d
# ╠═9843b780-883c-40a0-883e-e11aa8d0422d
# ╠═f1653109-b8be-465e-8119-639bba2bd0de
# ╠═9f5c2d53-1054-4b23-ade4-3bcabe906cad
# ╠═09fca34b-44a3-4918-8f6b-5a405b26ec33
# ╠═fa520196-ea57-4c25-9445-bdabbeab9a82
# ╟─4c0d4d36-dccb-420f-a383-c39a64e1b42d
# ╠═43748430-84c2-4915-a7af-3df50b1cc48a
# ╟─f71d9fac-345e-4031-8443-5c4cf54bcaf1
# ╠═8bf8ed10-3170-4404-8916-ba129efba390
# ╠═5cb8b16e-9d0d-46a7-813c-344a29a8c4b6
# ╠═a4c06559-0bc8-4547-afc5-a5d320a981be
# ╠═f3697f64-3d5a-4ace-80c8-54d646cc57e4
# ╠═dbde99bc-c062-4c13-b556-1f2dbee1cc41
# ╟─96fb1c88-7397-4e1c-80c5-4ce796a85514
# ╠═93f973db-f038-4c72-857a-a50df78a1e7e
# ╠═4753fc5d-616d-4f43-9d3c-032d882dc86f
# ╟─eab39035-c7f0-43fb-a6a3-ab5b004d3153
# ╟─9bc6158c-0d78-445d-a4d0-ee2133ea7252
# ╠═7664bf01-e5dd-49d5-9b64-56f4b8efc594
# ╠═bf00f5b9-311d-4f83-ac97-e5e5fe55b346
# ╠═0efd5aad-baac-4654-a7f9-216c2d5493e0
# ╟─4558cee9-c9bf-4f28-a7a6-bd3179edab3b
# ╠═83aa0bf6-ed72-4aad-9976-45a89dd24a05
# ╠═b0ad91db-660e-4cda-b229-a40fa987589c
