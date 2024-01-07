### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 0ff50fcc-a71a-403d-9e62-458125ac6dc4
using Pkg; Pkg.instantiate()

# ╔═╡ 6972d6e6-8a28-11ee-223b-8fe3cb023e45
using DrWatson

# ╔═╡ df5b07d1-5413-4768-8b1b-237b4300fdad
# ╠═╡ show_logs = false
@quickactivate "hd-loss"

# ╔═╡ 5abcd30b-bc3e-4d15-97be-6d53f5cc6481
# ╠═╡ show_logs = false
using CUDA; CUDA.set_runtime_version!(v"11.8")

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

# ╔═╡ 2f295ac4-d211-4969-bc82-1c6aeb8e8fc5
using DistanceTransformsPy: pytransform

# ╔═╡ 85ae8235-f118-4c78-a442-257dfa49e729
using DistanceTransforms: transform, boolean_indicator

# ╔═╡ d7959ac3-4366-41f4-9487-a2b9bc1f65d2
TableOfContents()

# ╔═╡ 4ccc2224-a361-4647-9dd6-bc598cc86bb0
md"""
# Setup
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

# ╔═╡ 26d6e4b4-bc46-4268-93d2-d04bc87223d7
md"""
## Helper functions
"""

# ╔═╡ ed981ddc-8b05-4a29-bf94-120ddb4a62b0
function create_unet_layers(
    kernel_size, de_kernel_size, channel_list;
    downsample = true)

    padding = (kernel_size - 1) ÷ 2

    conv1 = Conv((kernel_size, kernel_size, kernel_size), channel_list[1] => channel_list[2], stride=1, pad=padding)
    conv2 = Conv((kernel_size, kernel_size, kernel_size), channel_list[2] => channel_list[3], stride=1, pad=padding)

    relu1 = relu
    relu2 = relu
    bn1 = BatchNorm(channel_list[2])
    bn2 = BatchNorm(channel_list[3])

    bridge_conv = Conv((kernel_size, kernel_size, kernel_size), channel_list[1] => channel_list[3], stride=1, pad=padding)

    if downsample
        sample = Chain(
            Conv((de_kernel_size, de_kernel_size, de_kernel_size), channel_list[3] => channel_list[3], stride=2, pad=(de_kernel_size - 1) ÷ 2, dilation=1),
            BatchNorm(channel_list[3]),
            relu
        )
    else
        sample = Chain(
            ConvTranspose((de_kernel_size, de_kernel_size, de_kernel_size), channel_list[3] => channel_list[3], stride=2, pad=(de_kernel_size - 1) ÷ 2),
            BatchNorm(channel_list[3]),
            relu
        )
    end

    return (conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample)
end

# ╔═╡ 98910f33-05e4-4ca7-bc33-1b3fda75e7e0
md"""
## Unet module
"""

# ╔═╡ cbf4cfb3-8af6-4fc0-9e7a-eebc73e22a44
begin
    struct UNetModule <: Lux.AbstractExplicitContainerLayer{
        (:conv1, :conv2, :bn1, :bn2, :bridge_conv, :sample)
    }
        conv1::Conv
        conv2::Conv
        relu1::Function
        relu2::Function
        bn1::BatchNorm
        bn2::BatchNorm
        bridge_conv::Conv
        sample::Chain
    end

    function UNetModule(
        kernel_size, de_kernel_size, channel_list;
        downsample = true
    )

        conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample = create_unet_layers(
            kernel_size, de_kernel_size, channel_list;
            downsample = downsample
        )

        UNetModule(conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample)
    end

    function (m::UNetModule)(x, ps, st::NamedTuple)
        res, st_bridge_conv = m.bridge_conv(x, ps.bridge_conv, st.bridge_conv)
        x, st_conv1 = m.conv1(x, ps.conv1, st.conv1)
        x, st_bn1 = m.bn1(x, ps.bn1, st.bn1)
        x = relu(x)

        x, st_conv2 = m.conv2(x, ps.conv2, st.conv2)
        x, st_bn2 = m.bn2(x, ps.bn2, st.bn2)
        x = relu(x)

        x = x .+ res

        next_layer, st_sample = m.sample(x, ps.sample, st.sample)

        st = (conv1=st_conv1, conv2=st_conv2, bn1=st_bn1, bn2=st_bn2, bridge_conv=st_bridge_conv, sample=st_sample)
        return next_layer, x, st
    end
end

# ╔═╡ 40b24e17-5f39-4af8-aac3-77e914712808
md"""
## Deconv module
"""

# ╔═╡ ac8b2060-fe96-4a0f-b00e-5f16fe6f7f8a
begin
    struct DeConvModule <: Lux.AbstractExplicitContainerLayer{
        (:conv1, :conv2, :bn1, :bn2, :bridge_conv, :sample)
    }
        conv1::Conv
        conv2::Conv
        relu1::Function
        relu2::Function
        bn1::BatchNorm
        bn2::BatchNorm
        bridge_conv::Conv
        sample::Chain
    end

    function DeConvModule(
        kernel_size, de_kernel_size, channel_list;
        downsample = false)

        conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample = create_unet_layers(
            kernel_size, de_kernel_size, channel_list;
            downsample = downsample
        )

        DeConvModule(conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample)
    end

    function (m::DeConvModule)(x, ps, st::NamedTuple)
        x, x1 = x[1], x[2]
        x = cat(x, x1; dims=4)

        res, st_bridge_conv = m.bridge_conv(x, ps.bridge_conv, st.bridge_conv)

        x, st_conv1 = m.conv1(x, ps.conv1, st.conv1)
        x, st_bn1 = m.bn1(x, ps.bn1, st.bn1)
        x = relu(x)

        x, st_conv2 = m.conv2(x, ps.conv2, st.conv2)
        x, st_bn2 = m.bn2(x, ps.bn2, st.bn2)
        x = relu(x)

        x = x .+ res

        next_layer, st_sample = m.sample(x, ps.sample, st.sample)

        st = (conv1=st_conv1, conv2=st_conv2, bn1=st_bn1, bn2=st_bn2, bridge_conv=st_bridge_conv, sample=st_sample)
        return next_layer, st
    end
end

# ╔═╡ ed6e1bd5-1fa9-43a8-a28f-085313e36d46
md"""
## Model
"""

# ╔═╡ 6b0c097a-2025-4c72-b051-4fb1053cb02d
begin
    struct FCN <: Lux.AbstractExplicitContainerLayer{
        (:conv1, :conv2, :conv3, :conv4, :conv5, :de_conv1, :de_conv2, :de_conv3, :de_conv4, :last_conv)
    }
        conv1::Chain
        conv2::Chain
        conv3::UNetModule
        conv4::UNetModule
        conv5::UNetModule
        de_conv1::UNetModule
        de_conv2::DeConvModule
        de_conv3::DeConvModule
        de_conv4::DeConvModule
        last_conv::Conv
    end

    function FCN(channel)
        conv1 = Chain(
            Conv((5, 5, 5), 1 => channel, stride=1, pad=2),
            BatchNorm(channel),
            relu
        )
        conv2 = Chain(
            Conv((2, 2, 2), channel => 2 * channel, stride=2, pad=0),
            BatchNorm(2 * channel),
            relu
        )
        conv3 = UNetModule(5, 2, [2 * channel, 2 * channel, 4 * channel])
        conv4 = UNetModule(5, 2, [4 * channel, 4 * channel, 8 * channel])
        conv5 = UNetModule(5, 2, [8 * channel, 8 * channel, 16 * channel])

        de_conv1 = UNetModule(
            5, 2, [16 * channel, 32 * channel, 16 * channel];
            downsample = false
        )
        de_conv2 = DeConvModule(
            5, 2, [32 * channel, 8 * channel, 8 * channel];
            downsample = false
        )
        de_conv3 = DeConvModule(
            5, 2, [16 * channel, 4 * channel, 4 * channel];
            downsample = false
        )
        de_conv4 = DeConvModule(
            5, 2, [8 * channel, 2 * channel, channel];
            downsample = false
        )

        last_conv = Conv((1, 1, 1), 2 * channel => 1, stride=1, pad=0)

        FCN(conv1, conv2, conv3, conv4, conv5, de_conv1, de_conv2, de_conv3, de_conv4, last_conv)
    end

    function (m::FCN)(x, ps, st::NamedTuple)
        # Convolutional layers
        x, st_conv1 = m.conv1(x, ps.conv1, st.conv1)
        x_1 = x  # Store for skip connection
        x, st_conv2 = m.conv2(x, ps.conv2, st.conv2)

        # Downscaling UNet modules
        x, x_2, st_conv3 = m.conv3(x, ps.conv3, st.conv3)
        x, x_3, st_conv4 = m.conv4(x, ps.conv4, st.conv4)
        x, x_4, st_conv5 = m.conv5(x, ps.conv5, st.conv5)

        # Upscaling DeConv modules
        x, _, st_de_conv1 = m.de_conv1(x, ps.de_conv1, st.de_conv1)
        x, st_de_conv2 = m.de_conv2((x, x_4), ps.de_conv2, st.de_conv2)
        x, st_de_conv3 = m.de_conv3((x, x_3), ps.de_conv3, st.de_conv3)
        x, st_de_conv4 = m.de_conv4((x, x_2), ps.de_conv4, st.de_conv4)

        # Concatenate with first skip connection and apply last convolution
        x = cat(x, x_1; dims=4)
        x, st_last_conv = m.last_conv(x, ps.last_conv, st.last_conv)

        # Merge states
        st = (
        conv1=st_conv1, conv2=st_conv2, conv3=st_conv3, conv4=st_conv4, conv5=st_conv5, de_conv1=st_de_conv1, de_conv2=st_de_conv2, de_conv3=st_de_conv3, de_conv4=st_de_conv4, last_conv=st_last_conv
        )

        return x, st
    end
end

# ╔═╡ 09fca34b-44a3-4918-8f6b-5a405b26ec33
model = FCN(4)

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

# ╔═╡ c427229a-3c97-405e-abd9-a1625b8a082b
function compute_loss_hd_scipy(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
	
	y = y |> cpu_device()
	y_pred = y_pred |> cpu_device()
	
	local loss
	for b in axes(y, 5)
		_y_pred = copy(round.(y_pred[:, :, :, 1, b]))
		_y = copy(y[:, :, :, 1, b])

		local _y_dtm, _y_pred_dtm
		ignore_derivatives() do
			_y_dtm = pytransform(_y)
			_y_pred_dtm = pytransform(_y_pred)
		end
		hd = hausdorff_loss(_y_pred, _y, _y_pred_dtm, _y_dtm)
		
		dsc = dice_loss(_y_pred, _y)
		
		loss = hd + dsc
	end
	
	y = y |> gpu_device()
	y_pred = y_pred |> gpu_device()
	
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
# Training Timings
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
	
	ep_scipy = Float64[]
	ep_scipy_std = Float64[]

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
		
		# Python (Scipy)
		ep = time_epoch!(
			model, 
			ps, 
			st, 
			train_loader, 
			compute_loss_hd_scipy
		)
		push!(ep_scipy, BenchmarkTools.minimum(ep).time)
		push!(ep_scipy_std, BenchmarkTools.std(ep).time)

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
	scatterlines!(ep_scipy, label="Hausdorff Loss Scipy")
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
	ep_scipy = ep_scipy,
	ep_scipy_std = ep_scipy_std,
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
# ╠═6972d6e6-8a28-11ee-223b-8fe3cb023e45
# ╠═df5b07d1-5413-4768-8b1b-237b4300fdad
# ╠═0ff50fcc-a71a-403d-9e62-458125ac6dc4
# ╠═5abcd30b-bc3e-4d15-97be-6d53f5cc6481
# ╠═5eaabc58-aa01-419a-bfa1-7cc3eb96d8d1
# ╠═07dc4f9b-7e39-4d76-88e6-3c32068c99b0
# ╠═05d00e65-b22d-4ba9-83e7-3f53647080e7
# ╠═5dc05f88-8672-427b-ab52-d92358f4151c
# ╠═514a75f3-decc-46d0-b9b4-df971631f564
# ╠═2f295ac4-d211-4969-bc82-1c6aeb8e8fc5
# ╠═85ae8235-f118-4c78-a442-257dfa49e729
# ╠═d7959ac3-4366-41f4-9487-a2b9bc1f65d2
# ╟─4ccc2224-a361-4647-9dd6-bc598cc86bb0
# ╟─1b758e7b-df5c-4fc0-b534-ef3f76b5a59a
# ╠═c6dec210-71d4-4f07-b5d1-93af2dc2d6e3
# ╟─99abc39a-6f4d-4f99-8380-682a98d56b8e
# ╠═26d6e4b4-bc46-4268-93d2-d04bc87223d7
# ╠═ed981ddc-8b05-4a29-bf94-120ddb4a62b0
# ╟─98910f33-05e4-4ca7-bc33-1b3fda75e7e0
# ╠═cbf4cfb3-8af6-4fc0-9e7a-eebc73e22a44
# ╟─40b24e17-5f39-4af8-aac3-77e914712808
# ╠═ac8b2060-fe96-4a0f-b00e-5f16fe6f7f8a
# ╟─ed6e1bd5-1fa9-43a8-a28f-085313e36d46
# ╠═6b0c097a-2025-4c72-b051-4fb1053cb02d
# ╠═09fca34b-44a3-4918-8f6b-5a405b26ec33
# ╠═fa520196-ea57-4c25-9445-bdabbeab9a82
# ╟─4c0d4d36-dccb-420f-a383-c39a64e1b42d
# ╠═43748430-84c2-4915-a7af-3df50b1cc48a
# ╟─f71d9fac-345e-4031-8443-5c4cf54bcaf1
# ╠═8bf8ed10-3170-4404-8916-ba129efba390
# ╠═5cb8b16e-9d0d-46a7-813c-344a29a8c4b6
# ╠═c427229a-3c97-405e-abd9-a1625b8a082b
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
