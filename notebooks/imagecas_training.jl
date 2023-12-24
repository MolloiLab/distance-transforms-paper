### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ c3a61cfa-7bed-43de-aec3-d9c2f78fbd2a
# ╠═╡ show_logs = false
using Pkg; Pkg.instantiate()

# ╔═╡ 8fa34e64-9ba2-11ee-2b38-8fbf1139c9c9
using DrWatson

# ╔═╡ 3dad7c76-b87d-4a98-8d5d-e79807dc56ce
# ╠═╡ show_logs = false
@quickactivate "hd-loss"

# ╔═╡ 532d695d-5cd5-4636-b1f3-2ed0d85dc4aa
# ╠═╡ show_logs = false
using CSV: read

# ╔═╡ 3ea09ae8-11b9-4834-ac51-aba7165e7cce
using Statistics: mean, std

# ╔═╡ 76724b75-7b20-4e06-9f7f-67f3b741d440
using DataFrames: DataFrame

# ╔═╡ 6032d697-594b-4e64-a80e-1ae34ae22c9b
using ChainRulesCore: ignore_derivatives

# ╔═╡ 3796367c-41a7-4024-bfdb-afc5efbbb0cf
using DistanceTransforms: transform, boolean_indicator

# ╔═╡ 99ce5436-50ac-4237-b2a8-bd6c575cba0d
using Losers: dice_loss, hausdorff_loss

# ╔═╡ 247ef623-1521-474d-9943-4fdf2587f4d9
using NIfTI: niread

# ╔═╡ f6af8e21-e6f5-4670-b771-206ade2a7e47
using Glob: glob

# ╔═╡ 2f14c39b-355f-49a2-9d84-05e71fcae24a
# ╠═╡ show_logs = false
using PlutoUI, CairoMakie

# ╔═╡ 743ee7cb-4c25-49f0-aa15-d2354075e80a
using Lux, LuxCUDA, Random, Optimisers, Zygote

# ╔═╡ b8429eef-bd66-4934-af2a-42527e5ffbc1
# ╠═╡ show_logs = false
using CUDA; CUDA.set_runtime_version!(v"11.8")

# ╔═╡ 6164d96e-7db7-43a5-8e02-7997850537dc
using MLUtils

# ╔═╡ 076c9334-d6db-43e0-967c-3a118d3f3c45
using MLDatasets

# ╔═╡ 653c9fa1-ff86-4129-b432-2f9863c6f480
using FileIO

# ╔═╡ 77147872-8931-45d3-b819-c788ae0afc5f
using MLUtils: DataLoader

# ╔═╡ fe96874c-5e26-4f84-9c61-b6f1914e4ab3
md"""
# Setup
"""

# ╔═╡ 8da9f6ae-8df5-418b-b77a-dd7746d0d236
md"""
## Imports
"""

# ╔═╡ 21d86c36-78ab-4735-b5aa-05fb44fb3df4
TableOfContents()

# ╔═╡ 93599014-9428-407e-89a1-611390c6f4d2
md"""
# Data Preprocessing
"""

# ╔═╡ b751c68f-af71-45f4-98a5-aae84b8ff84c
begin
	function load_image(path)
	    img = niread(path).raw
	    return convert(Array{Float32}, img) / maximum(img)
	end
	
	function load_label(path)
	    label = niread(path).raw
	    return convert(Array{UInt8}, label) .+ 1
	end
	
	struct ImageLabelDataset
	    folders::Vector{String}
	end
	
	# Define the length method for your struct
	Base.length(data::ImageLabelDataset) = length(data.folders)
	
	# Define the getobs method for your struct
	function getobs(data::ImageLabelDataset, idx)
	    folder_path = data.folders[idx]
	    image_file = glob("img.nii.gz", folder_path)[1]
	    label_file = glob("label.nii.gz", folder_path)[1]
	    image = load_image(image_file)
	    label = load_label(label_file)
	    return (image, label)
	end
end

# ╔═╡ a2b2bb11-05a2-4b79-8529-9c1d89884381
begin
	# Directory containing patient folders
	data_dir = "/dfs7/symolloi-lab/imageCAS"
	patient_folders = [joinpath(data_dir, folder) for folder in readdir(data_dir)]
	
	dataset = ImageLabelDataset(patient_folders)
	dataloader = DataLoader(dataset, batchsize=1)

	count = 0
	for (image, label) in dataloader
		count += 1
		if count > 2
			break
		end
	end
end

# ╔═╡ a9caef6e-45aa-4d02-8782-d648869bdb58
# begin
# 	data_dir = "/dfs7/symolloi-lab/imageCAS"
# 	image_paths = [joinpath(data_dir, folder, "img.nii.gz") for folder in readdir(data_dir)]
# 	label_paths = [joinpath(data_dir, folder, "label.nii.gz") for folder in readdir(data_dir)]
	
# 	# Create FileDataset instances
# 	image_dataset = FileDataset(load_image, image_paths)
# 	label_dataset = FileDataset(load_label, label_paths)
# end

# ╔═╡ de160e29-903d-4388-9a2e-d1f62a3bc61c
# begin
# 	bs = 2
# 	dataloader = DataLoader((image_dataset, label_dataset), batchsize=bs)
	
# 	# for (images, labels) in dataloader
# 	#     @info size(images)
# 	# end
# end

# ╔═╡ 284e824c-7013-4f47-9a73-857c5f136520


# ╔═╡ 32007e30-2cae-425f-a9c3-f5234a77a5ef


# ╔═╡ 4e582530-3d16-4f68-9bfa-a412a6285b38


# ╔═╡ 87a5da94-2ec4-48c6-886d-2e3ddeb7cb93
md"""
## Randomness
"""

# ╔═╡ 1aa1582f-66a2-4d1f-ae0d-04189c6282ba
begin
	rng = MersenneTwister()
	Random.seed!(rng, 12345)
end;

# ╔═╡ 7251ba20-17ba-4518-a601-27b11a2860d1
md"""
# Model (FCN)
"""

# ╔═╡ 439939a1-48c6-4d53-8d9b-61bc6233f7eb
md"""
## Helper functions
"""

# ╔═╡ b45964ca-5f30-4731-8f05-3efd523a2b66
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

# ╔═╡ 61e86c10-6bc4-45b1-af0d-db23cb544f0b
md"""
## Unet module
"""

# ╔═╡ c0f42d65-7de1-4cbe-b1a2-729169454f65
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

# ╔═╡ 7f5d2c84-58df-4ba5-9f09-a08d637356c4
md"""
## Deconv module
"""

# ╔═╡ 7801a31f-8c70-4289-8bf5-c58d8f877032
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

# ╔═╡ d3f35a6c-eb52-47c3-9638-4ef7c1914211
md"""
## FCN
"""

# ╔═╡ 7d8bc6d7-38a1-4e35-9b59-b54cc609944d
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

# ╔═╡ c940f458-4b49-48e5-83a2-dd32a356fe42
model = FCN(4);

# ╔═╡ d86b5bdb-0d85-4f4b-80a1-2f14b45bfdeb
ps, st = Lux.setup(rng, model)

# ╔═╡ 6cc22ec9-932e-4428-8a16-32d0fd909f05
md"""
# Basics
"""

# ╔═╡ 8f6e03ea-60c4-47ea-8955-d871dd097566
md"""
## Optimizer
"""

# ╔═╡ 387be07e-0540-423c-98ca-95d8b554d31d
function create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
    return Optimisers.setup(opt, ps)
end

# ╔═╡ 345f92b7-5d03-4ac3-a958-5bc2217e1397
md"""
## Loss function(s)
"""

# ╔═╡ 209403e7-46e2-417d-bd72-154afc3efb29
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

# ╔═╡ 5fdbbfd3-25ec-4eef-8629-72cb0d1c0889
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

# ╔═╡ 510b4f66-f6cd-4e39-b787-276fa94d3ce4
md"""
# Training
"""

# ╔═╡ 8fa58c10-82e1-4c62-81a1-1ba2615051ad
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

# ╔═╡ ff74b51d-f747-4ccc-a369-5bcd17e4e8c1
function train!(model, ps, st, train_loader, compute_loss)
    dev = cpu_device()
    ps = ps |> dev
    st = st |> dev

    # Create the optimiser
    opt_state = create_optimiser(ps)
	run_epoch!(train_loader, dev, compute_loss, model, ps, st, opt_state)
end

# ╔═╡ 926479fe-6fc6-4b47-91c5-8cb28c14b3f9
img_size = (512, 512, 256)

# ╔═╡ b84872fe-843e-4e97-b272-3c45349458dd
channels = 1

# ╔═╡ 835781d1-033b-4c30-97da-ea855151cdb5
num_imgs = 2

# ╔═╡ 67e1a847-98bb-4089-a39c-6637268acaf4
batch_size = 1

# ╔═╡ 0101059c-f92a-4c51-bb89-52f308b05dd7
# begin
# 	img_data = rand(Float32, img_size..., channels, num_imgs)
# 	lbl_data = rand([0f0, 1f0], img_size..., channels, num_imgs)
# 	(img_train, lbl_train), (img_val, lbl_val) = splitobs((img_data, lbl_data); at=0.8, shuffle=true)
# 	train_loader = DataLoader(collect.((img_train, lbl_train)); batchsize=batch_size, shuffle=true)
# end

# ╔═╡ 1de3ae29-6ef4-4d64-bb8f-0b3e3c608ed6
# ╠═╡ skip_as_script = true
#=╠═╡
# train!(model, ps, st, train_loader, compute_loss_dice)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─fe96874c-5e26-4f84-9c61-b6f1914e4ab3
# ╟─8da9f6ae-8df5-418b-b77a-dd7746d0d236
# ╠═8fa34e64-9ba2-11ee-2b38-8fbf1139c9c9
# ╠═3dad7c76-b87d-4a98-8d5d-e79807dc56ce
# ╠═c3a61cfa-7bed-43de-aec3-d9c2f78fbd2a
# ╠═b8429eef-bd66-4934-af2a-42527e5ffbc1
# ╠═532d695d-5cd5-4636-b1f3-2ed0d85dc4aa
# ╠═3ea09ae8-11b9-4834-ac51-aba7165e7cce
# ╠═76724b75-7b20-4e06-9f7f-67f3b741d440
# ╠═6032d697-594b-4e64-a80e-1ae34ae22c9b
# ╠═3796367c-41a7-4024-bfdb-afc5efbbb0cf
# ╠═99ce5436-50ac-4237-b2a8-bd6c575cba0d
# ╠═247ef623-1521-474d-9943-4fdf2587f4d9
# ╠═f6af8e21-e6f5-4670-b771-206ade2a7e47
# ╠═2f14c39b-355f-49a2-9d84-05e71fcae24a
# ╠═743ee7cb-4c25-49f0-aa15-d2354075e80a
# ╠═6164d96e-7db7-43a5-8e02-7997850537dc
# ╠═076c9334-d6db-43e0-967c-3a118d3f3c45
# ╠═653c9fa1-ff86-4129-b432-2f9863c6f480
# ╠═21d86c36-78ab-4735-b5aa-05fb44fb3df4
# ╟─93599014-9428-407e-89a1-611390c6f4d2
# ╠═77147872-8931-45d3-b819-c788ae0afc5f
# ╠═b751c68f-af71-45f4-98a5-aae84b8ff84c
# ╠═a2b2bb11-05a2-4b79-8529-9c1d89884381
# ╠═a9caef6e-45aa-4d02-8782-d648869bdb58
# ╠═de160e29-903d-4388-9a2e-d1f62a3bc61c
# ╠═284e824c-7013-4f47-9a73-857c5f136520
# ╠═32007e30-2cae-425f-a9c3-f5234a77a5ef
# ╠═4e582530-3d16-4f68-9bfa-a412a6285b38
# ╟─87a5da94-2ec4-48c6-886d-2e3ddeb7cb93
# ╠═1aa1582f-66a2-4d1f-ae0d-04189c6282ba
# ╟─7251ba20-17ba-4518-a601-27b11a2860d1
# ╟─439939a1-48c6-4d53-8d9b-61bc6233f7eb
# ╠═b45964ca-5f30-4731-8f05-3efd523a2b66
# ╟─61e86c10-6bc4-45b1-af0d-db23cb544f0b
# ╠═c0f42d65-7de1-4cbe-b1a2-729169454f65
# ╟─7f5d2c84-58df-4ba5-9f09-a08d637356c4
# ╠═7801a31f-8c70-4289-8bf5-c58d8f877032
# ╟─d3f35a6c-eb52-47c3-9638-4ef7c1914211
# ╠═7d8bc6d7-38a1-4e35-9b59-b54cc609944d
# ╠═c940f458-4b49-48e5-83a2-dd32a356fe42
# ╠═d86b5bdb-0d85-4f4b-80a1-2f14b45bfdeb
# ╟─6cc22ec9-932e-4428-8a16-32d0fd909f05
# ╟─8f6e03ea-60c4-47ea-8955-d871dd097566
# ╠═387be07e-0540-423c-98ca-95d8b554d31d
# ╟─345f92b7-5d03-4ac3-a958-5bc2217e1397
# ╠═209403e7-46e2-417d-bd72-154afc3efb29
# ╠═5fdbbfd3-25ec-4eef-8629-72cb0d1c0889
# ╟─510b4f66-f6cd-4e39-b787-276fa94d3ce4
# ╠═8fa58c10-82e1-4c62-81a1-1ba2615051ad
# ╠═ff74b51d-f747-4ccc-a369-5bcd17e4e8c1
# ╠═926479fe-6fc6-4b47-91c5-8cb28c14b3f9
# ╠═b84872fe-843e-4e97-b272-3c45349458dd
# ╠═835781d1-033b-4c30-97da-ea855151cdb5
# ╠═67e1a847-98bb-4089-a39c-6637268acaf4
# ╠═0101059c-f92a-4c51-bb89-52f308b05dd7
# ╠═1de3ae29-6ef4-4d64-bb8f-0b3e3c608ed6
