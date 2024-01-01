### A Pluto.jl notebook ###
# v0.19.32

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

# ╔═╡ c3a61cfa-7bed-43de-aec3-d9c2f78fbd2a
# ╠═╡ show_logs = false
using Pkg; Pkg.instantiate()

# ╔═╡ 8fa34e64-9ba2-11ee-2b38-8fbf1139c9c9
using DrWatson

# ╔═╡ 3dad7c76-b87d-4a98-8d5d-e79807dc56ce
# ╠═╡ show_logs = false
@quickactivate "hd-loss"

# ╔═╡ 2f14c39b-355f-49a2-9d84-05e71fcae24a
# ╠═╡ show_logs = false
using PlutoUI: TableOfContents, bind, Slider

# ╔═╡ e4e6c5a0-c3df-4124-9e60-2af55719482e
using Random: MersenneTwister, seed!

# ╔═╡ 247ef623-1521-474d-9943-4fdf2587f4d9
using NIfTI: niread

# ╔═╡ 77147872-8931-45d3-b819-c788ae0afc5f
# ╠═╡ show_logs = false
using MLUtils: DataLoader, mapobs, splitobs, getobs

# ╔═╡ a4bef095-718e-4b5c-a68b-296454f7914f
using ImageTransformations: imresize

# ╔═╡ ed3122e6-2d3d-4582-9bd4-e4ed016f2686
# ╠═╡ show_logs = false
using Lux

# ╔═╡ e9885bd2-444a-42ff-8d36-0c7b5c5fbbfb
using LuxCUDA

# ╔═╡ b8429eef-bd66-4934-af2a-42527e5ffbc1
# ╠═╡ show_logs = false
using CUDA; CUDA.set_runtime_version!(v"11.8")

# ╔═╡ c53f5aba-5b87-4528-873f-7a36a673222f
using CairoMakie: Figure, Axis, heatmap!

# ╔═╡ 6032d697-594b-4e64-a80e-1ae34ae22c9b
using ChainRulesCore: ignore_derivatives

# ╔═╡ 3ea09ae8-11b9-4834-ac51-aba7165e7cce
using Statistics: mean, std

# ╔═╡ 21d86c36-78ab-4735-b5aa-05fb44fb3df4
TableOfContents()

# ╔═╡ 1aa1582f-66a2-4d1f-ae0d-04189c6282ba
begin
	rng = MersenneTwister()
	seed!(rng, 12345)
end;

# ╔═╡ a66c190a-df54-4c84-a5d8-d380dbbb83f7
begin
	struct ImageCASDataset
		image_paths::Vector{String}
		label_paths::Vector{String}
	end
	
	function ImageCASDataset(data_dir::String)
	    patient_folders = [joinpath(data_dir, folder) for folder in readdir(data_dir)]
	    image_paths = String[]
	    label_paths = String[]
	
	    for folder in patient_folders
	        push!(image_paths, joinpath(folder, "img.nii.gz"))
	        push!(label_paths, joinpath(folder, "label.nii.gz"))
	    end
	
	    return ImageCASDataset(image_paths, label_paths)
	end
	
	Base.length(d::ImageCASDataset) = length(d.image_paths)
	
	function Base.getindex(d::ImageCASDataset, i::Int)
	    image = niread(d.image_paths[i]).raw
	    label = niread(d.label_paths[i]).raw
	    return (image, label)
	end
	
	function Base.getindex(d::ImageCASDataset, idxs::AbstractVector{Int})
	    images = Vector{Array{Float32, 3}}(undef, length(idxs))
	    labels = Vector{Array{UInt8, 3}}(undef, length(idxs))
	    for (index, i) in enumerate(idxs)
	        images[index] = niread(d.image_paths[i]).raw
	        labels[index]  = niread(d.label_paths[i]).raw
	    end
	    return (images, labels)
	end
end

# ╔═╡ 363a2415-9c74-4f86-9eb1-eabaaa7f8de7
md"""
# Setup
"""

# ╔═╡ d16c87b3-4fea-4084-8d89-f03e84fff49e
md"""
## Environment
"""

# ╔═╡ 87a5da94-2ec4-48c6-886d-2e3ddeb7cb93
md"""
## Randomness
"""

# ╔═╡ a08c5e89-b1d2-43f1-ac11-5758235298fb
md"""
# Data Preparation
"""

# ╔═╡ e81b97f5-9a6b-4866-b0a8-6f7b69b7e045
md"""
## Dataset
"""

# ╔═╡ 187a9e23-987d-4a58-9862-4d7262f181fd
data_dir = "/dfs7/symolloi-lab/imageCAS"

# ╔═╡ 61ceda68-bd27-465b-8a9d-ded5b9c6191a
data = ImageCASDataset(data_dir)

# ╔═╡ e42fdf29-ac12-43ba-b7ed-55f63acd1794
md"""
## Preprocessing
"""

# ╔═╡ f1b57532-c87a-4e99-ab7a-e65891d25e6f
function adjust_image_size(
    volume::Array{T, 3}, 
    target_size::Tuple{Int, Int, Int};
    max_crop_percentage::Float64 = 0.15
) where T
    current_size = size(volume)
    
    # Check if the image is already the target size
    if isequal(current_size, target_size)
        return volume
    end

    # Calculate the maximum allowable crop size
    max_crop_size = round.(Int, current_size .* (1 - max_crop_percentage))

    # Resize if the image is smaller than the target size
    if all(x -> x[1] < x[2], zip(current_size, target_size))
        return imresize(volume, target_size)
    end

    # Adjust if the image is larger than the target size
    if all(x -> x[1] > x[2], zip(current_size, target_size))
        # Determine crop size, limited by the max crop size
        crop_size = max.(max_crop_size, target_size)

        # Center crop
        center = div.(current_size, 2)
        start_idx = max.(1, center .- div.(crop_size, 2))
        end_idx = start_idx .+ crop_size .- 1
		cropped_volume = volume[start_idx[1]:end_idx[1], start_idx[2]:end_idx[2], start_idx[3]:end_idx[3]]

        # Resize if cropped size is not yet the target size
        if any(x -> x[1] != x[2], zip(size(cropped_volume), target_size))
            return imresize(cropped_volume, target_size)
        else
            return cropped_volume
        end
    end

    # Return the original volume if none of the above conditions are met
    return volume
end

# ╔═╡ d9366696-2f06-4ac0-abea-e80d0262ef4b
function one_hot_encode(label::Array{T, 3}, num_classes::Int) where {T}
	one_hot = zeros(T, size(label)..., num_classes)
	
    for k in 1:num_classes
        one_hot[:, :, :, k] = label .== k-1
    end
	
    return one_hot
end

# ╔═╡ 4605121b-55a7-44b4-b1f2-3947dc3d4807
function preprocess_image_label_pair(pair, target_size)
    # Check if pair[1] and pair[2] are individual arrays or collections of arrays
    is_individual = ndims(pair[1]) == 3 && ndims(pair[2]) == 3

    if is_individual
        # Handle a single pair
        cropped_image = adjust_image_size(pair[1], target_size)
        cropped_label = one_hot_encode(adjust_image_size(pair[2], target_size), 2)
        processed_image = reshape(cropped_image, size(cropped_image)..., 1)
        return (Float32.(processed_image), Float32.(cropped_label))
    else
        # Handle a batch of pairs
		@info pair[1]
		cropped_images = [adjust_image_size(img, target_size) for img in pair[1]]
		cropped_labels = [one_hot_encode(adjust_image_size(lbl, target_size), 2) for lbl in pair[2]]
		processed_images = [reshape(img, size(img)..., 1) for img in cropped_images]
        return (Float32.(processed_images), Float32.(cropped_labels))
    end
end

# ╔═╡ ed3e593d-8580-4e70-bb21-bea904b594dd
if LuxCUDA.functional()
	target_size = (512, 512, 256)
else
	target_size = (64, 64, 32)
end

# ╔═╡ cc2ec55c-75c4-4033-8b0d-1fc15f543143
transformed_data = mapobs(
	x -> preprocess_image_label_pair(x, target_size),
	data
)

# ╔═╡ df15933f-ebd6-403b-b0d6-2abf048abbfa
md"""
## Dataloaders
"""

# ╔═╡ aa2b57d7-d6c2-431a-a4c0-809d2146809b
train_data, val_data = splitobs(transformed_data; at = 0.75)

# ╔═╡ 492d4b52-a19c-4277-a123-da856e7ff441
bs = 1

# ╔═╡ a3922176-be46-4c39-8e19-7ae1659a318c
begin
	train_loader = DataLoader(train_data; batchsize = bs, collate = true)
	val_loader = DataLoader(val_data; batchsize = bs, collate = true)
end

# ╔═╡ 2d2f7e84-fd88-4f05-90d1-b17b305f7bee
md"""
# Data Visualization
"""

# ╔═╡ 80b023f8-55a2-4312-b4ae-381f72885e45
md"""
## Original Data
"""

# ╔═╡ 4384cd1a-179b-487d-97da-298172a50de1
image_raw, label_raw = getobs(data, 1);

# ╔═╡ f4fd62b4-305a-46d3-9c95-9d6617b254b7
@bind z1 Slider(axes(image_raw, 3), show_value = true, default = div(size(image_raw, 3), 2))

# ╔═╡ c0119a87-10e1-4a12-b908-c1c09d862357
let
	f = Figure(size = (700, 500))
	ax = Axis(
		f[1, 1],
		title = "Original Image"
	)
	heatmap!(image_raw[:, :, z1]; colormap = :grays)

	ax = Axis(
		f[1, 2],
		title = "Original Label (Overlayed)"
	)
	heatmap!(image_raw[:, :, z1]; colormap = :grays)
	heatmap!(label_raw[:, :, z1]; colormap = (:jet, 0.4))
	f
end

# ╔═╡ 4c1da79b-8389-4248-ba50-cfd7577aa6df
md"""
## Transformed Data
"""

# ╔═╡ ceb2145b-1d2e-4615-9494-076f525f842a
image_tfm, label_tfm = getobs(transformed_data, 1);

# ╔═╡ d617ca19-d915-48a5-b282-5af0ec09b362
unique(label_tfm)

# ╔═╡ 84e9b251-c83e-4bb7-a065-6ca1a9cffb49
@bind z2 Slider(1:target_size[3], show_value = true, default = div(target_size[3], 2))

# ╔═╡ 337e609b-15d9-4ce3-82cd-5bceb0e479af
let
	f = Figure(size = (700, 500))
	ax = Axis(
		f[1, 1],
		title = "Transformed Image"
	)
	heatmap!(image_tfm[:, :, z2, 1]; colormap = :grays)

	ax = Axis(
		f[1, 2],
		title = "Transformed Label (Overlayed)"
	)
	heatmap!(image_tfm[:, :, z2, 1]; colormap = :grays)
	heatmap!(label_tfm[:, :, z2, 2]; colormap = (:jet, 0.4))
	f
end

# ╔═╡ e501bc02-a1f0-4a3c-9587-e81237a10bc8
md"""
# Model
"""

# ╔═╡ 50983336-c945-4835-a9ee-b56a6476ad74
md"""
## Helper functions
"""

# ╔═╡ 0b10cb62-6a4d-4acf-aef1-f471e21ef5e8
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

# ╔═╡ 0cd872e4-357b-49f6-9212-3b02e09aa020
md"""
## Contracting Block
"""

# ╔═╡ fb697e38-a345-4297-90b8-0d686f4ecec3
begin
    struct ContractBlock <: Lux.AbstractExplicitContainerLayer{
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

    function ContractBlock(
        kernel_size, de_kernel_size, channel_list;
        downsample = true
    )


		conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample = create_unet_layers(
            kernel_size, de_kernel_size, channel_list;
            downsample = downsample
        )

        ContractBlock(conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample)
    end

    function (m::ContractBlock)(x, ps, st::NamedTuple)
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

# ╔═╡ 9228b077-9b18-4485-93a0-07c84edd7c48
md"""
## Expanding Block
"""

# ╔═╡ 45ded54e-8ee3-4aa5-a1a5-79224a416b47
begin
    struct ExpandBlock <: Lux.AbstractExplicitContainerLayer{
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

    function ExpandBlock(
        kernel_size, de_kernel_size, channel_list;
        downsample = false)

		conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample = create_unet_layers(
            kernel_size, de_kernel_size, channel_list;
            downsample = downsample
        )

        ExpandBlock(conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample)
    end

    function (m::ExpandBlock)(x, ps, st::NamedTuple)
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

# ╔═╡ 8c0ea123-42db-4b95-bde8-46f2bafff2c9
md"""
## U-Net
"""

# ╔═╡ d762f8e2-853a-455c-9fc7-691e54a04593
begin
    struct UNet <: Lux.AbstractExplicitContainerLayer{
        (:conv1, :conv2, :conv3, :conv4, :conv5, :de_conv1, :de_conv2, :de_conv3, :de_conv4, :last_conv)
    }
        conv1::Chain
        conv2::Chain
        conv3::ContractBlock
        conv4::ContractBlock
        conv5::ContractBlock
        de_conv1::ContractBlock
        de_conv2::ExpandBlock
        de_conv3::ExpandBlock
        de_conv4::ExpandBlock
        last_conv::Conv
    end

    function UNet(channel)
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
        conv3 = ContractBlock(5, 2, [2 * channel, 2 * channel, 4 * channel])
        conv4 = ContractBlock(5, 2, [4 * channel, 4 * channel, 8 * channel])
        conv5 = ContractBlock(5, 2, [8 * channel, 8 * channel, 16 * channel])

        de_conv1 = ContractBlock(
            5, 2, [16 * channel, 32 * channel, 16 * channel];
            downsample = false
        )
        de_conv2 = ExpandBlock(
            5, 2, [32 * channel, 8 * channel, 8 * channel];
            downsample = false
        )
        de_conv3 = ExpandBlock(
            5, 2, [16 * channel, 4 * channel, 4 * channel];
            downsample = false
        )
        de_conv4 = ExpandBlock(
            5, 2, [8 * channel, 2 * channel, channel];
            downsample = false
        )

        last_conv = Conv((1, 1, 1), 2 * channel => 2, stride=1, pad=0)

		UNet(conv1, conv2, conv3, conv4, conv5, de_conv1, de_conv2, de_conv3, de_conv4, last_conv)
    end

    function (m::UNet)(x, ps, st::NamedTuple)
        # Convolutional layers
        x, st_conv1 = m.conv1(x, ps.conv1, st.conv1)
        x_1 = x  # Store for skip connection
        x, st_conv2 = m.conv2(x, ps.conv2, st.conv2)

        # Downscaling Blocks
        x, x_2, st_conv3 = m.conv3(x, ps.conv3, st.conv3)
        x, x_3, st_conv4 = m.conv4(x, ps.conv4, st.conv4)
        x, x_4, st_conv5 = m.conv5(x, ps.conv5, st.conv5)

        # Upscaling Blocks
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

# ╔═╡ b444cb9d-987c-482f-a0c8-634538a8829f
md"""
# Training Set Up
"""

# ╔═╡ 3796367c-41a7-4024-bfdb-afc5efbbb0cf
# using DistanceTransforms: transform, boolean_indicator

# ╔═╡ 99ce5436-50ac-4237-b2a8-bd6c575cba0d
# using Losers: dice_loss, hausdorff_loss

# ╔═╡ d9eb6d05-78fc-4c7b-a390-1ba4c1db1d47
import Zygote

# ╔═╡ 43fe8326-fd33-4869-9d0c-0ccfe4ec01cf
import Optimisers

# ╔═╡ 14e78c3b-1e86-4d64-b280-938e3e181f36
md"""
## Optimiser
"""

# ╔═╡ c7dbbbcd-65c5-4a0b-ad71-79f1b1ce1767
function create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
    return Optimisers.setup(opt, ps)
end

# ╔═╡ 0d10693a-a77a-4550-a1c1-e198ec11c985
md"""
## Loss function
"""

# ╔═╡ 3d9ffa18-cb55-4bf1-b874-d7e1aa04355b
# function compute_loss(x, y, model, ps, st, epoch)
#     alpha = max(1.0 - 0.01 * epoch, 0.01)
#     beta = 1.0 - alpha

#     y_pred, st = model(x, ps, st)

#     y_pred_softmax = softmax(y_pred, dims=4)
#     y_pred_binary = round.(y_pred_softmax[:, :, :, 2, :])
#     y_binary = y[:, :, :, 2, :]

#     # Compute loss
#     loss = 0.0
#     for b in axes(y, 5)
#         _y_pred = y_pred_binary[:, :, :, b]
#         _y = y_binary[:, :, :, b]

# 		local _y_dtm, _y_pred_dtm
# 		ignore_derivatives() do
# 			_y_dtm = transform(boolean_indicator(_y))
# 			_y_pred_dtm = transform(boolean_indicator(_y_pred))
# 		end
		
# 		hd = hausdorff_loss(_y_pred, _y, _y_pred_dtm, _y_dtm)
# 		dsc = dice_loss(_y_pred, _y)
# 		loss += alpha * dsc + beta * hd
#     end
	
#     return loss / size(y, 5), y_pred_binary, st
# end

# ╔═╡ 1066f4c7-d463-4119-821c-e7c50e84a909
function dice_loss(ŷ, y, ϵ=1e-5)
    return loss = 1 - ((2 * sum(ŷ .* y) + ϵ) / (sum(ŷ .* ŷ) + sum(y .* y) + ϵ))
end

# ╔═╡ 6990e258-7dbe-47f2-b225-fe9b1ff6f607
function compute_loss(x, y, model, ps, st)

    y_pred, st = model(x, ps, st)

    y_pred_softmax = softmax(y_pred, dims=4)
    y_pred_binary = round.(y_pred_softmax[:, :, :, 2, :])
    y_binary = y[:, :, :, 2, :]

    # Compute loss
    loss = 0.0
    for b in axes(y, 5)
        _y_pred = y_pred_binary[:, :, :, b]
        _y = y_binary[:, :, :, b]
		
		dsc = dice_loss(_y_pred, _y)
		loss += dsc
    end
	
    return loss / size(y, 5), y_pred_binary, st
end

# ╔═╡ 075c60a6-d1d3-4f64-9186-1fbf1a8abc08
md"""
# Train
"""

# ╔═╡ df358406-3547-472c-8d12-02dea6719145
dev = gpu_device()

# ╔═╡ 5ff89b95-29c0-4b62-b55e-0d23afba43f5
model = UNet(4)

# ╔═╡ 72efcca2-5300-4524-b2b5-09545bdacb92
begin
	ps, st = Lux.setup(rng, model)
	ps, st = ps |> dev, st |> dev
end

# ╔═╡ 6c4dcde0-c72b-4562-b4d9-ea4e15eba84c
function train_model(model, ps, st, train_loader, num_epochs, dev)
    opt_state = create_optimiser(ps)

    for epoch in 1:num_epochs
		@info "Epoch: $epoch"

		# Training Phase
        for (x, y) in train_loader
			x = x |> dev
			y = y |> dev
			
            # Forward pass
            y_pred, st = Lux.apply(model, x, ps, st)
            loss, y_pred, st = compute_loss(x, y, model, ps, st)
			# @info "Training Loss: $loss"

            # Backward pass
			(loss_grad, st_), back = Zygote.pullback(p -> Lux.apply(model, x, p, st), ps)
            gs = back((one.(loss_grad), nothing))[1]

            # Update parameters
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end

		# Validation Phase
		total_loss = 0.0
		num_batches = 0
	    for (x, y) in val_loader
			x, y = x |> dev, y |> dev
			
	        # Forward Pass
	        y_pred, st = Lux.apply(model, x, ps, st)
	        loss, _, _ = compute_loss(x, y, model, ps, st)
	
	        total_loss += loss
	        num_batches += 1
	    end
		avg_loss = total_loss / num_batches
		@info "Validation Loss: $avg_loss"
    end

    return ps, st
end

# ╔═╡ 2259e733-0210-4e4d-839d-8471c462c694
if LuxCUDA.functional()
	num_epochs = 20
else
	num_epochs = 2
end

# ╔═╡ 62a09464-b6d1-4131-8565-a1754ba634d5
# ╠═╡ disabled = true
#=╠═╡
train_model(model, ps, st, train_loader, num_epochs, dev)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─363a2415-9c74-4f86-9eb1-eabaaa7f8de7
# ╟─d16c87b3-4fea-4084-8d89-f03e84fff49e
# ╠═8fa34e64-9ba2-11ee-2b38-8fbf1139c9c9
# ╠═3dad7c76-b87d-4a98-8d5d-e79807dc56ce
# ╠═c3a61cfa-7bed-43de-aec3-d9c2f78fbd2a
# ╠═b8429eef-bd66-4934-af2a-42527e5ffbc1
# ╠═2f14c39b-355f-49a2-9d84-05e71fcae24a
# ╠═21d86c36-78ab-4735-b5aa-05fb44fb3df4
# ╟─87a5da94-2ec4-48c6-886d-2e3ddeb7cb93
# ╠═e4e6c5a0-c3df-4124-9e60-2af55719482e
# ╠═1aa1582f-66a2-4d1f-ae0d-04189c6282ba
# ╟─a08c5e89-b1d2-43f1-ac11-5758235298fb
# ╠═247ef623-1521-474d-9943-4fdf2587f4d9
# ╠═77147872-8931-45d3-b819-c788ae0afc5f
# ╠═a4bef095-718e-4b5c-a68b-296454f7914f
# ╠═ed3122e6-2d3d-4582-9bd4-e4ed016f2686
# ╠═e9885bd2-444a-42ff-8d36-0c7b5c5fbbfb
# ╟─e81b97f5-9a6b-4866-b0a8-6f7b69b7e045
# ╠═a66c190a-df54-4c84-a5d8-d380dbbb83f7
# ╠═187a9e23-987d-4a58-9862-4d7262f181fd
# ╠═61ceda68-bd27-465b-8a9d-ded5b9c6191a
# ╟─e42fdf29-ac12-43ba-b7ed-55f63acd1794
# ╠═f1b57532-c87a-4e99-ab7a-e65891d25e6f
# ╠═d9366696-2f06-4ac0-abea-e80d0262ef4b
# ╠═4605121b-55a7-44b4-b1f2-3947dc3d4807
# ╠═ed3e593d-8580-4e70-bb21-bea904b594dd
# ╠═cc2ec55c-75c4-4033-8b0d-1fc15f543143
# ╟─df15933f-ebd6-403b-b0d6-2abf048abbfa
# ╠═aa2b57d7-d6c2-431a-a4c0-809d2146809b
# ╠═492d4b52-a19c-4277-a123-da856e7ff441
# ╠═a3922176-be46-4c39-8e19-7ae1659a318c
# ╟─2d2f7e84-fd88-4f05-90d1-b17b305f7bee
# ╠═c53f5aba-5b87-4528-873f-7a36a673222f
# ╟─80b023f8-55a2-4312-b4ae-381f72885e45
# ╠═4384cd1a-179b-487d-97da-298172a50de1
# ╟─f4fd62b4-305a-46d3-9c95-9d6617b254b7
# ╟─c0119a87-10e1-4a12-b908-c1c09d862357
# ╟─4c1da79b-8389-4248-ba50-cfd7577aa6df
# ╠═ceb2145b-1d2e-4615-9494-076f525f842a
# ╠═d617ca19-d915-48a5-b282-5af0ec09b362
# ╟─84e9b251-c83e-4bb7-a065-6ca1a9cffb49
# ╟─337e609b-15d9-4ce3-82cd-5bceb0e479af
# ╟─e501bc02-a1f0-4a3c-9587-e81237a10bc8
# ╟─50983336-c945-4835-a9ee-b56a6476ad74
# ╠═0b10cb62-6a4d-4acf-aef1-f471e21ef5e8
# ╟─0cd872e4-357b-49f6-9212-3b02e09aa020
# ╠═fb697e38-a345-4297-90b8-0d686f4ecec3
# ╟─9228b077-9b18-4485-93a0-07c84edd7c48
# ╠═45ded54e-8ee3-4aa5-a1a5-79224a416b47
# ╟─8c0ea123-42db-4b95-bde8-46f2bafff2c9
# ╠═d762f8e2-853a-455c-9fc7-691e54a04593
# ╟─b444cb9d-987c-482f-a0c8-634538a8829f
# ╠═3796367c-41a7-4024-bfdb-afc5efbbb0cf
# ╠═99ce5436-50ac-4237-b2a8-bd6c575cba0d
# ╠═d9eb6d05-78fc-4c7b-a390-1ba4c1db1d47
# ╠═43fe8326-fd33-4869-9d0c-0ccfe4ec01cf
# ╠═6032d697-594b-4e64-a80e-1ae34ae22c9b
# ╟─14e78c3b-1e86-4d64-b280-938e3e181f36
# ╠═c7dbbbcd-65c5-4a0b-ad71-79f1b1ce1767
# ╟─0d10693a-a77a-4550-a1c1-e198ec11c985
# ╠═3d9ffa18-cb55-4bf1-b874-d7e1aa04355b
# ╠═1066f4c7-d463-4119-821c-e7c50e84a909
# ╠═6990e258-7dbe-47f2-b225-fe9b1ff6f607
# ╟─075c60a6-d1d3-4f64-9186-1fbf1a8abc08
# ╠═3ea09ae8-11b9-4834-ac51-aba7165e7cce
# ╠═df358406-3547-472c-8d12-02dea6719145
# ╠═5ff89b95-29c0-4b62-b55e-0d23afba43f5
# ╠═72efcca2-5300-4524-b2b5-09545bdacb92
# ╠═6c4dcde0-c72b-4562-b4d9-ea4e15eba84c
# ╠═2259e733-0210-4e4d-839d-8471c462c694
# ╠═62a09464-b6d1-4131-8565-a1754ba634d5
