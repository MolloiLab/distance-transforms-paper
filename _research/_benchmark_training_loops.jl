### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ c3821f1e-f7e4-4dc2-93e7-dff1b0749ddb
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	# Pkg.add("CUDA")
	# Pkg.add("Glob")
	# Pkg.add("Flux")
	# Pkg.add("NIfTI")
	# Pkg.add("Images")
	# Pkg.add("FastAI")
	# Pkg.add("Metalhead")
	# Pkg.add("FastVision")
	# Pkg.add("CairoMakie")
	# Pkg.add("StaticArrays")
	# Pkg.add("MLDataPattern")
	# Pkg.add("BenchmarkTools")
	# Pkg.add("ChainRulesCore")
	# Pkg.add("BSON")
	# Pkg.add("DataFrames")
	# Pkg.add("CSV")
	Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl", rev="master")
end

# ╔═╡ 65c6a0e2-bf72-4320-9e09-b2955aee80ff
begin
	using CUDA
	using Glob
	using Dates
	using NIfTI
	using Images
	using Statistics
	using StaticArrays
	using MLDataPattern
	using BenchmarkTools
	using ChainRulesCore
	using DistanceTransforms
	using DataFrames
	using CSV
	using FastAI, FastVision, Flux, Metalhead
	import CairoMakie; CairoMakie.activate!(type="png")
end

# ╔═╡ d86e89a0-da38-11ed-0628-bb1a64e58171
md"""
# 1. Set up notebook(Please change to Dr.Watson quickactivate here if needed)
"""

# ╔═╡ 4138e680-7be8-43cc-b7cf-4c35db1df82c
md"""
# 2. Load 'Task02_heart' into dataloaders and create models, etc.
"""

# ╔═╡ 1804332a-bff5-4ed2-ae54-c907c7371b09
begin
	#=--------------------- Prepare Data ---------------------=#
	data_dir = raw"C:\Users\wenbl13\OneDrive - UCI Health\Desktop\Task02_Heart"

	function loadfn_label(p)
	    a = NIfTI.niread(string(p)).raw
	    convert_a = convert(Array{UInt8}, a)
	    convert_a = convert_a .+ 1
	    return convert_a
	end
	
	function loadfn_image(p)
	    a = NIfTI.niread(string(p)).raw
	    convert_a = convert(Array{Float32}, a)
	    convert_a = convert_a / max(convert_a...)
	    return convert_a
	end

	images(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
	masks(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
	pre_data = (
	    images(joinpath(data_dir, "imagesTr")),
	    masks(joinpath(data_dir, "labelsTr")),
	)

	image_size = (96, 96, 96)

	function presize(files)
	    container_images = Array{Float32,4}(undef, image_size..., numobs(files))
	    container_masks = Array{Int64,4}(undef, image_size..., numobs(files))
	    for i in 1:numobs(files)
	        image, mask = FastAI.getobs(files, i)
	        img = imresize(image, image_size)
	        msk = round.(imresize(mask, image_size))
	        container_images[:, :, :, i] = img
	        container_masks[:, :, :, i] = msk
	    end
	    return container_images, container_masks
	end

	img_container, mask_container = presize(pre_data)
	data_resized = (img_container,mask_container)
	
	a, b = FastVision.imagedatasetstats(img_container, Gray{N0f8}) 
	means, stds = SVector{1, Float32}(a[1]), SVector{1, Float32}(b[1])

	task = SupervisedTask(
		    (FastVision.Image{3}(), Mask{3}(1:2)),
		    (
		        ProjectiveTransforms((image_size)),
		        ImagePreprocessing(means = means, stds = stds, C = Gray{N0f8}),
		        # ImagePreprocessing(C = Gray{N0f8}),
		        FastAI.OneHot()
		    )
		)

	train_files, val_files = MLDataPattern.splitobs(data_resized, 0.8)

	batch_size = 4
	tdl, vdl = FastAI.taskdataloaders(train_files, val_files, task, batch_size)





	
	#=--------------------- Data Loader ---------------------=#
	# # CPU
	# traindl = Tuple{Tuple{Array{Float32, 5}, Array{Float32, 5}}, Array{Float32, 5}}[] 
	# validdl = Tuple{Tuple{Array{Float32, 5}, Array{Float32, 5}}, Array{Float32, 5}}[] 
	
	# for (xs, ys) in tdl
	#     ys_dt = DistanceTransforms.transform(true, ys, Wenbo(), 0)
	#     push!(traindl, ((xs,ys), ys_dt))
	# end
	
	# for (xs, ys) in vdl
	#     ys_dt = DistanceTransforms.transform(true, ys, Wenbo(), 0)
	#     push!(validdl, ((xs,ys), ys_dt))
	# end

	# GPU
	traindl = Tuple{Tuple{CuArray{Float32, 5, CUDA.Mem.DeviceBuffer}, CuArray{Float32, 5, CUDA.Mem.DeviceBuffer}}, CuArray{Float32, 5, CUDA.Mem.DeviceBuffer}}[] 
	validdl = Tuple{Tuple{CuArray{Float32, 5, CUDA.Mem.DeviceBuffer}, CuArray{Float32, 5, CUDA.Mem.DeviceBuffer}}, CuArray{Float32, 5, CUDA.Mem.DeviceBuffer}}[] 
	
	
	for (xs, ys) in tdl
		ys_dt = DistanceTransforms.transform(true, ys, Wenbo(), 0)
	    push!(traindl, ((CuArray(xs),CuArray(ys)), CuArray(ys_dt)))
	end
	
	for (xs, ys) in vdl
	    ys_dt = DistanceTransforms.transform(true, ys, Wenbo(), 0)
	    push!(validdl, ((CuArray(xs),CuArray(ys)), CuArray(ys_dt)))
	end
	tdl = nothing
	vdl = nothing


	

	#=--------------------- U-Net Model ---------------------=#
	conv = (stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=SamePad())
	tran = (stride, in, out) -> ConvTranspose((3, 3, 3), in=>out, stride=stride, pad=SamePad())
	
	conv1 = (in, out) -> Chain(conv(1, in, out), BatchNorm(out, leakyrelu))
	conv2 = (in, out) -> Chain(conv(2, in, out), BatchNorm(out, leakyrelu))
	conv3 = (in, out) -> Chain(conv(1, in, out), x -> softmax(x; dims = 4))
	tran2 = (in, out) -> Chain(tran(2, in, out), BatchNorm(out, leakyrelu))
	
	
	
	function unet3D_mini(in_chs, lbl_chs)
	    # Contracting layers
	    l1 = Chain(conv1(in_chs, 16))
	    l2 = Chain(l1, conv2(16, 32), conv1(32, 32))
	    l3 = Chain(l2, conv2(32, 64), conv1(64, 64))
	    l4 = Chain(l3, conv2(64, 128), conv1(128, 128))
	    l5 = Chain(l4, conv2(128, 256), conv1(256, 256))
	
	    # Expanding layers
	    l6 = Chain(l5, tran2(256, 128))
	    l7 = Chain(Parallel(FastVision.Models.catchannels,l4,l6), conv1(256, 128), tran2(128, 64))
	    l8 = Chain(Parallel(FastVision.Models.catchannels,l3,l7), conv1(128, 64), tran2(64, 32))
	    l9 = Chain(Parallel(FastVision.Models.catchannels,l2,l8), conv1(64, 32), tran2(32, 16))
	    l10 = Chain(l9, conv3(16, lbl_chs))
	end

	
	# _conv = (stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=SamePad())
	# _tran = (stride, in, out) -> ConvTranspose((2, 2, 2), in=>out, stride=stride, pad=SamePad())
	
	# conv1 = (in, out) -> Chain(_conv(1, in, out), BatchNorm(out, leakyrelu))
	# conv2 = (in, out) -> Chain(_conv(2, in, out), BatchNorm(out, leakyrelu))
	# conv3 = (in, out) -> Chain(_conv(1, in, out), x -> softmax(x; dims = 4))
	# tran2 = (in, out) -> Chain(_tran(2, in, out), BatchNorm(out, leakyrelu))
	
	
	
	function unet3D(in_chs, lbl_chs)
	    # Contracting layers
	    l1 = Chain(conv1(in_chs, 8), conv1(8, 32))
	    l2 = Chain(l1, MaxPool((2,2,2), stride=2), conv1(32, 32), conv1(32, 64))
	    l3 = Chain(l2, MaxPool((2,2,2), stride=2), conv1(64, 64), conv1(64, 128))
	    l4 = Chain(l3, MaxPool((2,2,2), stride=2), conv1(128, 128), conv1(128, 256), tran2(256, 256))
	
	    # # Expanding layers
	    l5 = Chain(Parallel(FastVision.Models.catchannels,l4,l3), 
	                conv1(128+256, 128),
	                conv1(128, 128),
	                tran2(128, 128))
	    l6 = Chain(Parallel(FastVision.Models.catchannels,l5,l2), 
	                conv1(64+128, 64),
	                conv1(64, 64),
	                tran2(64, 64))
	    l7 = Chain(Parallel(FastVision.Models.catchannels,l6,l1), 
	                conv1(32+64, 32),
	                conv1(32, 32),
	                conv3(32, lbl_chs))
	end
end

# ╔═╡ d2a6f657-4fe0-4012-8323-f1f19019949c
begin
	#=--------------------- Loss Functions ---------------------=#
	function dice_loss(ŷ, y; ϵ=1.0f-5)
	    num_channels = size(ŷ)[4]
	    loss_dice = 0f0
	    for chan_idx = 1 : num_channels
	        @inbounds loss_dice += 
	        1.0f0 - (muladd(2.0f0, sum(ŷ[:,:,:,chan_idx,:] .* y[:,:,:,chan_idx,:]), ϵ) / (sum(ŷ[:,:,:,chan_idx,:] .^ 2) + sum(y[:,:,:,chan_idx,:] .^ 2) + ϵ))
	    end
	    loss_dice /= num_channels
	    return loss_dice
	end

	function dice_hausdorff_loss_Scipy(ŷ, y, y_dtm, epoch_idx; ϵ=1.0f-5)
	    num_channels = size(ŷ)[4]
	    num_batches = size(ŷ)[5]
	    ŷ_dtm = similar(ŷ)
	    ignore_derivatives() do
	        for chan_idx = 1 : num_channels
	            for batch_idx = 1 : num_batches
	                ŷ_dtm[:,:,:, chan_idx, batch_idx] = 
						DistanceTransformsPy.transform(round.(@views(ŷ[:,:,:, chan_idx, batch_idx])), Scipy())
	            end
	        end
	        ŷ_dtm = round.(ŷ_dtm .^ 2)
	    end
	
	    loss_hd = mean(((ŷ .- y) .^ 2) .* (ŷ_dtm .+ y_dtm))
	    loss_dice = dice_loss(ŷ, y)
	    loss_hd < 1f5 || return loss_dice
	    α = (epoch_idx-1.0f0) * 2f-4
	    loss = α * loss_hd + (1.0f0 - α) * loss_dice
	    return loss
	end

	function dice_hausdorff_loss_Felzenszwalb(ŷ, y, y_dtm, epoch_idx; ϵ=1.0f-5)
	    num_channels = size(ŷ)[4]
	    num_batches = size(ŷ)[5]
	    ŷ_dtm = similar(ŷ)
	    ignore_derivatives() do
	        for chan_idx = 1 : num_channels
	            for batch_idx = 1 : num_batches
	                ŷ_dtm[:,:,:, chan_idx, batch_idx] = 
						DistanceTransforms.transform(round.(@views(ŷ[:,:,:, chan_idx, batch_idx])), Felzenszwalb(), 16)
	            end
	        end
	        ŷ_dtm = round.(ŷ_dtm .^ 2)
	    end
	
	    loss_hd = mean(((ŷ .- y) .^ 2) .* (ŷ_dtm .+ y_dtm))
	    loss_dice = dice_loss(ŷ, y)
	    loss_hd < 1f5 || return loss_dice
	    α = (epoch_idx-1.0f0) * 2f-4
	    loss = α * loss_hd + (1.0f0 - α) * loss_dice
	    return loss
	end
end

# ╔═╡ 8544d4ed-68b0-4edd-8c46-3c99297fe8e0
begin
	#=--------------------- Custom Loops ---------------------=#
	function train_1_epoch_with_DICE(epoch_idx, model, model_ps, train_dl, valid_dl, optimizer)
		curr_epoch = @timed begin
		    # Epoch start
		    for (img_mask, mask_dtm) in train_dl
		      img, mask = img_mask 
		      # Step start
		      gs = gradient(model_ps) do
		        pred_mask = model(img)
		        training_loss = dice_loss(pred_mask, mask)
				println(training_loss)
		        return training_loss
		      end
		      Flux.update!(optimizer, model_ps, gs)
		      # Step finished
		    end
		    # Epoch finished
		end
		println()
	    return (curr_epoch.time - curr_epoch.gctime)*10^9
	end

	function train_1_epoch_with_HD_DICE_Scipy(epoch_idx, model, model_ps, train_dl, valid_dl, optimizer)
		curr_epoch = @timed begin
	    # Epoch start
	    ct = 1
	    for (img_mask, mask_dtm) in train_dl
		      img, mask = img_mask 
		      # Step start
		      gs = gradient(model_ps) do
		        pred_mask = model(img)
		        training_loss = dice_hausdorff_loss_Scipy(pred_mask, mask, mask_dtm, epoch_idx)
		        return training_loss
		      end
		      Flux.update!(optimizer, model_ps, gs)
		      # Step finished
		    end
		    # Epoch finished
		end
	    return (curr_epoch.time - curr_epoch.gctime)*10^9
	end

	function train_1_epoch_with_HD_DICE_Felzenszwalb(epoch_idx, model, model_ps, train_dl, valid_dl, optimizer)
		curr_epoch = @timed begin
	    # Epoch start
	    ct = 1
	    for (img_mask, mask_dtm) in train_dl
		      img, mask = img_mask 
		      # Step start
		      gs = gradient(model_ps) do
		        pred_mask = model(img)
		        training_loss = dice_hausdorff_loss_Felzenszwalb(pred_mask, mask, mask_dtm, epoch_idx)
		        return training_loss
		      end
		      Flux.update!(optimizer, model_ps, gs)
		      # Step finished
		    end
		    # Epoch finished
		end
	    return (curr_epoch.time - curr_epoch.gctime)*10^9
	end
	
	
end

# ╔═╡ 9c87ab0b-8521-4914-871b-95f9e86f382f
md"""
# 3. Benchmark
"""

# ╔═╡ 052a137b-c393-4047-ae7c-57f3f01d590b
# Create an inital model
model_org = unet3D_mini(1, 2);

# ╔═╡ b7be55de-9ccb-4e8b-97a4-c0e8ba1167c9
md"""
# 3.1 Pure Dice
"""

# ╔═╡ 16f2dfec-eb48-4401-8f93-aa231142c3c4
begin
	model1 = model_org |> gpu
	model_ps1 = Flux.params(model1)
	optimizer = Adam(0.001) 
	Dice_times = []
	for epoch_idx = 1:10
	        curr_time = train_1_epoch_with_DICE(epoch_idx, model1, model_ps1, traindl, validdl, optimizer)
		push!(Dice_times, curr_time)
    end
end


# ╔═╡ f15c0563-713b-4bf1-b477-c43ebf5e02cf
Dice_times

# ╔═╡ 37f95f77-390e-4ffc-9094-21f94ac67df8
md"""
# 3.2 Dice + HD(Scipy)
"""

# ╔═╡ bfc90fc1-a432-453a-b1f3-ab0005de2d7b
begin
	model2 = model_org |> gpu
	model_ps2 = Flux.params(model2)
	optimizer2 = Adam(0.001) 
	Dice_HD_Scipy_times = []
	for epoch_idx = 1:10
	        curr_time = train_1_epoch_with_HD_DICE_Scipy(epoch_idx, model2, model_ps2, traindl, validdl, optimizer2)
		push!(Dice_HD_Scipy_times, curr_time)
    end
end


# ╔═╡ 635783ee-9966-40c3-8f67-7703efc97a47
Dice_HD_Scipy_times

# ╔═╡ a48f20ad-4c5a-491f-8fca-71ba04bd7283
md"""
# 3.3 Dice + HD(Felzenszwalb)
"""

# ╔═╡ b1aa165b-2f00-4438-ae02-80ca392706bb
begin
	model3 = model_org |> gpu
	model_ps3 = Flux.params(model3)
	optimizer3 = Adam(0.001) 
	Dice_HD_Felzenszwalb_times = []
	for epoch_idx = 1:10
	        curr_time = train_1_epoch_with_HD_DICE_Felzenszwalb(epoch_idx, model3, model_ps3, traindl, validdl, optimizer3)
		push!(Dice_HD_Felzenszwalb_times, curr_time)
    end
end

# ╔═╡ 0eb3437f-6069-4652-b7d0-1c3484c48a7e
Dice_HD_Felzenszwalb_times

# ╔═╡ bccabec5-ca78-426b-9fd9-e5b118a52060
md"""
# 4. Record Data
"""

# ╔═╡ Cell order:
# ╟─d86e89a0-da38-11ed-0628-bb1a64e58171
# ╠═c3821f1e-f7e4-4dc2-93e7-dff1b0749ddb
# ╠═65c6a0e2-bf72-4320-9e09-b2955aee80ff
# ╟─4138e680-7be8-43cc-b7cf-4c35db1df82c
# ╠═1804332a-bff5-4ed2-ae54-c907c7371b09
# ╠═d2a6f657-4fe0-4012-8323-f1f19019949c
# ╠═8544d4ed-68b0-4edd-8c46-3c99297fe8e0
# ╟─9c87ab0b-8521-4914-871b-95f9e86f382f
# ╠═052a137b-c393-4047-ae7c-57f3f01d590b
# ╟─b7be55de-9ccb-4e8b-97a4-c0e8ba1167c9
# ╠═16f2dfec-eb48-4401-8f93-aa231142c3c4
# ╠═f15c0563-713b-4bf1-b477-c43ebf5e02cf
# ╟─37f95f77-390e-4ffc-9094-21f94ac67df8
# ╠═bfc90fc1-a432-453a-b1f3-ab0005de2d7b
# ╠═635783ee-9966-40c3-8f67-7703efc97a47
# ╟─a48f20ad-4c5a-491f-8fca-71ba04bd7283
# ╠═b1aa165b-2f00-4438-ae02-80ca392706bb
# ╠═0eb3437f-6069-4652-b7d0-1c3484c48a7e
# ╟─bccabec5-ca78-426b-9fd9-e5b118a52060
