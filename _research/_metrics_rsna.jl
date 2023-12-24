### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 8775ec12-de47-11ed-129a-190e18306f76
begin
	# Julia 1.8.4
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
	# Pkg.add(url = "https://github.com/Dale-Black/ComputerVisionMetrics.jl")
	# Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl", rev="wenbo")
end

# ╔═╡ c61589ef-04a6-4696-b061-2e3924ff86ab
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
	using Printf
	using DataFrames
	using CSV
	using ComputerVisionMetrics
	using FastAI, FastVision, Flux, Metalhead
	import CairoMakie; CairoMakie.activate!(type="png")
end

# ╔═╡ 089209e5-7770-41e9-a77f-06a13fe113e9
begin
	mask_dir = raw"E:\RSNA_compressed_data\96_96_96\segmentations"
	train_image_dir = raw"E:\RSNA_compressed_data\96_96_96\train_images"
	mask_names = []
	for f in readdir(mask_dir, join=false)
	    push!(mask_names, splitext(splitext(f)[1])[1])
	end
end

# ╔═╡ 266e17d5-88fa-408a-8cf1-31611a615df2
begin
	matched_names = []
	ct = 0
	for img in readdir(train_image_dir, join=false)
	    f = splitext(img)[1]
	    if f in mask_names
	        ct += 1
	        push!(matched_names, f)
	    end
	end
	println("Found $ct of $(size(mask_names)[1])")
end

# ╔═╡ b043ed2b-f97e-4caf-8351-15c297727c43
begin
	image_size = (96, 96, 96)
	container_images = Array{Float32, 4}(undef, image_size..., ct)
	container_masks = Array{Float32, 4}(undef, image_size..., ct)
	# Load 87 images from saved data.
	Threads.@threads for i = 1 : ct
	    curr_path = @sprintf "%s\\%s.txt" train_image_dir matched_names[i]
	    temp = zeros(image_size)
	    read!(curr_path, temp)
	    container_images[:,:,:, i] = deepcopy(temp)
	end
	# Normalize images
	for i = 1 : 87
	    curr_img = container_images[:,:,:, i]
	    curr_max, curr_min = maximum(curr_img), minimum(curr_img)
	    curr_img = (curr_img .- curr_min) ./ (curr_max - curr_min)
	    container_images[:,:,:, i] = curr_img
	end
	# Load 87 masks from saved data.
	Threads.@threads for i = 1 : ct
	    curr_path = @sprintf "%s\\%s.nii.txt" mask_dir matched_names[i]
	    temp = zeros(image_size)
	    read!(curr_path, temp)
	    container_masks[:,:,:, i] = deepcopy(temp)
	end
	container_masks = Int64.(round.(container_masks) .+ 1)
	data_ready = (container_images, container_masks);
	
	image1, mask1 = sample = (container_images[:,:,:,1], container_masks[:,:,:,1]);
	
	data_mean, data_std = FastVision.imagedatasetstats(container_images, Gray{N0f8}) 
	data_mean, data_std = StaticArraysCore.SVector{1, Float32}(data_mean[1]), StaticArraysCore.SVector{1, Float32}(data_std[1])
	
	task = SupervisedTask(
	    (FastVision.Image{3}(), Mask{3}(1:8)), # 1: background, 2-8: 7 classes
	    (
	        FastVision.ProjectiveTransforms((image_size)),
	        ImagePreprocessing(means = data_mean, stds = data_std, C = Gray{N0f8}),
	        FastAI.OneHot()
	    )
	)
	
	batch_size = 1
	
	@assert checkblock(task.blocks.sample, sample)
	train_files, val_files = MLDataPattern.splitobs(data_ready, 0.8)
	tdl, vdl = FastAI.taskdataloaders(train_files, val_files, task, batch_size);
end

# ╔═╡ 16880cb8-66b6-49a4-a15c-899c578e0106
begin
	dice_metric_of_dice_model_train_set = []
	hd_metric_of_dice_model_train_set = []
	dice_metric_of_HD_Dice_model_train_set = []
	hd_metric_of_HD_Dice_model_train_set = []
	
	dice_metric_of_dice_model_valid_set = []
	hd_metric_of_dice_model_valid_set = []
	dice_metric_of_HD_Dice_model_valid_set = []
	hd_metric_of_HD_Dice_model_valid_set = []
	
	for epoch_idx = 10 : 10 : 10
	    print("$epoch_idx...\t")
	    # Load models
	    DICE_model_path = string("E:/RSNA_savedmodels/3_bigger_NN_0.0001_Dice_", epoch_idx, ".jld2")
	    HD_DICE_model_path = string("E:/RSNA_savedmodels/bigger_NN_0.001_HD_Dice_", epoch_idx, ".jld2")
	    _ , model_Dice = loadtaskmodel(DICE_model_path)
	    _ , model_HD_Dice = loadtaskmodel(HD_DICE_model_path)
	    # Get metrics
	
	    # Traverse train dataloader
	    curr_dice_metric_of_dice_model = []
	    curr_hd_metric_of_dice_model = []
	    curr_dice_metric_of_HD_Dice_model = []
	    curr_hd_metric_of_HD_Dice_model = []
	    for (img, _mask) in tdl
	        mask = _mask .>= 0.5
	
	        # Predictions
	        pred_mask_DICE = model_Dice(img) .>= 0.5
	        pred_mask_HD_DICE = model_HD_Dice(img) .>= 0.5
	
	        for channel_idx = 2:8
	            # Dice metric for Dice model
	            curr_curr_dice_metric_of_dice_model = ComputerVisionMetrics.dice(pred_mask_DICE[:,:,:,i,1], mask[:,:,:,i,1])
	    
	            # HD metric for Dice model
	            curr_curr_hd_metric_of_dice_model = ComputerVisionMetrics.hausdorff(pred_mask_DICE[:,:,:,i,1], mask[:,:,:,i,1])
	    
	            # Dice metric for HD+Dice model
	            curr_curr_dice_metric_of_HD_Dice_model = ComputerVisionMetrics.dice(pred_mask_HD_DICE[:,:,:,i,1], mask[:,:,:,i,1])
	    
	            # HD metric for HD+Dice model
	            curr_curr_hd_metric_of_HD_Dice_model = ComputerVisionMetrics.hausdorff(pred_mask_HD_DICE[:,:,:,i,1], mask[:,:,:,i,1])
	    
	            # Record
	            push!(curr_dice_metric_of_dice_model, curr_curr_dice_metric_of_dice_model)
	            push!(curr_hd_metric_of_dice_model, curr_curr_hd_metric_of_dice_model)
	            push!(curr_dice_metric_of_HD_Dice_model, curr_curr_dice_metric_of_HD_Dice_model)
	            push!(curr_hd_metric_of_HD_Dice_model, curr_curr_hd_metric_of_HD_Dice_model)
	        end
	    end
	    push!(dice_metric_of_dice_model_train_set, mean(curr_dice_metric_of_dice_model))
	    push!(hd_metric_of_dice_model_train_set, mean(curr_hd_metric_of_dice_model))
	    push!(dice_metric_of_HD_Dice_model_train_set, mean(curr_dice_metric_of_HD_Dice_model))
	    push!(hd_metric_of_HD_Dice_model_train_set, mean(curr_hd_metric_of_HD_Dice_model))
	
	
	    curr_dice_metric_of_dice_model = []
	    curr_hd_metric_of_dice_model = []
	    curr_dice_metric_of_HD_Dice_model = []
	    curr_hd_metric_of_HD_Dice_model = []
	    # Traverse valid dataloader
	    for (img, mask) in vdl
	        mask = _mask .>= 0.5
	
	        # Predictions
	        pred_mask_DICE = model_Dice(img) .>= 0.5
	        pred_mask_HD_DICE = model_HD_Dice(img) .>= 0.5
	
	        for channel_idx = 2:8
	            # Dice metric for Dice model
	            curr_curr_dice_metric_of_dice_model = ComputerVisionMetrics.dice(pred_mask_DICE[:,:,:,i,1], mask[:,:,:,i,1])
	    
	            # HD metric for Dice model
	            curr_curr_hd_metric_of_dice_model = ComputerVisionMetrics.hausdorff(pred_mask_DICE[:,:,:,i,1], mask[:,:,:,i,1])
	    
	            # Dice metric for HD+Dice model
	            curr_curr_dice_metric_of_HD_Dice_model = ComputerVisionMetrics.dice(pred_mask_HD_DICE[:,:,:,i,1], mask[:,:,:,i,1])
	    
	            # HD metric for HD+Dice model
	            curr_curr_hd_metric_of_HD_Dice_model = ComputerVisionMetrics.hausdorff(pred_mask_HD_DICE[:,:,:,i,1], mask[:,:,:,i,1])
	    
	            # Record
	            push!(curr_dice_metric_of_dice_model, curr_curr_dice_metric_of_dice_model)
	            push!(curr_hd_metric_of_dice_model, curr_curr_hd_metric_of_dice_model)
	            push!(curr_dice_metric_of_HD_Dice_model, curr_curr_dice_metric_of_HD_Dice_model)
	            push!(curr_hd_metric_of_HD_Dice_model, curr_curr_hd_metric_of_HD_Dice_model)
	        end
	    end
	    push!(dice_metric_of_dice_model_valid_set, mean(curr_dice_metric_of_dice_model))
	    push!(hd_metric_of_dice_model_valid_set, mean(curr_hd_metric_of_dice_model))
	    push!(dice_metric_of_HD_Dice_model_valid_set, mean(curr_dice_metric_of_HD_Dice_model))
	    push!(hd_metric_of_HD_Dice_model_valid_set, mean(curr_hd_metric_of_HD_Dice_model))
	    curr_dice_metric_of_dice_model = nothing
	    curr_hd_metric_of_dice_model = nothing
	    curr_dice_metric_of_HD_Dice_model = nothing
	    curr_hd_metric_of_HD_Dice_model = nothing
	end
	println()
end

# ╔═╡ 80132a5b-a278-4d87-8d03-2dff235063bd
md"""
# Record
"""

# ╔═╡ e6c714f3-3668-4be7-b9fa-381e40d9092e
begin
	df_metrics = DataFrame(dice_metric_of_dice_model_train_set = dice_metric_of_dice_model_train_set,
	    hd_metric_of_dice_model_train_set = hd_metric_of_dice_model_train_set,
	    dice_metric_of_HD_Dice_model_train_set = dice_metric_of_HD_Dice_model_train_set,
	    hd_metric_of_HD_Dice_model_train_set = hd_metric_of_HD_Dice_model_train_set,
	
	    dice_metric_of_dice_model_valid_set = dice_metric_of_dice_model_valid_set,
	    hd_metric_of_dice_model_valid_set = hd_metric_of_dice_model_valid_set,
	    dice_metric_of_HD_Dice_model_valid_set = dice_metric_of_HD_Dice_model_valid_set,
	    hd_metric_of_HD_Dice_model_valid_set = hd_metric_of_HD_Dice_model_valid_set)
	CSV.write("E:/CSVs/metrics_RSNA.csv", df_metrics)
end

# ╔═╡ Cell order:
# ╠═8775ec12-de47-11ed-129a-190e18306f76
# ╠═c61589ef-04a6-4696-b061-2e3924ff86ab
# ╠═089209e5-7770-41e9-a77f-06a13fe113e9
# ╠═266e17d5-88fa-408a-8cf1-31611a615df2
# ╠═b043ed2b-f97e-4caf-8351-15c297727c43
# ╠═16880cb8-66b6-49a4-a15c-899c578e0106
# ╟─80132a5b-a278-4d87-8d03-2dff235063bd
# ╠═e6c714f3-3668-4be7-b9fa-381e40d9092e
