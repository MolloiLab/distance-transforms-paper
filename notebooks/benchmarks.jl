### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 8401528c-5484-42f7-8f92-94dcca409526
using Pkg; Pkg.instantiate()

# ╔═╡ bbb7b211-fa87-489d-87dc-1890cd848812
# ╠═╡ show_logs = false
using DrWatson

# ╔═╡ 8cbf0cfd-6b41-4a2c-a4af-a1f43c7afe3f
# ╠═╡ show_logs = false
@quickactivate "hd-loss"

# ╔═╡ 0354c88d-a4e9-446a-b0dc-7bd36583120d
# ╠═╡ show_logs = false
begin
	using CUDA
	CUDA.set_runtime_version!(v"11.8")
end

# ╔═╡ 09db0c44-358a-4d3d-83e6-f6eed0696013
# ╠═╡ show_logs = false
using PlutoUI, BenchmarkTools, CairoMakie, DataFrames, CSV, CSVFiles, Statistics

# ╔═╡ 545b7f75-29b1-438e-883a-e2ded8053eed
using ImageMorphology: distance_transform, feature_transform

# ╔═╡ 7afb4707-888d-49ff-be62-2db208a57246
using DistanceTransformsPy: pytransform

# ╔═╡ 1e38338d-eaa8-4252-9a05-698abe4d7756
using DistanceTransforms: boolean_indicator, transform

# ╔═╡ 7e5e5691-0239-4c56-ad98-bce50d3d132c
using Losers: dice_loss, hausdorff_loss

# ╔═╡ cbf06207-8ad8-42fa-9758-602bcfe7e4aa
TableOfContents()

# ╔═╡ e53d73e1-126f-42bd-99df-87b59903eb70
begin
	range_size_2D = range(4, 4096, 20)
	range_size_3D = range(4, 256, 20)
	# range_size_2D = range(4, 512, 2)
	# range_size_3D = range(4, 64, 2)
end

# ╔═╡ 71da875a-8420-4f38-8ddd-eb68c110e6fd
md"""
# Distance Transforms
"""

# ╔═╡ 0e2eca71-ab2d-45d8-b4fe-963d9bf2f3b9
md"""
## 2D
"""

# ╔═╡ f3ec467e-ca19-4cd3-917b-2d11344e8785
# ╠═╡ disabled = true
#=╠═╡
begin
	sizes = Float64[]
	
	dt_scipy = Float64[]
	dt_scipy_std = Float64[]

	dt_maurer = Float64[]
	dt_maurer_std = Float64[]

	dt_fenz = Float64[]
	dt_fenz_std = Float64[]

	dt_proposed = Float64[]
	dt_proposed_std = Float64[]
	
	for _n in range_size_2D
		n = round(Int, _n)
		@info n
		push!(sizes, n^2)
		f = Float32.(rand([0, 1], n, n))
		
		# Python (Scipy)
		dt = @benchmark pytransform($f)
		push!(dt_scipy, BenchmarkTools.minimum(dt).time)
		push!(dt_scipy_std, BenchmarkTools.std(dt).time)

		# Maurer (ImageMorphology.jl)
		bool_f = Bool.(f)
		dt = @benchmark distance_transform($feature_transform($bool_f))
		push!(dt_maurer, BenchmarkTools.minimum(dt).time) # ns
		push!(dt_maurer_std, BenchmarkTools.std(dt).time)
		
		# Felzenszwalb (DistanceTransforms.jl)
		dt = @benchmark transform($boolean_indicator($f))
		push!(dt_fenz, BenchmarkTools.minimum(dt).time)
		push!(dt_fenz_std, BenchmarkTools.std(dt).time)

		# Proposed-GPU (DistanceTransforms.jl)
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			dt = @benchmark transform($boolean_indicator($f_cuda))
			push!(dt_proposed, BenchmarkTools.minimum(dt).time)
			push!(dt_proposed_std, BenchmarkTools.std(dt).time)
		end
	end
end
  ╠═╡ =#

# ╔═╡ 00bd4240-ad66-4aa2-914b-cdf64fe4d639
md"""
## 3D
"""

# ╔═╡ 37b3e0fc-80d1-427a-bd26-0a17f48db118
# ╠═╡ disabled = true
#=╠═╡
begin
	sizes_3D = Float64[]
	
	dt_scipy_3D = Float64[]
	dt_scipy_std_3D = Float64[]

	dt_maurer_3D = Float64[]
	dt_maurer_std_3D = Float64[]

	dt_fenz_3D = Float64[]
	dt_fenz_std_3D = Float64[]

	dt_proposed_3D = Float64[]
	dt_proposed_std_3D = Float64[]
	
	for _n in range_size_3D
		n = round(Int, _n)
		@info n
		push!(sizes_3D, n^3)
		f = Float32.(rand([0, 1], n, n, n))
		
		# Python (Scipy)
		dt = @benchmark pytransform($f)
		push!(dt_scipy_3D, BenchmarkTools.minimum(dt).time)
		push!(dt_scipy_std_3D, BenchmarkTools.std(dt).time)

		# Maurer (ImageMorphology.jl)
		bool_f = Bool.(f)
		dt = @benchmark distance_transform($feature_transform($bool_f))
		push!(dt_maurer_3D, BenchmarkTools.minimum(dt).time) # ns
		push!(dt_maurer_std_3D, BenchmarkTools.std(dt).time)
		
		# Felzenszwalb (DistanceTransforms.jl)
		dt = @benchmark transform($boolean_indicator($f))
		push!(dt_fenz_3D, BenchmarkTools.minimum(dt).time)
		push!(dt_fenz_std_3D, BenchmarkTools.std(dt).time)

		# Proposed-GPU (DistanceTransforms.jl)
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			dt = @benchmark transform($boolean_indicator($f_cuda))
			push!(dt_proposed_3D, BenchmarkTools.minimum(dt).time)
			push!(dt_proposed_std_3D, BenchmarkTools.std(dt).time)
		end
	end
end
  ╠═╡ =#

# ╔═╡ 57dc4b37-7165-4f97-89b6-c2238888db5a
#=╠═╡
let
	f = CairoMakie.Figure()
	ax = Axis(
		f[1, 1],
		title="2D"
	)

	scatterlines!(dt_scipy, label="Python")
	scatterlines!(dt_maurer, label="Maurer")
	scatterlines!(dt_fenz, label="Felzenszwalb")
	scatterlines!(dt_proposed, label="Proposed")

	ax = Axis(
		f[2, 1],
		title="3D"
	)
	scatterlines!(dt_scipy_3D, label="Python")
	scatterlines!(dt_maurer_3D, label="Maurer")
	scatterlines!(dt_fenz_3D, label="Felzenszwalb")
	scatterlines!(dt_proposed_3D, label="Proposed")

	f[1:2, 2] = Legend(f, ax, "Distance Transforms", framevisible = false)

	Label(f[0, 1:2]; text="Distance Transforms", fontsize=30)
	f
end
  ╠═╡ =#

# ╔═╡ 709116ce-64d2-4407-aad5-22c256cf9de7
# ╠═╡ disabled = true
#=╠═╡
begin
	global df_dt
	if has_cuda_gpu()
		df_dt = DataFrame(
			sizes = sizes,
			dt_scipy = dt_scipy,
			dt_scipy_std = dt_scipy_std,
			dt_maurer = dt_maurer,
			dt_maurer_std = dt_maurer_std,
			dt_fenz = dt_fenz,
			dt_fenz_std = dt_fenz_std,
			dt_proposed = dt_proposed,
			dt_proposed_std = dt_proposed_std,
		
			sizes_3D = sizes_3D,
			dt_scipy_3D = dt_scipy_3D,
			dt_scipy_std_3D = dt_scipy_std_3D,
			dt_maurer_3D = dt_maurer_3D,
			dt_maurer_std_3D = dt_maurer_std_3D,
			dt_fenz_3D = dt_fenz_3D,
			dt_fenz_std_3D = dt_fenz_std_3D,
			dt_proposed_3D = dt_proposed_3D,
			dt_proposed_std_3D = dt_proposed_std_3D,
		)
	else
		df_dt = DataFrame(
			sizes = sizes,
			dt_scipy = dt_scipy,
			dt_scipy_std = dt_scipy_std,
			dt_maurer = dt_maurer,
			dt_maurer_std = dt_maurer_std,
			dt_fenz = dt_fenz,
			dt_fenz_std = dt_fenz_std,
		
			sizes_3D = sizes_3D,
			dt_scipy_3D = dt_scipy_3D,
			dt_scipy_std_3D = dt_scipy_std_3D,
			dt_maurer_3D = dt_maurer_3D,
			dt_maurer_std_3D = dt_maurer_std_3D,
			dt_fenz_3D = dt_fenz_3D,
			dt_fenz_std_3D = dt_fenz_std_3D,
		)
	end
end
  ╠═╡ =#

# ╔═╡ df1e62e3-4ff7-4ca5-bce9-ce4a29c34ec9
# ╠═╡ disabled = true
#=╠═╡
save(datadir("results", "dts.csv"), df_dt)
  ╠═╡ =#

# ╔═╡ 11431e55-0f01-441a-8855-24952f8bbfcd
md"""
# Loss Functions
"""

# ╔═╡ e908389d-671f-4cea-a717-218cb6e00bd7
md"""
## 2D
"""

# ╔═╡ 9e11696a-9961-4ad1-a9a7-d70fdbd5423c
begin
	sizes_hd = Float64[]

	dice = Float64[]
	dice_std = Float64[]
	
	hd_scipy = Float64[]
	hd_scipy_std = Float64[]

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
		
		# Python Hausdorff Loss (Scipy)
		loss = @benchmark hausdorff_loss(
			$f, 
			$f2, 
			$pytransform($f), 
			$pytransform($f2)
		)
		push!(hd_scipy, BenchmarkTools.minimum(loss).time)
		push!(hd_scipy_std, BenchmarkTools.std(loss).time)

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

# ╔═╡ 1d51b3af-6819-4996-8eea-8285b5c7a1b3
md"""
## 3D
"""

# ╔═╡ 8327f001-b7e6-4f9c-8bef-21c4e992b7be
begin
	sizes_hd_3D = Float64[]

	dice_3D = Float64[]
	dice_std_3D = Float64[]

	hd_scipy_3D = Float64[]
	hd_scipy_std_3D = Float64[]
	
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
		
		# Python Hausdorff Loss (Scipy)
		loss = @benchmark hausdorff_loss(
			$f, 
			$f2, 
			$pytransform($f), 
			$pytransform($f2)
		)
		push!(hd_scipy_3D, BenchmarkTools.minimum(loss).time)
		push!(hd_scipy_std_3D, BenchmarkTools.std(loss).time)

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

# ╔═╡ 85014ff8-05e4-44bb-a8f6-e2e8cf92f81f
let
	f = CairoMakie.Figure()
	ax = Axis(
		f[1, 1],
		title="2D"
	)
	scatterlines!(dice, label="Dice Loss")
	scatterlines!(hd_scipy, label="Python Hausdorff Loss")
	scatterlines!(hd_maurer, label="Maurer Hausdorff Loss")
	scatterlines!(hd_fenz, label="Felzenszwalb Hausdorff Loss")
	scatterlines!(hd_proposed, label="Proposed Hausdorff Loss")
	

	ax = Axis(
		f[2, 1],
		title="3D"
	)
	scatterlines!(dice_3D, label="Dice Loss")
	scatterlines!(hd_scipy_3D, label="Python Hausdorff Loss")
	scatterlines!(hd_maurer_3D, label="Maurer Hausdorff Loss")
	scatterlines!(hd_fenz_3D, label="Felzenszwalb Hausdorff Loss")
	scatterlines!(hd_proposed_3D, label="Proposed Hausdorff Loss")

	f[1:2, 2] = Legend(f, ax, "Loss Functions", framevisible = false)

	Label(f[0, 1:2]; text="Loss Functions", fontsize=30)
	
	f
end

# ╔═╡ 036fcfad-ceba-4914-b2f8-744f22fb9969
begin
	global df_loss
	if has_cuda_gpu()
		df_loss = DataFrame(
			sizes_hd = sizes_hd,
			dice = dice,
			dice_std = dice_std,
			hd_scipy = hd_scipy,
			hd_scipy_std = hd_scipy_std,
			hd_maurer = hd_maurer,
			hd_maurer_std = hd_maurer_std,
			hd_fenz = hd_fenz,
			hd_fenz_std = hd_fenz_std,
			hd_proposed = hd_proposed,
			hd_proposed_std = hd_proposed_std,
		
			sizes_hd_3D = sizes_hd_3D,
			dice_3D = dice_3D,
			dice_std_3D = dice_std_3D,
			hd_scipy_3D = hd_scipy_3D,
			hd_scipy_std_3D = hd_scipy_std_3D,
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
			hd_scipy = hd_scipy,
			hd_scipy_std = hd_scipy_std,
			hd_maurer = hd_maurer,
			hd_maurer_std = hd_maurer_std,
			hd_fenz = hd_fenz,
			hd_fenz_std = hd_fenz_std,
		
			sizes_hd_3D = sizes_hd_3D,
			dice_3D = dice_3D,
			dice_std_3D = dice_std_3D,
			hd_scipy_3D = hd_scipy_3D,
			hd_maurer_3D = hd_maurer_3D,
			hd_maurer_std_3D = hd_maurer_std_3D,
			hd_scipy_std_3D = hd_scipy_std_3D,
			hd_fenz_3D = hd_fenz_3D,
			hd_fenz_std_3D = hd_fenz_std_3D,
		)
	end
end

# ╔═╡ 9368f91e-97d8-4452-9138-2c1e5965cd76
save(datadir("results", "losses.csv"), df_loss)

# ╔═╡ Cell order:
# ╠═bbb7b211-fa87-489d-87dc-1890cd848812
# ╠═8cbf0cfd-6b41-4a2c-a4af-a1f43c7afe3f
# ╠═8401528c-5484-42f7-8f92-94dcca409526
# ╠═0354c88d-a4e9-446a-b0dc-7bd36583120d
# ╠═09db0c44-358a-4d3d-83e6-f6eed0696013
# ╠═545b7f75-29b1-438e-883a-e2ded8053eed
# ╠═7afb4707-888d-49ff-be62-2db208a57246
# ╠═1e38338d-eaa8-4252-9a05-698abe4d7756
# ╠═7e5e5691-0239-4c56-ad98-bce50d3d132c
# ╠═cbf06207-8ad8-42fa-9758-602bcfe7e4aa
# ╠═e53d73e1-126f-42bd-99df-87b59903eb70
# ╟─71da875a-8420-4f38-8ddd-eb68c110e6fd
# ╟─0e2eca71-ab2d-45d8-b4fe-963d9bf2f3b9
# ╠═f3ec467e-ca19-4cd3-917b-2d11344e8785
# ╟─00bd4240-ad66-4aa2-914b-cdf64fe4d639
# ╠═37b3e0fc-80d1-427a-bd26-0a17f48db118
# ╟─57dc4b37-7165-4f97-89b6-c2238888db5a
# ╠═709116ce-64d2-4407-aad5-22c256cf9de7
# ╠═df1e62e3-4ff7-4ca5-bce9-ce4a29c34ec9
# ╟─11431e55-0f01-441a-8855-24952f8bbfcd
# ╟─e908389d-671f-4cea-a717-218cb6e00bd7
# ╠═9e11696a-9961-4ad1-a9a7-d70fdbd5423c
# ╟─1d51b3af-6819-4996-8eea-8285b5c7a1b3
# ╠═8327f001-b7e6-4f9c-8bef-21c4e992b7be
# ╟─85014ff8-05e4-44bb-a8f6-e2e8cf92f81f
# ╠═036fcfad-ceba-4914-b2f8-744f22fb9969
# ╠═9368f91e-97d8-4452-9138-2c1e5965cd76
