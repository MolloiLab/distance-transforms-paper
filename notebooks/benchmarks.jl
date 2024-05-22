### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 8401528c-5484-42f7-8f92-94dcca409526
# ╠═╡ show_logs = false
using Pkg; Pkg.activate(".."), Pkg.instantiate();

# ╔═╡ 0354c88d-a4e9-446a-b0dc-7bd36583120d
# ╠═╡ show_logs = false
begin
	using CUDA
	# CUDA.set_runtime_version!(v"11.8")
end

# ╔═╡ bbb7b211-fa87-489d-87dc-1890cd848812
# ╠═╡ show_logs = false
using DrWatson: datadir

# ╔═╡ 09db0c44-358a-4d3d-83e6-f6eed0696013
# ╠═╡ show_logs = false
using PlutoUI, BenchmarkTools, CairoMakie, DataFrames, CSV, CSVFiles, Statistics

# ╔═╡ 545b7f75-29b1-438e-883a-e2ded8053eed
using ImageMorphology: distance_transform, feature_transform

# ╔═╡ 1e38338d-eaa8-4252-9a05-698abe4d7756
using DistanceTransforms: boolean_indicator, transform

# ╔═╡ 1d2871ea-be83-40d4-9811-f01d99f0e92b
md"""
# TODO

- Add single-threaded timings
- Import proper packages (CUDA.jl, AMDGPU.jl, oneAPI.jl, and Metal.jl)
- Modify GPU timings to save specific to various backends (CUDA, AMDGPU, Metal, OneAPI)
"""

# ╔═╡ cbf06207-8ad8-42fa-9758-602bcfe7e4aa
TableOfContents()

# ╔═╡ e53d73e1-126f-42bd-99df-87b59903eb70
begin
	# range_size_2D = range(4, 4096, 20)
	# range_size_3D = range(4, 256, 20)
	range_size_2D = range(4, 512, 2)
	range_size_3D = range(4, 64, 2)
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
begin
	sizes = Float64[]

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

# ╔═╡ 00bd4240-ad66-4aa2-914b-cdf64fe4d639
md"""
## 3D
"""

# ╔═╡ 37b3e0fc-80d1-427a-bd26-0a17f48db118
begin
	sizes_3D = Float64[]

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
begin
	global df_dt
	if has_cuda_gpu()
		df_dt = DataFrame(
			sizes = sizes,
			dt_maurer = dt_maurer,
			dt_maurer_std = dt_maurer_std,
			dt_fenz = dt_fenz,
			dt_fenz_std = dt_fenz_std,
			dt_proposed = dt_proposed,
			dt_proposed_std = dt_proposed_std,
		
			sizes_3D = sizes_3D,
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
			dt_maurer = dt_maurer,
			dt_maurer_std = dt_maurer_std,
			dt_fenz = dt_fenz,
			dt_fenz_std = dt_fenz_std,
		
			sizes_3D = sizes_3D,
			dt_maurer_3D = dt_maurer_3D,
			dt_maurer_std_3D = dt_maurer_std_3D,
			dt_fenz_3D = dt_fenz_3D,
			dt_fenz_std_3D = dt_fenz_std_3D,
		)
	end
end

# ╔═╡ df1e62e3-4ff7-4ca5-bce9-ce4a29c34ec9
# ╠═╡ disabled = true
#=╠═╡
save(datadir("results", "dts.csv"), df_dt)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─1d2871ea-be83-40d4-9811-f01d99f0e92b
# ╠═8401528c-5484-42f7-8f92-94dcca409526
# ╠═0354c88d-a4e9-446a-b0dc-7bd36583120d
# ╠═bbb7b211-fa87-489d-87dc-1890cd848812
# ╠═09db0c44-358a-4d3d-83e6-f6eed0696013
# ╠═545b7f75-29b1-438e-883a-e2ded8053eed
# ╠═1e38338d-eaa8-4252-9a05-698abe4d7756
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
