### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ bbb7b211-fa87-489d-87dc-1890cd848812
using DrWatson

# ╔═╡ 8cbf0cfd-6b41-4a2c-a4af-a1f43c7afe3f
# ╠═╡ show_logs = false
@quickactivate "hd-loss"

# ╔═╡ a0bcf2cb-ceb7-400e-b2b8-56ed98fc705a
using PlutoUI, BenchmarkTools, CairoMakie, DataFrames, CSV, DistanceTransforms, CUDA, Losers, DistanceTransformsPy

# ╔═╡ cbf06207-8ad8-42fa-9758-602bcfe7e4aa
TableOfContents()

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
	sizes = []
	
	dt_scipy = []
	dt_scipy_std = []

	dt_fenz = []
	dt_fenz_std = []

	dt_fenz_gpu = []
	dt_fenz_gpu_std = []
	
	for n in 1:30:100
		@info n
		push!(sizes, n^2)
		f = Bool.(rand([0, 1], n, n))
		
		# Python
		dt1 = @benchmark DistanceTransformsPy.transform($f, $Scipy())
		push!(dt_scipy, BenchmarkTools.mean(dt1).time)
		push!(dt_scipy_std, BenchmarkTools.std(dt1).time)
		
		# Felzenszwalb
		nthreads = Threads.nthreads()
		dt2 = @benchmark DistanceTransforms.transform!(boolean_indicator($f), $Felzenszwalb(), $nthreads)
		push!(dt_fenz, BenchmarkTools.mean(dt2).time)
		push!(dt_fenz_std, BenchmarkTools.std(dt2).time)

		# Felzenszwalb GPU
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			dt3 = @benchmark DistanceTransforms.transform(boolean_indicator($f_cuda), $Felzenszwalb())
			push!(dt_fenz_gpu, BenchmarkTools.mean(dt3).time)
			push!(dt_fenz_gpu_std, BenchmarkTools.std(dt3).time)
		end
	end
end

# ╔═╡ cf0657f5-1a5e-4e72-b6c7-b2bbe5c8e7b5
dt_py

# ╔═╡ 00bd4240-ad66-4aa2-914b-cdf64fe4d639
md"""
## 3D
"""

# ╔═╡ 37b3e0fc-80d1-427a-bd26-0a17f48db118
begin
	sizes_3D = []
	
	dt_scipy_3D = []
	dt_scipy_std_3D = []

	dt_fenz_3D = []
	dt_fenz_std_3D = []

	dt_fenz_gpu_3D = []
	dt_fenz_gpu_std_3D = []
	
	for n in 1:30:100
		@info n
		push!(sizes_3D, n^23)
		f = Bool.(rand([0, 1], n, n, n))
		
		# Python
		dt1 = @benchmark DistanceTransformsPy.transform($f, $Scipy())
		push!(dt_scipy_3D, BenchmarkTools.mean(dt1).time)
		push!(dt_scipy_std_3D, BenchmarkTools.std(dt1).time)
		
		# Felzenszwalb
		nthreads = Threads.nthreads()
		dt2 = @benchmark DistanceTransforms.transform!($f, $Felzenszwalb(), $nthreads)
		push!(dt_fenz_3D, BenchmarkTools.mean(dt2).time)
		push!(dt_fenz_std_3D, BenchmarkTools.std(dt2).time)

		# Felzenszwalb GPU
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			dt3 = @benchmark DistanceTransforms.transform($f_cuda, $Felzenszwalb())
			push!(dt_fenz_gpu_3D, BenchmarkTools.mean(dt3).time)
			push!(dt_fenz_gpu_std_3D, BenchmarkTools.std(dt3).time)
		end
	end
end

# ╔═╡ 57dc4b37-7165-4f97-89b6-c2238888db5a
let
	f = CairoMakie.Figure(
		title="Distance Transforms"
	)
	ax = Axis(
		f[1, 1],
		title="2D"
	)

	lines!(Float32.(dt_scipy), label="Scipy")
	lines!(Float32.(dt_fenz), label="Felzenszwalb")

	ax = Axis(
		f[2, 1],
		title="3D"
	)
	lines!(Float32.(dt_scipy_3D), label="Scipy")
	lines!(Float32.(dt_fenz_3D), label="Felzenszwalb")

	f[1:2, 2] = Legend(f, ax, "DT Algorithms", framevisible = false)
	
	f
end

# ╔═╡ 851f761f-a7d3-4cc3-978a-a0009e5c1ec4
dt_py_3D

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
	sizes_hd = []
	
	hd_scipy = []
	hd_scipy_std = []

	hd_fenz = []
	hd_fenz_std = []

	hd_fenz_gpu = []
	hd_fenz_gpu_std = []
	
	for n in 1:30:100
		@info n
		nthreads = Threads.nthreads()
		push!(sizes_hd, n^2)
		f = Bool.(rand([0, 1], n, n))
		f_dtm = DistanceTransforms.transform!(f, Felzenszwalb(), nthreads)
		
		# Scipy
		dt1 = @benchmark hausdorff($f, $f, DistanceTransformsPy.transform($f, $Scipy()), DistanceTransformsPy.transform($f, $Scipy()))
		# dt1 = @benchmark hausdorff($f, $f, $f_dtm, $f_dtm)
		push!(hd_scipy, BenchmarkTools.mean(dt1).time)
		push!(hd_scipy_std, BenchmarkTools.std(dt1).time)
		
		# Felzenszwalb
		dt2 = @benchmark hausdorff($f, $f, DistanceTransforms.transform!($f, $Felzenszwalb(), $nthreads), DistanceTransforms.transform!($f, $Felzenszwalb(), $nthreads))
		push!(hd_fenz, BenchmarkTools.mean(dt2).time)
		push!(hd_fenz_std, BenchmarkTools.std(dt2).time)

		# Felzenszwalb GPU
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			dt3 = @benchmark hausdorff($f, $f, DistanceTransforms.transform($f_cuda, $Felzenszwalb()), DistanceTransforms.transform($f_cuda, $Felzenszwalb()))
			push!(hd_fenz_gpu, BenchmarkTools.mean(dt3).time)
			push!(hd_fenz_gpu_std, BenchmarkTools.std(dt3).time)
		end
	end
end

# ╔═╡ 409f3581-2283-41a9-b8cc-3bb6d2094bcf
dice_baseline, dice_hd_scipy

# ╔═╡ 1d51b3af-6819-4996-8eea-8285b5c7a1b3
md"""
## 3D
"""

# ╔═╡ 8327f001-b7e6-4f9c-8bef-21c4e992b7be
begin
	sizes_hd_3D = []
	
	hd_scipy_3D = []
	hd_scipy_std_3D = []

	hd_fenz_3D = []
	hd_fenz_std_3D = []

	hd_fenz_gpu_3D = []
	hd_fenz_gpu_std_3D = []
	
	for n in 1:30:100
		@info n
		nthreads = Threads.nthreads()
		push!(sizes_hd_3D, n^3)
		f = Bool.(rand([0, 1], n, n, n))
		f_dtm = DistanceTransforms.transform!(f, Felzenszwalb(), nthreads)
		
		# Scipy
		dt1 = @benchmark hausdorff($f, $f, DistanceTransformsPy.transform($f, $Scipy()), DistanceTransformsPy.transform($f, $Scipy()))
		# dt1 = @benchmark hausdorff($f, $f, $f_dtm, $f_dtm)
		push!(hd_scipy_3D, BenchmarkTools.mean(dt1).time)
		push!(hd_scipy_std_3D, BenchmarkTools.std(dt1).time)
		
		# Felzenszwalb
		dt2 = @benchmark hausdorff($f, $f, DistanceTransforms.transform!($f, $Felzenszwalb(), $nthreads), DistanceTransforms.transform!($f, $Felzenszwalb(), $nthreads))
		push!(hd_fenz_3D, BenchmarkTools.mean(dt2).time)
		push!(hd_fenz_std_3D, BenchmarkTools.std(dt2).time)

		# Felzenszwalb GPU
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			dt3 = @benchmark hausdorff($f_cuda, $f_cuda, DistanceTransforms.transform($f_cuda, $Felzenszwalb()), DistanceTransforms.transform($f_cuda, $Felzenszwalb()))
			push!(hd_fenz_gpu_3D, BenchmarkTools.mean(dt3).time)
			push!(hd_fenz_gpu_std_3D, BenchmarkTools.std(dt3).time)
		end
	end
end

# ╔═╡ 85014ff8-05e4-44bb-a8f6-e2e8cf92f81f
let
	f = CairoMakie.Figure(
		title="Loss Functions"
	)
	ax = Axis(
		f[1, 1],
		title="2D"
	)
	lines!(Float32.(hd_scipy), label="Scipy")
	lines!(Float32.(hd_fenz), label="Felzenszwalb")
	

	ax = Axis(
		f[2, 1],
		title="3D"
	)
	lines!(Float32.(hd_scipy_3D), label="Scipy")
	lines!(Float32.(hd_fenz_3D), label="Felzenszwalb")

	f[1:2, 2] = Legend(f, ax, "DT Algorithms", framevisible = false)
	
	f
end

# ╔═╡ Cell order:
# ╠═bbb7b211-fa87-489d-87dc-1890cd848812
# ╠═8cbf0cfd-6b41-4a2c-a4af-a1f43c7afe3f
# ╠═a0bcf2cb-ceb7-400e-b2b8-56ed98fc705a
# ╠═cbf06207-8ad8-42fa-9758-602bcfe7e4aa
# ╟─71da875a-8420-4f38-8ddd-eb68c110e6fd
# ╟─0e2eca71-ab2d-45d8-b4fe-963d9bf2f3b9
# ╠═f3ec467e-ca19-4cd3-917b-2d11344e8785
# ╠═cf0657f5-1a5e-4e72-b6c7-b2bbe5c8e7b5
# ╟─00bd4240-ad66-4aa2-914b-cdf64fe4d639
# ╠═37b3e0fc-80d1-427a-bd26-0a17f48db118
# ╟─57dc4b37-7165-4f97-89b6-c2238888db5a
# ╠═851f761f-a7d3-4cc3-978a-a0009e5c1ec4
# ╟─11431e55-0f01-441a-8855-24952f8bbfcd
# ╟─e908389d-671f-4cea-a717-218cb6e00bd7
# ╠═9e11696a-9961-4ad1-a9a7-d70fdbd5423c
# ╠═409f3581-2283-41a9-b8cc-3bb6d2094bcf
# ╟─1d51b3af-6819-4996-8eea-8285b5c7a1b3
# ╠═8327f001-b7e6-4f9c-8bef-21c4e992b7be
# ╠═85014ff8-05e4-44bb-a8f6-e2e8cf92f81f