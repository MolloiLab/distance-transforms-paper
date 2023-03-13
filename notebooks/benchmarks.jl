### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 10122c53-0904-464c-bb47-e91bc5882cfb
# ╠═╡ show_logs = false
begin
	using DrWatson; @quickactivate "hd-loss"
	using PlutoUI, BenchmarkTools, CairoMakie, DataFrames, CSV, DistanceTransforms, CUDA
	using DistanceTransformsPy
end

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
	
	dt_py = []
	dt_py_std = []

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
		push!(dt_py, BenchmarkTools.mean(dt1).time)
		push!(dt_py_std, BenchmarkTools.std(dt1).time)
		
		# Felzenszwalb
		nthreads = Threads.nthreads()
		dt2 = @benchmark DistanceTransforms.transform!($f, $Felzenszwalb(), $nthreads)
		push!(dt_fenz, BenchmarkTools.mean(dt2).time)
		push!(dt_fenz_std, BenchmarkTools.std(dt2).time)

		# Felzenszwalb GPU
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			dt3 = @benchmark DistanceTransformsPy.transform($f_cuda, $Felzenszwalb())
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
	
	dt_py_3D = []
	dt_py_std_3D = []

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
		push!(dt_py_3D, BenchmarkTools.mean(dt1).time)
		push!(dt_py_std_3D, BenchmarkTools.std(dt1).time)
		
		# Felzenszwalb
		nthreads = Threads.nthreads()
		dt2 = @benchmark DistanceTransforms.transform!($f, $Felzenszwalb(), $nthreads)
		push!(dt_fenz_3D, BenchmarkTools.mean(dt2).time)
		push!(dt_fenz_std_3D, BenchmarkTools.std(dt2).time)

		# Felzenszwalb GPU
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			dt3 = @benchmark DistanceTransformsPy.transform($f_cuda, $Felzenszwalb())
			push!(dt_fenz_gpu_3D, BenchmarkTools.mean(dt3).time)
			push!(dt_fenz_gpu_std_3D, BenchmarkTools.std(dt3).time)
		end
	end
end

# ╔═╡ 851f761f-a7d3-4cc3-978a-a0009e5c1ec4
dt_py_3D

# ╔═╡ 11431e55-0f01-441a-8855-24952f8bbfcd
md"""
# Loss Functions
"""

# ╔═╡ Cell order:
# ╠═10122c53-0904-464c-bb47-e91bc5882cfb
# ╠═cbf06207-8ad8-42fa-9758-602bcfe7e4aa
# ╟─71da875a-8420-4f38-8ddd-eb68c110e6fd
# ╟─0e2eca71-ab2d-45d8-b4fe-963d9bf2f3b9
# ╠═f3ec467e-ca19-4cd3-917b-2d11344e8785
# ╠═cf0657f5-1a5e-4e72-b6c7-b2bbe5c8e7b5
# ╟─00bd4240-ad66-4aa2-914b-cdf64fe4d639
# ╠═37b3e0fc-80d1-427a-bd26-0a17f48db118
# ╠═851f761f-a7d3-4cc3-978a-a0009e5c1ec4
# ╟─11431e55-0f01-441a-8855-24952f8bbfcd
