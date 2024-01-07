### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ ffe6663f-cb17-4e92-a175-82eca15564cf
begin
	using Pkg
# 	Pkg.activate(".")
# 	Pkg.instantiate()
# 	# Pkg.add("DrWatson")
# 	# Pkg.add("PlutoUI")
# 	# Pkg.add("BenchmarkTools")
# 	# Pkg.add("CairoMakie")
# 	# Pkg.add("DataFrames")
# 	# Pkg.add("CSV")
	Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl", rev="master")
# 	# Pkg.add("CUDA")
# 	# Pkg.add("Losers")
# 	# Pkg.add("CSVFiles")
# 	# Pkg.add(url="https://github.com/Dale-Black/DistanceTransformsPy.jl", rev="main")
end

# ╔═╡ bbb7b211-fa87-489d-87dc-1890cd848812
using DrWatson

# ╔═╡ 8cbf0cfd-6b41-4a2c-a4af-a1f43c7afe3f
# ╠═╡ show_logs = false
@quickactivate "hd-loss"

# ╔═╡ a0bcf2cb-ceb7-400e-b2b8-56ed98fc705a
# ╠═╡ show_logs = false
using PlutoUI, BenchmarkTools, CairoMakie, DataFrames, CSV, DistanceTransforms, CUDA, Losers, DistanceTransformsPy, Statistics

# ╔═╡ cbf06207-8ad8-42fa-9758-602bcfe7e4aa
TableOfContents()

# ╔═╡ e53d73e1-126f-42bd-99df-87b59903eb70
begin
	range_size_2d = range(4, 4096, 20)
	range_size_3d = range(4, 256, 20)
	# range_size_2d = range(4, 512, 5)
	# range_size_3d = range(4, 64, 5)
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
	sizes = []
	
	dt_scipy = []
	dt_scipy_std = []

	dt_fenz = []
	dt_fenz_std = []

	dt_fenz_gpu = []
	dt_fenz_gpu_std = []
	
	for _n in range_size_2d
		n = round(Int, _n)
		@info n
		push!(sizes, n^2)
		f = Float32.(rand([0, 1], n, n))
		
		# Python
		dt1 = @benchmark DistanceTransformsPy.transform($f, $Scipy())
		push!(dt_scipy, BenchmarkTools.mean(dt1).time)
		push!(dt_scipy_std, BenchmarkTools.std(dt1).time)
		
		# Felzenszwalb
		nthreads = Threads.nthreads()
		dt2 = @benchmark DistanceTransforms.transform(boolean_indicator($f), $Felzenszwalb(), $nthreads)
		push!(dt_fenz, BenchmarkTools.mean(dt2).time)
		push!(dt_fenz_std, BenchmarkTools.std(dt2).time)

		# Felzenszwalb GPU
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			# 1st run to compile 
			temp = DistanceTransforms.transform(boolean_indicator(f_cuda), Felzenszwalb())
			CUDA.unsafe_free!(temp)
			CUDA.reclaim()
			
			# Then custom benchmark
			dt3 = []
			for trial = 1:1000
			    # run 
				curr_trial = @timed DistanceTransforms.transform(boolean_indicator(f_cuda), Felzenszwalb())
			    # record time
			    push!(dt3, (curr_trial.time - curr_trial.gctime)*10^9)
			    # clear mem
			    CUDA.unsafe_free!(curr_trial.value)
			    # CUDA.reclaim()
			end
			curr_trial = nothing
			GC.gc(true)
			CUDA.reclaim()

			# Record data
			push!(dt_fenz_gpu, mean(dt3))
			push!(dt_fenz_gpu_std, Statistics.std(dt3))
		end
	end
end

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
	
	for _n in range_size_3d
		n = round(Int, _n)
		@info n
		push!(sizes_3D, n^3)
		f = Float32.(rand([0, 1], n, n, n))
		
		# Python
		dt1 = @benchmark DistanceTransformsPy.transform($f, $Scipy())
		push!(dt_scipy_3D, BenchmarkTools.mean(dt1).time)
		push!(dt_scipy_std_3D, BenchmarkTools.std(dt1).time)
		
		# Felzenszwalb
		nthreads = Threads.nthreads()
		dt2 = @benchmark DistanceTransforms.transform(boolean_indicator($f), $Felzenszwalb(), $nthreads)
		push!(dt_fenz_3D, BenchmarkTools.mean(dt2).time)
		push!(dt_fenz_std_3D, BenchmarkTools.std(dt2).time)

		# Felzenszwalb GPU
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			# 1st run to compile 
			temp = DistanceTransforms.transform(boolean_indicator(f_cuda), Felzenszwalb())
			CUDA.unsafe_free!(temp)
			CUDA.reclaim()
			
			# Then custom benchmark
			dt3 = []
			for trial = 1:1000
			    # run 
				curr_trial = @timed DistanceTransforms.transform(boolean_indicator(f_cuda), Felzenszwalb())
			    # record time
			    push!(dt3, (curr_trial.time - curr_trial.gctime)*10^9)
			    # clear mem
			    CUDA.unsafe_free!(curr_trial.value)
			    # CUDA.reclaim()
			end
			curr_trial = nothing
			GC.gc(true)
			CUDA.reclaim()

			# Record data
			push!(dt_fenz_gpu_3D, mean(dt3))
			push!(dt_fenz_gpu_std_3D, Statistics.std(dt3))
		end
	end
end

# ╔═╡ 57dc4b37-7165-4f97-89b6-c2238888db5a
let
	f = CairoMakie.Figure()
	ax = Axis(
		f[1, 1],
		title="2D"
	)

	lines!(Float32.(dt_scipy), label="Scipy")
	lines!(Float32.(dt_fenz), label="Felzenszwalb (CPU)")
	lines!(Float32.(dt_fenz_gpu), label="Felzenszwalb (GPU)")

	ax = Axis(
		f[2, 1],
		title="3D"
	)
	lines!(Float32.(dt_scipy_3D), label="Scipy")
	lines!(Float32.(dt_fenz_3D), label="Felzenszwalb (CPU)")
	lines!(Float32.(dt_fenz_gpu_3D), label="Felzenszwalb (GPU)")

	f[1:2, 2] = Legend(f, ax, "DT Algorithms", framevisible = false)

	Label(f[0, 1:2]; text="Distance Transforms", textsize=30)
	f
end

# ╔═╡ 709116ce-64d2-4407-aad5-22c256cf9de7
begin
	global df_dt
	if has_cuda_gpu()
		df_dt = DataFrame(
			sizes = sizes,
			dt_scipy = dt_scipy,
			dt_scipy_std = dt_scipy_std,
			dt_fenz = dt_fenz,
			dt_fenz_std = dt_fenz_std,
			dt_fenz_gpu = dt_fenz_gpu,
			dt_fenz_gpu_std = dt_fenz_gpu_std,
		
			sizes_3D = sizes_3D,
			dt_scipy_3D = dt_scipy_3D,
			dt_scipy_std_3D = dt_scipy_std_3D,
			dt_fenz_3D = dt_fenz_3D,
			dt_fenz_std_3D = dt_fenz_std_3D,
			dt_fenz_gpu_3D = dt_fenz_gpu_3D,
			dt_fenz_gpu_std_3D = dt_fenz_gpu_std_3D,
		)
	else
		df_dt = DataFrame(
			sizes = sizes,
			dt_scipy = dt_scipy,
			dt_scipy_std = dt_scipy_std,
			dt_fenz = dt_fenz,
			dt_fenz_std = dt_fenz_std,
		
			sizes_3D = sizes_3D,
			dt_scipy_3D = dt_scipy_3D,
			dt_scipy_std_3D = dt_scipy_std_3D,
			dt_fenz_3D = dt_fenz_3D,
			dt_fenz_std_3D = dt_fenz_std_3D,
		)
	end
end

# ╔═╡ df1e62e3-4ff7-4ca5-bce9-ce4a29c34ec9
save(datadir("results", "dts.csv"), df_dt)

# ╔═╡ 11431e55-0f01-441a-8855-24952f8bbfcd
md"""
# Loss Functions
"""

# ╔═╡ e908389d-671f-4cea-a717-218cb6e00bd7
md"""
## 2D
"""

# ╔═╡ 070822e7-bf8e-4622-99a2-8ff96527568b
function my_hausdorff(ŷ, y, ŷ_dtm, y_dtm)
    return mean((ŷ .- y) .^ 2 .* (ŷ_dtm .+ y_dtm))
end

# ╔═╡ 9e11696a-9961-4ad1-a9a7-d70fdbd5423c
begin
	sizes_hd = []
	
	hd_scipy = []
	hd_scipy_std = []

	hd_fenz = []
	hd_fenz_std = []

	hd_fenz_gpu = []
	hd_fenz_gpu_std = []
	
	for _n in range_size_2d
		n = round(Int, _n)
		@info n
		nthreads = Threads.nthreads()
		push!(sizes_hd, n^2)
		f = Float32.(rand([0, 1], n, n))
		f2 = Float32.(rand([0, 1], n, n))
		
		# Scipy
		dt1 = @benchmark my_hausdorff($f, $f2, DistanceTransformsPy.transform($f, $Scipy()), DistanceTransformsPy.transform($f2, $Scipy()))
	
		push!(hd_scipy, BenchmarkTools.mean(dt1).time)
		push!(hd_scipy_std, BenchmarkTools.std(dt1).time)
		
		# Felzenszwalb
		dt2 = @benchmark my_hausdorff($f, $f2, DistanceTransforms.transform(boolean_indicator($f), $Felzenszwalb(), $nthreads), DistanceTransforms.transform(boolean_indicator($f2), $Felzenszwalb(), $nthreads))
		
		push!(hd_fenz, BenchmarkTools.mean(dt2).time)
		push!(hd_fenz_std, BenchmarkTools.std(dt2).time)

		# Felzenszwalb GPU
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			f2_cuda = CuArray(f2)
			# 1st run to compile 
			f_dtm = DistanceTransforms.transform(boolean_indicator(f_cuda), Felzenszwalb())
			f2_dtm = DistanceTransforms.transform(boolean_indicator(f2_cuda), Felzenszwalb())
			temp = my_hausdorff(f_cuda, f2_cuda, f_dtm, f2_dtm)
			f_dtm, f2_dtm, temp = nothing, nothing, nothing
			GC.gc(true)
			CUDA.reclaim()
			
			# Then custom benchmark
			dt3 = []
			for trial = 1:1000
			    # run 
				curr_trial = @timed begin
					f_dtm = DistanceTransforms.transform(boolean_indicator(f_cuda), Felzenszwalb())
					f2_dtm = DistanceTransforms.transform(boolean_indicator(f2_cuda), Felzenszwalb())
					temp = my_hausdorff(f_cuda, f2_cuda, f_dtm, f2_dtm)
				end
				
			    # clear mem
			    CUDA.unsafe_free!(f_dtm)
			    CUDA.unsafe_free!(f2_dtm)
			    temp = nothing
				
			    # record time
			    push!(dt3, (curr_trial.time - curr_trial.gctime)*10^9)
			end
			GC.gc(true)
			CUDA.reclaim()

			# Record data
			push!(hd_fenz_gpu, mean(dt3))
			push!(hd_fenz_gpu_std, Statistics.std(dt3))
		end
	end
end

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
	
	for _n in range_size_3d
		n = round(Int, _n)
		@info n
		nthreads = Threads.nthreads()
		push!(sizes_hd_3D, n^3)
		f = Float32.(rand([0, 1], n, n, n))
		f2 = Float32.(rand([0, 1], n, n, n))
		
		# Scipy
		dt1 = @benchmark my_hausdorff($f, $f2, DistanceTransformsPy.transform($f, $Scipy()), DistanceTransformsPy.transform($f2, $Scipy()))
		
		push!(hd_scipy_3D, BenchmarkTools.mean(dt1).time)
		push!(hd_scipy_std_3D, BenchmarkTools.std(dt1).time)
		
		# Felzenszwalb
		dt2 = @benchmark my_hausdorff($f, $f2, DistanceTransforms.transform(boolean_indicator($f), $Felzenszwalb(), $nthreads), DistanceTransforms.transform(boolean_indicator($f2), $Felzenszwalb(), $nthreads))
		
		push!(hd_fenz_3D, BenchmarkTools.mean(dt2).time)
		push!(hd_fenz_std_3D, BenchmarkTools.std(dt2).time)

		# Felzenszwalb GPU
		if has_cuda_gpu()
			f_cuda = CuArray(f)
			f2_cuda = CuArray(f2)
			# 1st run to compile 
			f_dtm = DistanceTransforms.transform(boolean_indicator(f_cuda), Felzenszwalb())
			f2_dtm = DistanceTransforms.transform(boolean_indicator(f2_cuda), Felzenszwalb())
			temp = my_hausdorff(f_cuda, f2_cuda, f_dtm, f2_dtm)
			f_dtm, f2_dtm, temp = nothing, nothing, nothing
			GC.gc(true)
			CUDA.reclaim()
			
			# Then custom benchmark
			dt3 = []
			for trial = 1:1000
			    # run 
				curr_trial = @timed begin
					f_dtm = DistanceTransforms.transform(boolean_indicator(f_cuda), Felzenszwalb())
					f2_dtm = DistanceTransforms.transform(boolean_indicator(f2_cuda), Felzenszwalb())
					temp = my_hausdorff(f_cuda, f2_cuda, f_dtm, f2_dtm)
				end
				
			    # clear mem
			    CUDA.unsafe_free!(f_dtm)
			    CUDA.unsafe_free!(f2_dtm)
			    temp = nothing
				
			    # record time
			    push!(dt3, (curr_trial.time - curr_trial.gctime)*10^9)
			end
			GC.gc(true)
			CUDA.reclaim()

			# Record data
			push!(hd_fenz_gpu_3D, mean(dt3))
			push!(hd_fenz_gpu_std_3D, Statistics.std(dt3))
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
	lines!(Float32.(hd_scipy), label="Scipy")
	lines!(Float32.(hd_fenz), label="Felzenszwalb (CPU)")
	lines!(Float32.(hd_fenz_gpu), label="Felzenszwalb (GPU)")
	

	ax = Axis(
		f[2, 1],
		title="3D"
	)
	lines!(Float32.(hd_scipy_3D), label="Scipy")
	lines!(Float32.(hd_fenz_3D), label="Felzenszwalb (CPU)")
	lines!(Float32.(hd_fenz_gpu_3D), label="Felzenszwalb (GPU)")

	f[1:2, 2] = Legend(f, ax, "HD Algorithms", framevisible = false)

	Label(f[0, 1:2]; text="Loss Functions", textsize=30)
	
	f
end

# ╔═╡ 036fcfad-ceba-4914-b2f8-744f22fb9969
begin
	global df_loss
	if has_cuda_gpu()
		df_loss = DataFrame(
			sizes_hd = sizes_hd,
			hd_scipy = hd_scipy,
			hd_scipy_std = hd_scipy_std,
			hd_fenz = hd_fenz,
			hd_fenz_std = hd_fenz_std,
			hd_fenz_gpu = hd_fenz_gpu,
			hd_fenz_gpu_std = hd_fenz_gpu_std,
		
			sizes_hd_3D = sizes_hd_3D,
			hd_scipy_3D = hd_scipy_3D,
			hd_scipy_std_3D = hd_scipy_std_3D,
			hd_fenz_3D = hd_fenz_3D,
			hd_fenz_std_3D = hd_fenz_std_3D,
			hd_fenz_gpu_3D = hd_fenz_gpu_3D,
			hd_fenz_gpu_std_3D = hd_fenz_gpu_std_3D,
		)
	else
		df_loss = DataFrame(
			sizes_hd = sizes_hd,
			hd_scipy = hd_scipy,
			hd_scipy_std = hd_scipy_std,
			hd_fenz = hd_fenz,
			hd_fenz_std = hd_fenz_std,
		
			sizes_hd_3D = sizes_hd_3D,
			hd_scipy_3D = hd_scipy_3D,
			hd_scipy_std_3D = hd_scipy_std_3D,
			hd_fenz_3D = hd_fenz_3D,
			hd_fenz_std_3D = hd_fenz_std_3D,
		)
	end
end

# ╔═╡ 9368f91e-97d8-4452-9138-2c1e5965cd76
save(datadir("results", "losses.csv"), df_loss)

# ╔═╡ Cell order:
# ╠═ffe6663f-cb17-4e92-a175-82eca15564cf
# ╠═bbb7b211-fa87-489d-87dc-1890cd848812
# ╠═8cbf0cfd-6b41-4a2c-a4af-a1f43c7afe3f
# ╠═a0bcf2cb-ceb7-400e-b2b8-56ed98fc705a
# ╠═cbf06207-8ad8-42fa-9758-602bcfe7e4aa
# ╠═e53d73e1-126f-42bd-99df-87b59903eb70
# ╟─71da875a-8420-4f38-8ddd-eb68c110e6fd
# ╟─0e2eca71-ab2d-45d8-b4fe-963d9bf2f3b9
# ╠═f3ec467e-ca19-4cd3-917b-2d11344e8785
# ╟─00bd4240-ad66-4aa2-914b-cdf64fe4d639
# ╠═37b3e0fc-80d1-427a-bd26-0a17f48db118
# ╟─57dc4b37-7165-4f97-89b6-c2238888db5a
# ╟─709116ce-64d2-4407-aad5-22c256cf9de7
# ╠═df1e62e3-4ff7-4ca5-bce9-ce4a29c34ec9
# ╟─11431e55-0f01-441a-8855-24952f8bbfcd
# ╟─e908389d-671f-4cea-a717-218cb6e00bd7
# ╠═070822e7-bf8e-4622-99a2-8ff96527568b
# ╠═9e11696a-9961-4ad1-a9a7-d70fdbd5423c
# ╟─1d51b3af-6819-4996-8eea-8285b5c7a1b3
# ╠═8327f001-b7e6-4f9c-8bef-21c4e992b7be
# ╟─85014ff8-05e4-44bb-a8f6-e2e8cf92f81f
# ╠═036fcfad-ceba-4914-b2f8-744f22fb9969
# ╠═9368f91e-97d8-4452-9138-2c1e5965cd76
