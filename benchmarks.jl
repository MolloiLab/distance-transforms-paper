### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 8401528c-5484-42f7-8f92-94dcca409526
# ╠═╡ show_logs = false
using Pkg; Pkg.activate(".."), Pkg.instantiate();

# ╔═╡ 57c6afc9-8a71-4239-8187-27c1081f31e5
using CUDA

# ╔═╡ cc254c52-27ed-4b64-b7c7-a723f35038fb
using Metal

# ╔═╡ 49b3c214-181b-4689-b843-d545684ed205
# ╠═╡ show_logs = false
using AMDGPU

# ╔═╡ d0f96ba1-b81d-485c-872e-f1054dbecb4b
# ╠═╡ show_logs = false
using oneAPI

# ╔═╡ bbb7b211-fa87-489d-87dc-1890cd848812
# ╠═╡ show_logs = false
using DrWatson: datadir

# ╔═╡ bbcee247-a25f-4bae-b33d-76498e71310d
using PlutoUI: TableOfContents

# ╔═╡ 5687eefa-b54b-4d0e-b317-d92a09f4927b
using BenchmarkTools: minimum, std, @benchmark

# ╔═╡ 82b961f6-8718-4216-8b96-5f46ca0b1730
using CairoMakie: Figure, Axis, barplot!, Label, PolyElement, Legend, @L_str

# ╔═╡ 6de6ce85-5dcb-47f7-bff3-7d990cd2dddd
using CairoMakie # For the use of `Makie.wong_colors`

# ╔═╡ 64fd0e96-e8c3-478d-b9e0-dbf6aa15387c
using DataFrames: DataFrame

# ╔═╡ 970bba1d-9dde-4ade-8b38-246a925aaa3b
using CSV: write

# ╔═╡ 545b7f75-29b1-438e-883a-e2ded8053eed
using ImageMorphology: distance_transform, feature_transform

# ╔═╡ 1e38338d-eaa8-4252-9a05-698abe4d7756
using DistanceTransforms: boolean_indicator, transform

# ╔═╡ e966157c-0688-4d64-85ac-c0e724fe0032
# CUDA.set_runtime_version!(v"11.8")

# ╔═╡ a9fcdbcc-96a8-4cb4-89be-6e176d1e9251
CUDA.functional()

# ╔═╡ aa226ffd-8077-4957-8200-792608c51676
Metal.functional()

# ╔═╡ a4ebcc97-81e2-4129-9b8a-40db60e6cbd3
AMDGPU.functional()

# ╔═╡ 0aa1d88e-ab1a-4316-870e-65027745a2e2
oneAPI.functional()

# ╔═╡ cbf06207-8ad8-42fa-9758-602bcfe7e4aa
TableOfContents()

# ╔═╡ 1cc970c6-0bb2-41cc-a3ea-f3ba452a2395
s = 10 # number of samples

# ╔═╡ 4c364b78-b72b-4aa8-999d-71f78b90be39
e = 1 # number of evals

# ╔═╡ 71da875a-8420-4f38-8ddd-eb68c110e6fd
md"""
# Distance Transforms
"""

# ╔═╡ 0e2eca71-ab2d-45d8-b4fe-963d9bf2f3b9
md"""
## 2D
"""

# ╔═╡ 7ca89176-4723-49f0-bd07-b6141aa6d8a4
range_size_2D = [2^i for i in 3:12]

# ╔═╡ ab3a72b3-4446-479e-949d-20b24234503f
range_names_2D = [L"2^3", L"2^4", L"2^5", L"2^6", L"2^7", L"2^8", L"2^9", L"2^{10}", L"2^{11}", L"2^{12}"]

# ╔═╡ f3ec467e-ca19-4cd3-917b-2d11344e8785
begin
    sizes = Float64[]

    dt_maurer = Float64[]
    dt_maurer_std = Float64[]

    dt_fenz = Float64[]
    dt_fenz_std = Float64[]

    dt_fenz_multi = Float64[]
    dt_fenz_multi_std = Float64[]

    dt_proposed_cuda = Float64[]
    dt_proposed_cuda_std = Float64[]

    dt_proposed_metal = Float64[]
    dt_proposed_metal_std = Float64[]

    dt_proposed_amdgpu = Float64[]
    dt_proposed_amdgpu_std = Float64[]
    
	for n in range_size_2D
		@info n
		push!(sizes, n^2)
		f = Float32.(rand([0, 1], n, n))
	
		# Maurer (ImageMorphology.jl)
		bool_f = Bool.(f)
		dt = @benchmark(distance_transform($feature_transform($bool_f)), samples=s, evals=e)
		push!(dt_maurer, minimum(dt).time) # ns
		push!(dt_maurer_std, std(dt).time)
		
		# Felzenszwalb (DistanceTransforms.jl)
		dt = @benchmark(transform($boolean_indicator($f); threaded = false), samples=s, evals=e)
		push!(dt_fenz, minimum(dt).time)
		push!(dt_fenz_std, std(dt).time)
	
		# Felzenszwalb Multi-threaded (DistanceTransforms.jl)
		dt = @benchmark(transform($boolean_indicator($f)), samples=s, evals=e)
		push!(dt_fenz_multi, minimum(dt).time)
		push!(dt_fenz_multi_std, std(dt).time)
	
		# Proposed-CUDA (DistanceTransforms.jl)
		if CUDA.functional()
			f_cuda = CuArray(f)
			dt = @benchmark(transform($boolean_indicator($f_cuda)), samples=s, evals=e)
			push!(dt_proposed_cuda, minimum(dt).time)
			push!(dt_proposed_cuda_std, std(dt).time)
		end
	
		# Proposed-Metal (DistanceTransforms.jl)
		if Metal.functional()
			f_metal = MtlArray(f)
			dt = @benchmark(transform($boolean_indicator($f_metal)), samples=s, evals=e)
			push!(dt_proposed_metal, minimum(dt).time)
			push!(dt_proposed_metal_std, std(dt).time)
		end
	
		# Proposed-AMDGPU (DistanceTransforms.jl)
		if AMDGPU.functional()
			f_amdgpu = ROCArray(f)
			dt = @benchmark(transform($boolean_indicator($f_amdgpu)), samples=s, evals=e)
			push!(dt_proposed_amdgpu, minimum(dt).time)
			push!(dt_proposed_amdgpu_std, std(dt).time)
		end
	end
end

# ╔═╡ d1103e59-7c4f-432f-bca2-4e1bfce52a64
function create_barplot(
	title, dt_names, sizes, dt_maurer, dt_fenz, dt_fenz_multi;
	dt_proposed_cuda = [],
	dt_proposed_metal = [],
	dt_proposed_oneapi = [],
	dt_proposed_amdgpu = [],
	x_names = sizes
	)
    dt_heights = zeros(length(dt_names) * length(sizes))
    
    heights = hcat(
        dt_maurer,
        dt_fenz,
        dt_fenz_multi,
        isempty(dt_proposed_cuda) ? zeros(length(dt_maurer)) : dt_proposed_cuda,
        isempty(dt_proposed_metal) ? zeros(length(dt_maurer)) : dt_proposed_metal,
        isempty(dt_proposed_oneapi) ? zeros(length(dt_maurer)) : dt_proposed_oneapi,
        isempty(dt_proposed_amdgpu) ? zeros(length(dt_maurer)) : dt_proposed_amdgpu
    )

    offset = 1
    for i in eachrow(heights)
        dt_heights[offset:(offset+length(i) - 1)] .= i
        offset += 7
    end

    cat = repeat(1:length(sizes), inner = length(dt_names))
    grp = repeat(1:length(dt_names), length(sizes))
    colors = Makie.wong_colors()

    f = Figure(size = (800, 600))
    ax = Axis(
        f[1, 1:2],
        ylabel = "Time (ns)",
        title = title,
		titlesize = 25,
        xticks = (1:length(sizes), x_names),
        yscale = log10
    )

    barplot!(
        cat, dt_heights;
        dodge = grp,
        color = colors[grp],
    )

    # X axis label
    Label(f[2, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

    # Legend
    labels = dt_names
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    title = "Distance Transforms"

    Legend(f[1, 3], elements, labels, title)
    return f
end

# ╔═╡ 910cf44a-7df1-4a8b-a5ce-8b4f9d301280
title_2D = "Performance Comparison \nof Distance Transforms (2D)"

# ╔═╡ 1d29ab0d-dfe1-4cc0-b81c-a59edeef8724
dt_names = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (Metal)", "Proposed (oneAPI)", "Proposed (AMDGPU)"]

# ╔═╡ 98b469f9-c8eb-4565-b5fa-5ae6b5a87f38
create_barplot(
	title_2D, dt_names, sizes, dt_maurer, dt_fenz, dt_fenz_multi;
	dt_proposed_cuda = dt_proposed_cuda,
	dt_proposed_metal = dt_proposed_metal,
	dt_proposed_amdgpu = dt_proposed_amdgpu,
	x_names = range_names_2D
)

# ╔═╡ 00bd4240-ad66-4aa2-914b-cdf64fe4d639
md"""
## 3D
"""

# ╔═╡ 53961fa7-a182-47df-92b5-f711bbb20ec9
range_size_3D = [2^i for i in 0:8]

# ╔═╡ 5bfc74f8-9f06-4e15-acfa-ae65f63d44b4
range_names_3D = [L"2^0", L"2^1", L"2^2", L"2^3", L"2^4", L"2^5", L"2^6", L"2^7", L"2^8"]

# ╔═╡ 37b3e0fc-80d1-427a-bd26-0a17f48db118
begin
    sizes_3D = Float64[]

    dt_maurer_3D = Float64[]
    dt_maurer_std_3D = Float64[]

    dt_fenz_3D = Float64[]
    dt_fenz_std_3D = Float64[]

    dt_fenz_multi_3D = Float64[]
    dt_fenz_multi_std_3D = Float64[]

    dt_proposed_cuda_3D = Float64[]
    dt_proposed_cuda_std_3D = Float64[]

    dt_proposed_metal_3D = Float64[]
    dt_proposed_metal_std_3D = Float64[]

    dt_proposed_amdgpu_3D = Float64[]
    dt_proposed_amdgpu_std_3D = Float64[]
    
	for n in range_size_3D
		@info n
		push!(sizes_3D, n^3)
		f = Float32.(rand([0, 1], n, n, n))
	
		# Maurer (ImageMorphology.jl)
		bool_f = Bool.(f)
		dt = @benchmark(distance_transform($feature_transform($bool_f)), samples=s, evals=e)
		push!(dt_maurer_3D, minimum(dt).time) # ns
		push!(dt_maurer_std_3D, std(dt).time)
		
		# Felzenszwalb (DistanceTransforms.jl)
		dt = @benchmark(transform($boolean_indicator($f); threaded = false), samples=s, evals=e)
		push!(dt_fenz_3D, minimum(dt).time)
		push!(dt_fenz_std_3D, std(dt).time)
	
		# Felzenszwalb Multi-threaded (DistanceTransforms.jl)
		dt = @benchmark(transform($boolean_indicator($f)), samples=s, evals=e)
		push!(dt_fenz_multi_3D, minimum(dt).time)
		push!(dt_fenz_multi_std_3D, std(dt).time)
	
		# Proposed-CUDA (DistanceTransforms.jl)
		if CUDA.functional()
			f_cuda = CuArray(f)
			dt = @benchmark(transform($boolean_indicator($f_cuda)), samples=s, evals=e)
			push!(dt_proposed_cuda_3D, minimum(dt).time)
			push!(dt_proposed_cuda_std_3D, std(dt).time)
		end
	
		# Proposed-Metal (DistanceTransforms.jl)
		if Metal.functional()
			f_metal = MtlArray(f)
			dt = @benchmark(transform($boolean_indicator($f_metal)), samples=s, evals=e)
			push!(dt_proposed_metal_3D, minimum(dt).time)
			push!(dt_proposed_metal_std_3D, std(dt).time)
		end
	
		# Proposed-AMDGPU (DistanceTransforms.jl)
		if AMDGPU.functional()
			f_amdgpu = ROCArray(f)
			dt = @benchmark(transform($boolean_indicator($f_amdgpu)), samples=s, evals=e)
			push!(dt_proposed_amdgpu_3D, minimum(dt).time)
			push!(dt_proposed_amdgpu_std_3D, std(dt).time)
		end
	end
end

# ╔═╡ 6a8e7b4d-3da6-4d94-9d10-0d955fed93d1
title_3D = "Performance Comparison \nof Distance Transforms (3D)"

# ╔═╡ c048cc96-cc86-4dba-b199-87bbf6873cea
create_barplot(
	title_3D, dt_names, sizes_3D, dt_maurer_3D, dt_fenz_3D, dt_fenz_multi_3D;
	dt_proposed_cuda = dt_proposed_cuda_3D,
	dt_proposed_metal = dt_proposed_metal_3D,
	dt_proposed_amdgpu = dt_proposed_amdgpu_3D,
	x_names = range_names_3D
)

# ╔═╡ bdfb296f-41a1-43ff-9c83-f39d88145286
md"""
## Dataframe
"""

# ╔═╡ 0d70978d-3326-451d-a713-91ef4df0fd96
os_info = sprint(versioninfo; context=:compact=>true)

# ╔═╡ 941127b9-0772-4a6b-abf8-0b19deee4460
if Metal.functional()
	gpu_info = sprint(Metal.versioninfo; context=:compact=>true)
elseif CUDA.functional()
	gpu_info = sprint(CUDA.versioninfo; context=:compact=>true)
elseif oneAPI.functional()
	gpu_info = sprint(oneAPI.versioninfo; context=:compact=>true)
elseif AMDGPU.functional()
	gpu_info = sprint(AMDGPU.versioninfo; context=:compact=>true)
end

# ╔═╡ da864008-caf0-424d-b039-3294b4c5d307
begin
	# Create the 2D DataFrame
	df_dt_2D = DataFrame(
	    os_info = os_info,
	    gpu_info = gpu_info,
	    sizes = sizes,
	    dt_maurer = dt_maurer,
	    dt_maurer_std = dt_maurer_std,
		    dt_fenz = dt_fenz,
	    dt_fenz_std = dt_fenz_std,
	    dt_fenz_multi = dt_fenz_multi,
	    dt_fenz_multi_std = dt_fenz_multi_std
	)

	if CUDA.functional()
	    df_dt_2D[!, :dt_proposed_cuda] = dt_proposed_cuda
	    df_dt_2D[!, :dt_proposed_cuda_std] = dt_proposed_cuda_std
	end

	if Metal.functional()
	    df_dt_2D[!, :dt_proposed_metal] = dt_proposed_metal
	    df_dt_2D[!, :dt_proposed_metal_std] = dt_proposed_metal_std
	end

	if oneAPI.functional()
	    df_dt_2D[!, :dt_proposed_oneapi] = dt_proposed_oneapi
	    df_dt_2D[!, :dt_proposed_oneapi_std] = dt_proposed_oneapi_std
	end

	if AMDGPU.functional()
	    df_dt_2D[!, :dt_proposed_amdgpu] = dt_proposed_amdgpu
	    df_dt_2D[!, :dt_proposed_amdgpu_std] = dt_proposed_amdgpu_std
	end
	df_dt_2D
end

# ╔═╡ c5825049-e570-4e8a-a666-80e2ed444540
begin
	# Create the 3D DataFrame
	df_dt_3D = DataFrame(
	    os_info = os_info,
	    gpu_info = gpu_info,
	    sizes_3D = sizes_3D,
	    dt_maurer_3D = dt_maurer_3D,
	    dt_maurer_std_3D = dt_maurer_std_3D,
	    dt_fenz_3D = dt_fenz_3D,
	    dt_fenz_std_3D = dt_fenz_std_3D,
	    dt_fenz_multi_3D = dt_fenz_multi_3D,
	    dt_fenz_multi_std_3D = dt_fenz_multi_std_3D
	)
	
	# Add proposed methods data to 2D DataFrame if available
	if CUDA.functional()
	    df_dt_3D[!, :dt_proposed_cuda_3D] = dt_proposed_cuda_3D
	    df_dt_3D[!, :dt_proposed_cuda_std_3D] = dt_proposed_cuda_std_3D
	end
	
	if Metal.functional()
	    df_dt_3D[!, :dt_proposed_metal_3D] = dt_proposed_metal_3D
	    df_dt_3D[!, :dt_proposed_metal_std_3D] = dt_proposed_metal_std_3D
	end
	
	if oneAPI.functional()
	    df_dt_3D[!, :dt_proposed_oneapi_3D] = dt_proposed_oneapi_3D
	    df_dt_3D[!, :dt_proposed_oneapi_std_3D] = dt_proposed_oneapi_std_3D
	end
	
	if AMDGPU.functional()
	    df_dt_3D[!, :dt_proposed_amdgpu_3D] = dt_proposed_amdgpu_3D
	    df_dt_3D[!, :dt_proposed_amdgpu_std_3D] = dt_proposed_amdgpu_std_3D
	end
	df_dt_3D
end

# ╔═╡ f8030286-0d3e-439b-ada6-ec92f653d6a7
begin
	# Determine the accelerator type
	accelerator = "CPU"
	if Metal.functional()
	    accelerator = "Metal"
	elseif CUDA.functional()
	    accelerator = "CUDA"
	elseif oneAPI.functional()
	    accelerator = "oneAPI"
	elseif AMDGPU.functional()
	    accelerator = "AMDGPU"
	end
end

# ╔═╡ 27bae356-b6cc-4839-8042-d56308261c81
md"""
## Save
"""

# ╔═╡ 84859dca-d05b-45e4-8bd8-7a994c5b3495
begin
	csv_filename_2D = "dt_2D_$(accelerator).csv"
	csv_filename_3D = "dt_3D_$(accelerator).csv"
end

# ╔═╡ f7f9334d-c152-4dd5-8986-ff463835fce3
begin
	write(datadir(csv_filename_2D), df_dt_2D)
	write(datadir(csv_filename_3D), df_dt_3D)
end

# ╔═╡ Cell order:
# ╠═8401528c-5484-42f7-8f92-94dcca409526
# ╠═e966157c-0688-4d64-85ac-c0e724fe0032
# ╠═57c6afc9-8a71-4239-8187-27c1081f31e5
# ╠═a9fcdbcc-96a8-4cb4-89be-6e176d1e9251
# ╠═cc254c52-27ed-4b64-b7c7-a723f35038fb
# ╠═aa226ffd-8077-4957-8200-792608c51676
# ╠═49b3c214-181b-4689-b843-d545684ed205
# ╠═a4ebcc97-81e2-4129-9b8a-40db60e6cbd3
# ╠═d0f96ba1-b81d-485c-872e-f1054dbecb4b
# ╠═0aa1d88e-ab1a-4316-870e-65027745a2e2
# ╠═bbb7b211-fa87-489d-87dc-1890cd848812
# ╠═bbcee247-a25f-4bae-b33d-76498e71310d
# ╠═5687eefa-b54b-4d0e-b317-d92a09f4927b
# ╠═82b961f6-8718-4216-8b96-5f46ca0b1730
# ╠═6de6ce85-5dcb-47f7-bff3-7d990cd2dddd
# ╠═64fd0e96-e8c3-478d-b9e0-dbf6aa15387c
# ╠═970bba1d-9dde-4ade-8b38-246a925aaa3b
# ╠═545b7f75-29b1-438e-883a-e2ded8053eed
# ╠═1e38338d-eaa8-4252-9a05-698abe4d7756
# ╠═cbf06207-8ad8-42fa-9758-602bcfe7e4aa
# ╠═1cc970c6-0bb2-41cc-a3ea-f3ba452a2395
# ╠═4c364b78-b72b-4aa8-999d-71f78b90be39
# ╟─71da875a-8420-4f38-8ddd-eb68c110e6fd
# ╟─0e2eca71-ab2d-45d8-b4fe-963d9bf2f3b9
# ╠═7ca89176-4723-49f0-bd07-b6141aa6d8a4
# ╠═ab3a72b3-4446-479e-949d-20b24234503f
# ╠═f3ec467e-ca19-4cd3-917b-2d11344e8785
# ╟─d1103e59-7c4f-432f-bca2-4e1bfce52a64
# ╠═910cf44a-7df1-4a8b-a5ce-8b4f9d301280
# ╠═1d29ab0d-dfe1-4cc0-b81c-a59edeef8724
# ╟─98b469f9-c8eb-4565-b5fa-5ae6b5a87f38
# ╟─00bd4240-ad66-4aa2-914b-cdf64fe4d639
# ╠═53961fa7-a182-47df-92b5-f711bbb20ec9
# ╠═5bfc74f8-9f06-4e15-acfa-ae65f63d44b4
# ╠═37b3e0fc-80d1-427a-bd26-0a17f48db118
# ╠═6a8e7b4d-3da6-4d94-9d10-0d955fed93d1
# ╠═c048cc96-cc86-4dba-b199-87bbf6873cea
# ╟─bdfb296f-41a1-43ff-9c83-f39d88145286
# ╠═0d70978d-3326-451d-a713-91ef4df0fd96
# ╠═941127b9-0772-4a6b-abf8-0b19deee4460
# ╠═da864008-caf0-424d-b039-3294b4c5d307
# ╠═c5825049-e570-4e8a-a666-80e2ed444540
# ╠═f8030286-0d3e-439b-ada6-ec92f653d6a7
# ╟─27bae356-b6cc-4839-8042-d56308261c81
# ╠═84859dca-d05b-45e4-8bd8-7a994c5b3495
# ╠═f7f9334d-c152-4dd5-8986-ff463835fce3
