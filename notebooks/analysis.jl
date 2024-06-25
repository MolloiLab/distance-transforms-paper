### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 2c729da6-40e6-47cd-a14d-c152b8789b17
# ╠═╡ show_logs = false
using Pkg; Pkg.activate(".."), Pkg.instantiate();

# ╔═╡ 5e56b525-cd2d-4e2d-9010-8210a08611be
using CUDA

# ╔═╡ ee82e108-12d0-483e-84f4-92a7a3f677c1
using Metal

# ╔═╡ d3307b8b-48f5-42a7-ad2e-e8b69aac53e7
# ╠═╡ show_logs = false
using AMDGPU

# ╔═╡ 345a4744-bbf6-4334-ba15-aeb721416f5b
using oneAPI

# ╔═╡ 33e02405-1750-48f9-9776-d1d2d261f63f
using DrWatson: datadir

# ╔═╡ a968bcd8-fc42-45ec-af7c-68e73e8f1cd5
using PlutoUI: TableOfContents

# ╔═╡ 50e24ebe-403a-4d89-b02f-7a1577222838
using CSV: read

# ╔═╡ 50bfb09f-4dbb-4488-9284-7eef837ffe75
using DataFrames: DataFrame

# ╔═╡ d1a12515-a9d0-468b-8978-dbb26a1ee667
using CairoMakie: Figure, Axis, barplot!, Label, PolyElement, Legend, @L_str

# ╔═╡ e39675a9-08c7-4a4a-8eba-021862757a40
using CairoMakie # For the use of `Makie.wong_colors`

# ╔═╡ 30f67101-9626-4d01-a6fd-c260cd5c29b6
# CUDA.set_runtime_version!(v"11.8")

# ╔═╡ a4bd50ba-c963-4058-a648-2660b4eb29a7
CUDA.functional()

# ╔═╡ 0b951e25-5a96-439c-8fd3-df3e628b45dd
Metal.functional()

# ╔═╡ fc95a4bc-d3c8-4080-8634-1136b12210dd
AMDGPU.functional()

# ╔═╡ 75216390-cf6c-415c-83b9-3d139f712297
oneAPI.functional()

# ╔═╡ 278dfa0e-46e1-4789-9f51-eb3463a9fb00
TableOfContents()

# ╔═╡ 0def112f-e739-4a35-a223-c9a244a0d139
md"""
## Load DataFrames
"""

# ╔═╡ ad97f6cb-c331-4898-9c6c-485582058e4d
df_metal_2d = read(datadir("new", "dt_2D_Metal.csv"), DataFrame);

# ╔═╡ 83f4fd58-e801-4dda-9ba7-f5eec56722f6
df_cuda_2d = read(datadir("new", "dt_2D_CUDA.csv"), DataFrame);

# ╔═╡ eb190959-b90f-4dbb-8ae7-09b964e1a1c2
df_metal_3d = read(datadir("new", "dt_3D_Metal.csv"), DataFrame);

# ╔═╡ 1936dff5-1d17-4773-9009-51ec95eb9411
df_cuda_3d = read(datadir("new", "dt_3D_CUDA.csv"), DataFrame);

# ╔═╡ 97a7f98f-738d-4870-8872-4d0e7ba88c4a
md"""
## 2D
"""

# ╔═╡ 3175900c-7676-4ab2-b31a-a0ca3b1afde7
function create_barplot(
	title, dt_names;
	sizes = [],
	dt_maurer = [],
	dt_fenz = [],
	dt_fenz_multi = [],
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

# ╔═╡ 492df5fa-e20e-4dcb-8c1f-b7e14d9fc2de
title_2d = "Performance Comparison \nof Distance Transforms (2D)"

# ╔═╡ 7bc02cb0-76e9-4654-b17a-9d95089bf472
dt_names_2d = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (Metal)", "Proposed (oneAPI)", "Proposed (AMDGPU)"]

# ╔═╡ b50a4061-4f49-4578-8671-1746d532c9dc
range_names_2d = [L"2^3", L"2^4", L"2^5", L"2^6", L"2^7", L"2^8", L"2^9", L"2^{10}", L"2^{11}", L"2^{12}"]

# ╔═╡ 6de86d5d-753c-478e-bf9a-0b7c111192bb
dt_fig_2d = create_barplot(
	title_2d, dt_names_2d;
	sizes = df_metal_2d[:, :sizes],
	dt_maurer = df_metal_2d[:, :dt_maurer],
	dt_fenz = df_metal_2d[:, :dt_fenz],
	dt_fenz_multi = df_metal_2d[:, :dt_fenz_multi],
	dt_proposed_cuda = df_cuda_2d[:, :dt_proposed_cuda],
	dt_proposed_metal = df_metal_2d[:, :dt_proposed_metal],
	x_names = range_names_2d
)

# ╔═╡ 1510781f-cf67-4cd1-852f-4200d4a437a8
save(joinpath(dirname(pwd()), "plots", "julia_benchmarks_2d.png"), dt_fig_2d)

# ╔═╡ 54b95ed5-04e1-401d-b240-5a561b2e6713
md"""
## 3D
"""

# ╔═╡ 08676d5b-f098-43a9-8bc3-b5cda3282b2a
title_3d = "Performance Comparison \nof Distance Transforms (3D)"

# ╔═╡ f093102d-4796-4d05-943c-c314febe7342
dt_names_3d = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (Metal)", "Proposed (oneAPI)", "Proposed (AMDGPU)"]

# ╔═╡ 0c09ef6c-d05e-4f73-9075-78d9ba986bb9
range_names_3d = [L"2^0", L"2^1", L"2^2", L"2^3", L"2^4", L"2^5", L"2^6", L"2^7", L"2^8"]

# ╔═╡ 7b4e4368-8903-4279-9c16-32cf8b603949
dt_fig_3d = create_barplot(
	title_3d, dt_names_3d;
	sizes = df_metal_3d[:, :sizes_3D],
	dt_maurer = df_metal_3d[:, :dt_maurer_3D],
	dt_fenz = df_metal_3d[:, :dt_fenz_3D],
	dt_fenz_multi = df_metal_3d[:, :dt_fenz_multi_3D],
	dt_proposed_cuda = df_cuda_3d[:, :dt_proposed_cuda_3D],
	dt_proposed_metal = df_metal_3d[:, :dt_proposed_metal_3D],
	x_names = range_names_3d
)

# ╔═╡ b69b8849-2dc7-4eaa-b58e-ae1525e18b80
save(joinpath(dirname(pwd()), "plots", "julia_benchmarks_3d.png"), dt_fig_3d)

# ╔═╡ Cell order:
# ╠═2c729da6-40e6-47cd-a14d-c152b8789b17
# ╠═30f67101-9626-4d01-a6fd-c260cd5c29b6
# ╠═5e56b525-cd2d-4e2d-9010-8210a08611be
# ╠═a4bd50ba-c963-4058-a648-2660b4eb29a7
# ╠═ee82e108-12d0-483e-84f4-92a7a3f677c1
# ╠═0b951e25-5a96-439c-8fd3-df3e628b45dd
# ╠═d3307b8b-48f5-42a7-ad2e-e8b69aac53e7
# ╠═fc95a4bc-d3c8-4080-8634-1136b12210dd
# ╠═345a4744-bbf6-4334-ba15-aeb721416f5b
# ╠═75216390-cf6c-415c-83b9-3d139f712297
# ╠═33e02405-1750-48f9-9776-d1d2d261f63f
# ╠═a968bcd8-fc42-45ec-af7c-68e73e8f1cd5
# ╠═50e24ebe-403a-4d89-b02f-7a1577222838
# ╠═50bfb09f-4dbb-4488-9284-7eef837ffe75
# ╠═d1a12515-a9d0-468b-8978-dbb26a1ee667
# ╠═e39675a9-08c7-4a4a-8eba-021862757a40
# ╠═278dfa0e-46e1-4789-9f51-eb3463a9fb00
# ╟─0def112f-e739-4a35-a223-c9a244a0d139
# ╠═ad97f6cb-c331-4898-9c6c-485582058e4d
# ╠═83f4fd58-e801-4dda-9ba7-f5eec56722f6
# ╠═eb190959-b90f-4dbb-8ae7-09b964e1a1c2
# ╠═1936dff5-1d17-4773-9009-51ec95eb9411
# ╟─97a7f98f-738d-4870-8872-4d0e7ba88c4a
# ╟─3175900c-7676-4ab2-b31a-a0ca3b1afde7
# ╠═492df5fa-e20e-4dcb-8c1f-b7e14d9fc2de
# ╠═7bc02cb0-76e9-4654-b17a-9d95089bf472
# ╠═b50a4061-4f49-4578-8671-1746d532c9dc
# ╟─6de86d5d-753c-478e-bf9a-0b7c111192bb
# ╠═1510781f-cf67-4cd1-852f-4200d4a437a8
# ╟─54b95ed5-04e1-401d-b240-5a561b2e6713
# ╠═08676d5b-f098-43a9-8bc3-b5cda3282b2a
# ╠═f093102d-4796-4d05-943c-c314febe7342
# ╠═0c09ef6c-d05e-4f73-9075-78d9ba986bb9
# ╟─7b4e4368-8903-4279-9c16-32cf8b603949
# ╠═b69b8849-2dc7-4eaa-b58e-ae1525e18b80
