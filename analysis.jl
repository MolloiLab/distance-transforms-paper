### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 2c729da6-40e6-47cd-a14d-c152b8789b17
# ╠═╡ show_logs = false
using Pkg; Pkg.activate("."), Pkg.instantiate();

# ╔═╡ 5e56b525-cd2d-4e2d-9010-8210a08611be
using CUDA

# ╔═╡ ee82e108-12d0-483e-84f4-92a7a3f677c1
using Metal

# ╔═╡ d3307b8b-48f5-42a7-ad2e-e8b69aac53e7
# ╠═╡ show_logs = false
using AMDGPU

# ╔═╡ 345a4744-bbf6-4334-ba15-aeb721416f5b
# ╠═╡ show_logs = false
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

# ╔═╡ 75216390-cf6c-415c-83b9-3d139f712297
oneAPI.functional()

# ╔═╡ 0b951e25-5a96-439c-8fd3-df3e628b45dd
Metal.functional()

# ╔═╡ fc95a4bc-d3c8-4080-8634-1136b12210dd
AMDGPU.functional()

# ╔═╡ a4bd50ba-c963-4058-a648-2660b4eb29a7
CUDA.functional()

# ╔═╡ 278dfa0e-46e1-4789-9f51-eb3463a9fb00
TableOfContents()

# ╔═╡ 8b786cbc-dd81-4208-9eee-2d7f7bbfa23f
md"""
# Julia DT Benchmarks
"""

# ╔═╡ ad97f6cb-c331-4898-9c6c-485582058e4d
df_metal_2d = read(datadir("dt_2D_Metal.csv"), DataFrame);

# ╔═╡ 83f4fd58-e801-4dda-9ba7-f5eec56722f6
df_cuda_2d = read(datadir("dt_2D_CUDA.csv"), DataFrame);

# ╔═╡ eb190959-b90f-4dbb-8ae7-09b964e1a1c2
df_metal_3d = read(datadir("dt_3D_Metal.csv"), DataFrame);

# ╔═╡ 1936dff5-1d17-4773-9009-51ec95eb9411
df_cuda_3d = read(datadir("dt_3D_CUDA.csv"), DataFrame);

# ╔═╡ 3175900c-7676-4ab2-b31a-a0ca3b1afde7
function create_barplot(
	title, dt_names;
	sizes = [],
	dt_maurer = [],
	dt_fenz = [],
	dt_fenz_multi = [],
	dt_proposed_cuda = [],
	dt_proposed_metal = [],
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
        isempty(dt_proposed_amdgpu) ? zeros(length(dt_maurer)) : dt_proposed_amdgpu
    )

    offset = 1
    for i in eachrow(heights)
        dt_heights[offset:(offset+length(i) - 1)] .= i
        offset += 6
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
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
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

# ╔═╡ 97a7f98f-738d-4870-8872-4d0e7ba88c4a
md"""
## 2D
"""

# ╔═╡ 492df5fa-e20e-4dcb-8c1f-b7e14d9fc2de
title_2d = "Performance Comparison \nof Julia Distance Transforms (2D)"

# ╔═╡ 7bc02cb0-76e9-4654-b17a-9d95089bf472
dt_names_2d = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (Metal)", "Proposed (AMDGPU)"]

# ╔═╡ b50a4061-4f49-4578-8671-1746d532c9dc
range_names_2d = [L"(2^3)^2", L"(2^4)^2", L"(2^5)^2", L"(2^6)^2", L"(2^7)^2", L"(2^8)^2", L"(2^9)^2", L"(2^{10})^2", L"(2^{11})^2", L"(2^{12})^2"]

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
save(joinpath(pwd(), "plots", "julia_benchmarks_2d.png"), dt_fig_2d);

# ╔═╡ 54b95ed5-04e1-401d-b240-5a561b2e6713
md"""
## 3D
"""

# ╔═╡ 08676d5b-f098-43a9-8bc3-b5cda3282b2a
title_3d = "Performance Comparison \nof Julia Distance Transforms (3D)"

# ╔═╡ f093102d-4796-4d05-943c-c314febe7342
dt_names_3d = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (Metal)", "Proposed (AMDGPU)"]

# ╔═╡ 0c09ef6c-d05e-4f73-9075-78d9ba986bb9
range_names_3d = [L"(2^0)^3", L"(2^1)^3", L"(2^2)^3", L"(2^3)^3", L"(2^4)^3", L"(2^5)^3", L"(2^6)^3", L"(2^7)^3", L"(2^8)^3"]

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
save(joinpath(pwd(), "plots", "julia_benchmarks_3d.png"), dt_fig_3d);

# ╔═╡ 3fc76221-c854-46ce-9c85-8baa43ff7e14
md"""
# Python DT Benchmarks
"""

# ╔═╡ 3a436af8-66d5-4a82-85e1-d860fe52421f
df_py_2d = read(datadir("dt_py_2D_CUDA.csv"), DataFrame);

# ╔═╡ 8460f3a8-8c18-46ef-927e-125520db0db6
df_py_3d = read(datadir("dt_py_3D_CUDA.csv"), DataFrame);

# ╔═╡ 65d76d1e-6caf-4ad5-acdb-bed143711ee7
md"""
## 2D
"""

# ╔═╡ 1a75f51d-e66c-45d1-8f0a-b72cacbe263e
function create_barplot_py_2d(
	title, dt_names;
	sizes = [],
	dt_scipy = [],
	dt_tfa = [],
	dt_fastgeodis = [],
	dt_opencv = [],
	dt_pydt_single = [],
	dt_pydt_multi = [],
	dt_pydt_cuda = [],
	x_names = sizes
	)
    dt_heights = zeros(length(dt_names) * length(sizes))
    
    heights = hcat(
		dt_scipy,
        dt_tfa,
        dt_fastgeodis,
		dt_opencv,
		dt_pydt_single,
        dt_pydt_multi,
		dt_pydt_cuda
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
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
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

# ╔═╡ 124ab9eb-0959-4a16-9659-f58b01ccf463
title_2d_py = "Performance Comparison \nof Python Distance Transforms (2D)"

# ╔═╡ 5c9a9a74-ad6e-43d3-9dce-226f03dc3535
dt_names_2d_py = ["Scipy", "Tensorflow", "FastGeodis", "OpenCV", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)"]

# ╔═╡ d98d995b-09b6-4f3a-b7a7-4bb9effbaf7d
dt_fig_2d_py = create_barplot_py_2d(
	title_2d_py, dt_names_2d_py;
	sizes = df_py_2d[:, :sizes],
	dt_scipy = df_py_2d[:, :dt_scipy],
	dt_tfa = df_py_2d[:, :dt_tfa],
	dt_fastgeodis = df_py_2d[:, :dt_fastgeodis],
	dt_opencv = df_py_2d[:, :dt_opencv],
	dt_pydt_single = df_py_2d[:, :dt_pydt_single],
	dt_pydt_multi = df_py_2d[:, :dt_pydt_multi],
	dt_pydt_cuda = df_py_2d[:, :dt_pydt_multi],
	x_names = range_names_2d
)

# ╔═╡ 0caab0a0-667f-4862-9e1d-d20e331032fd
save(joinpath(pwd(), "plots", "py_benchmarks_2d.png"), dt_fig_2d_py);

# ╔═╡ 971ab05c-a849-4f44-a484-fec73f8f2326
md"""
## 3D
"""

# ╔═╡ e93f3bd7-9f84-435c-bc57-2aeee49c08f2
function create_barplot_py_3d(
	title, dt_names;
	sizes = [],
	dt_scipy = [],
	dt_tfa = [],
	dt_fastgeodis = [],
	dt_pydt_single = [],
	dt_pydt_multi = [],
	dt_pydt_cuda = [],
	x_names = sizes
	)
    dt_heights = zeros(length(dt_names) * length(sizes))
    
    heights = hcat(
		dt_scipy,
        dt_tfa,
        dt_fastgeodis,
		dt_pydt_single,
        dt_pydt_multi,
		dt_pydt_cuda
	)

    offset = 1
    for i in eachrow(heights)
        dt_heights[offset:(offset+length(i) - 1)] .= i
        offset += 6
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
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
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

# ╔═╡ 7760006c-a8cd-4019-a54f-a4256d17f39b
title_3d_py = "Performance Comparison \nof Python Distance Transforms (3D)"

# ╔═╡ ea709965-e6dc-43a2-8472-8169fffb8447
dt_names_3d_py = ["Scipy", "Tensorflow", "FastGeodis", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)"]

# ╔═╡ 520a1602-1d22-4ec5-9410-01124c98feb9
dt_fig_3d_py = create_barplot_py_3d(
	title_3d_py, dt_names_3d_py;
	sizes = df_py_3d[:, :sizes_3D],
	dt_scipy = df_py_3d[:, :dt_scipy_3D],
	dt_tfa = df_py_3d[:, :dt_tfa_3D],
	dt_fastgeodis = df_py_3d[:, :dt_fastgeodis_3D],
	dt_pydt_single = df_py_3d[:, :dt_pydt_single_3D],
	dt_pydt_multi = df_py_3d[:, :dt_pydt_multi_3D],
	dt_pydt_cuda = df_py_3d[:, :dt_pydt_cuda_3D],
	x_names = range_names_3d
)

# ╔═╡ 37087a33-18d1-4e4d-b7e0-229b3b4aa797
save(joinpath(pwd(), "plots", "py_benchmarks_3d.png"), dt_fig_3d_py);

# ╔═╡ Cell order:
# ╠═2c729da6-40e6-47cd-a14d-c152b8789b17
# ╠═30f67101-9626-4d01-a6fd-c260cd5c29b6
# ╠═5e56b525-cd2d-4e2d-9010-8210a08611be
# ╠═ee82e108-12d0-483e-84f4-92a7a3f677c1
# ╠═d3307b8b-48f5-42a7-ad2e-e8b69aac53e7
# ╠═345a4744-bbf6-4334-ba15-aeb721416f5b
# ╠═33e02405-1750-48f9-9776-d1d2d261f63f
# ╠═a968bcd8-fc42-45ec-af7c-68e73e8f1cd5
# ╠═50e24ebe-403a-4d89-b02f-7a1577222838
# ╠═50bfb09f-4dbb-4488-9284-7eef837ffe75
# ╠═d1a12515-a9d0-468b-8978-dbb26a1ee667
# ╠═e39675a9-08c7-4a4a-8eba-021862757a40
# ╠═75216390-cf6c-415c-83b9-3d139f712297
# ╠═0b951e25-5a96-439c-8fd3-df3e628b45dd
# ╠═fc95a4bc-d3c8-4080-8634-1136b12210dd
# ╠═a4bd50ba-c963-4058-a648-2660b4eb29a7
# ╠═278dfa0e-46e1-4789-9f51-eb3463a9fb00
# ╟─8b786cbc-dd81-4208-9eee-2d7f7bbfa23f
# ╠═ad97f6cb-c331-4898-9c6c-485582058e4d
# ╠═83f4fd58-e801-4dda-9ba7-f5eec56722f6
# ╠═eb190959-b90f-4dbb-8ae7-09b964e1a1c2
# ╠═1936dff5-1d17-4773-9009-51ec95eb9411
# ╟─3175900c-7676-4ab2-b31a-a0ca3b1afde7
# ╟─97a7f98f-738d-4870-8872-4d0e7ba88c4a
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
# ╟─3fc76221-c854-46ce-9c85-8baa43ff7e14
# ╠═3a436af8-66d5-4a82-85e1-d860fe52421f
# ╠═8460f3a8-8c18-46ef-927e-125520db0db6
# ╟─65d76d1e-6caf-4ad5-acdb-bed143711ee7
# ╟─1a75f51d-e66c-45d1-8f0a-b72cacbe263e
# ╠═124ab9eb-0959-4a16-9659-f58b01ccf463
# ╠═5c9a9a74-ad6e-43d3-9dce-226f03dc3535
# ╟─d98d995b-09b6-4f3a-b7a7-4bb9effbaf7d
# ╠═0caab0a0-667f-4862-9e1d-d20e331032fd
# ╟─971ab05c-a849-4f44-a484-fec73f8f2326
# ╟─e93f3bd7-9f84-435c-bc57-2aeee49c08f2
# ╠═7760006c-a8cd-4019-a54f-a4256d17f39b
# ╠═ea709965-e6dc-43a2-8472-8169fffb8447
# ╟─520a1602-1d22-4ec5-9410-01124c98feb9
# ╠═37087a33-18d1-4e4d-b7e0-229b3b4aa797
