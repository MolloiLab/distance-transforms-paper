using Pkg; Pkg.activate("."), Pkg.instantiate();
using DrWatson: datadir
using DataFrames: DataFrame
using CairoMakie: Figure, Axis, barplot!, Label, PolyElement, Legend, @L_str
using CairoMakie # For the use of `Makie.wong_colors`
using DataFrames: rename!, Not, select
# using Images
import JSON3
import CSV

# CUDA.set_runtime_version!(v"11.8")

########################################
# Julia DT Benchmarks
########################################

df_metal_2d = CSV.read(datadir("dt_2D_Metal.csv"), DataFrame);
df_cuda_2d = CSV.read(datadir("dt_2D_CUDA.csv"), DataFrame);
df_oneapi_2d = CSV.read(datadir("dt_2D_oneAPI.csv"), DataFrame);
df_amdgpu_2d = CSV.read(datadir("dt_2D_AMDGPU.csv"), DataFrame);
df_metal_3d = CSV.read(datadir("dt_3D_Metal.csv"), DataFrame);
df_cuda_3d = CSV.read(datadir("dt_3D_CUDA.csv"), DataFrame);
df_oneapi_3d = CSV.read(datadir("dt_3D_oneAPI.csv"), DataFrame);
df_amdgpu_3d = CSV.read(datadir("dt_3D_AMDGPU.csv"), DataFrame);

title_2d = "Performance Comparison \nof Julia Distance Transforms (2D)"
dt_names_2d = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (AMDGPU)", "Proposed (Metal)", "Proposed (oneAPI)"]
range_names_2d = [L"(2^3)^2", L"(2^4)^2", L"(2^5)^2", L"(2^6)^2", L"(2^7)^2", L"(2^8)^2", L"(2^9)^2", L"(2^{10})^2", L"(2^{11})^2", L"(2^{12})^2"]
title_3d = "Performance Comparison \nof Julia Distance Transforms (3D)"
dt_names_3d = ["Maurer", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)", "Proposed (AMDGPU)", "Proposed (Metal)", "Proposed (oneAPI)"]
range_names_3d = [L"(2^0)^3", L"(2^1)^3", L"(2^2)^3", L"(2^3)^3", L"(2^4)^3", L"(2^5)^3", L"(2^6)^3", L"(2^7)^3", L"(2^8)^3"]

########################################
## Combined Table
########################################

function combine_dataframes(df_2d::DataFrame, df_3d::DataFrame; cols_2d, cols_3d=cols_2d)
	# Select specified columns from both dataframes
	df_2d_selected = df_2d[!, cols_2d]
	df_3d_selected = df_3d[!, cols_3d]
	# Rename columns in df_3d to match those in df_2d
	rename!(df_3d_selected, names(df_2d_selected))
	# Vertically concatenate the selected dataframes
	df_combined = vcat(df_2d_selected, df_3d_selected)
	return df_combined
end

function process_multiple_dataframe_pairs(pairs)
	combined_dfs = []
	for pair in pairs
		# Get cols_3d if specified; otherwise, default to cols_2d
		cols_3d = get(pair, :cols_3d, pair.cols_2d)
		# Combine the dataframes
		df_combined = combine_dataframes(pair.df_2d, pair.df_3d; cols_2d=pair.cols_2d, cols_3d=cols_3d)
		push!(combined_dfs, df_combined)
	end
	# Horizontally concatenate all combined dataframes
	df_comprehensive = hcat(combined_dfs...; makeunique = true)
	return df_comprehensive
end

pairs = [
	(
		df_2d = df_cuda_2d, df_3d = df_cuda_3d, cols_2d = 3:size(df_metal_2d, 2)
	),
	(
		df_2d = df_metal_2d, df_3d = df_metal_3d, cols_2d = (size(df_cuda_2d, 2)-1):size(df_cuda_2d, 2)
	),
	(
		df_2d = df_amdgpu_2d, df_3d = df_amdgpu_3d, cols_2d = 6:7
	),
	(
		df_2d = df_oneapi_2d, df_3d = df_oneapi_3d, cols_2d = 6:7
	)
];

df_comprehensive = process_multiple_dataframe_pairs(pairs);

begin
	df_clean = select(df_comprehensive, Not(r"std"i))
	rename!(df_clean, Dict(
	    "sizes" => "Array Size",
		"dt_maurer" => "Maurer (Multi-Threaded)",
		"dt_fenz" => "Felzenszwalb",
		"dt_fenz_multi" => "Felzenszwalb (Multi-Threaded)",
		"dt_proposed_cuda" => "Proposed (CUDA)",
		"dt_proposed_metal" => "Proposed (Metal)",
		"dt_proposed_amdgpu" => "Proposed (AMDGPU)",
		"dt_proposed_oneapi" => "Proposed (oneAPI)",
	))
	df_clean[!, "Array Size"] .= [
		"8²", "16²", "32²", "64²", "128²", "256²", "512²", "1024²", "2048²", "4096²", "1³", "2³", "4³", "8³", "16³", "32³", "64³", "128³", "256³"
	]

	df_clean_cp = copy(df_clean)
	for (i, col) in enumerate(eachcol(df_clean))
		if eltype(col) == Float64
			df_clean_cp[:, i] = round.(col; sigdigits=3)
		end
	end
	df_clean_cp
end

let
	f = Figure(size = (800, 600))
	ax = Axis(
		f[1, 1],
		yscale = log10
	)
	scatterlines!(df_clean_cp[!, "Maurer (Multi-Threaded)"]; label = "Maurer (Multi-Threaded)")
	scatterlines!(df_clean_cp[!, "Felzenszwalb"]; label = "Felzenszwalb")
	scatterlines!(df_clean_cp[!, "Felzenszwalb (Multi-Threaded)"]; label = "Felzenszwalb (Multi-Threaded)")
	scatterlines!(df_clean_cp[!, "Proposed (CUDA)"]; label = "Proposed (CUDA)")
	scatterlines!(df_clean_cp[!, "Proposed (Metal)"]; label = "Proposed (Metal)")
	scatterlines!(df_clean_cp[!, "Proposed (AMDGPU)"]; label = "Proposed (AMDGPU)")
	scatterlines!(df_clean_cp[!, "Proposed (oneAPI)"]; label = "Proposed (oneAPI)")

	axislegend(ax; position = :lt)
	f
end

########################################
## Combined Barplot
########################################

let
	### ------------------- 2D PLOT ------------------- ###
	title_2d = "Performance Comparison \nof Julia Distance Transforms (2D)"
	# dt_names_2d = dt_names_2d
	sizes_2d = df_metal_2d[:, :sizes]
	dt_maurer_2d = df_metal_2d[:, :dt_maurer]
	dt_fenz_2d = df_metal_2d[:, :dt_fenz]
	dt_fenz_multi_2d = df_metal_2d[:, :dt_fenz_multi]
	dt_proposed_cuda_2d = df_cuda_2d[:, :dt_proposed_cuda]
	dt_proposed_amdgpu_2d = df_amdgpu_2d[:, :dt_proposed_amdgpu]
	dt_proposed_metal_2d = df_metal_2d[:, :dt_proposed_metal]
	dt_proposed_oneapi_2d = df_oneapi_2d[:, :dt_proposed_oneapi]
	x_names_2d = range_names_2d
	
	dt_heights_2d = zeros(length(dt_names_2d) * length(sizes_2d))
	
	heights_2d = hcat(
		dt_maurer_2d,
		dt_fenz_2d,
		dt_fenz_multi_2d,
		dt_proposed_cuda_2d,
		dt_proposed_amdgpu_2d,
		dt_proposed_metal_2d,
		dt_proposed_oneapi_2d,
	)

	offset_2d = 1
	for i in eachrow(heights_2d)
		dt_heights_2d[offset_2d:(offset_2d+length(i) - 1)] .= i
		offset_2d += 7
	end

	cat_2d = repeat(1:length(sizes_2d), inner = length(dt_names_2d))
	grp_2d = repeat(1:length(dt_names_2d), length(sizes_2d))
	colors = Makie.wong_colors()

	f = Figure(size = (800, 900))
	ax_2d = Axis(
		f[1:2, 1:2],
		ylabel = "Time (ns)",
		title = title_2d,
		titlesize = 25,
		xticks = (1:length(sizes_2d), x_names_2d),
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
		yscale = log10,
		xgridvisible = false,
		ygridvisible = false
	)

	barplot!(
		cat_2d, dt_heights_2d;
		dodge = grp_2d,
		color = colors[grp_2d],
	)
	
	# X axis label
	Label(f[3, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

	### ------------------- 3D PLOT ------------------- ###
	title_3d = "Performance Comparison \nof Julia Distance Transforms (3D)"
	# dt_names_3d = dt_names_3d
	sizes_3d = df_metal_3d[:, :sizes_3D]
	dt_maurer_3d = df_metal_3d[:, :dt_maurer_3D]
	dt_fenz_3d = df_metal_3d[:, :dt_fenz_3D]
	dt_fenz_multi_3d = df_metal_3d[:, :dt_fenz_multi_3D]
	dt_proposed_cuda_3d = df_cuda_3d[:, :dt_proposed_cuda_3D]
	dt_proposed_amdgpu_3d = df_amdgpu_3d[:, :dt_proposed_amdgpu_3D]
	dt_proposed_metal_3d = df_metal_3d[:, :dt_proposed_metal_3D]
	dt_proposed_oneapi_3d = df_oneapi_3d[:, :dt_proposed_oneapi_3D]
	x_names_3d = range_names_3d
	
	dt_heights_3d = zeros(length(dt_names_3d) * length(sizes_3d))
	
	heights_3d = hcat(
		dt_maurer_3d,
		dt_fenz_3d,
		dt_fenz_multi_3d,
		dt_proposed_cuda_3d,
		dt_proposed_amdgpu_3d,
		dt_proposed_metal_3d,
		dt_proposed_oneapi_3d,
	)

	offset_3d = 1
	for i in eachrow(heights_3d)
		dt_heights_3d[offset_3d:(offset_3d+length(i) - 1)] .= i
		offset_3d += 7
	end

	cat_3d = repeat(1:length(sizes_3d), inner = length(dt_names_3d))
	grp_3d = repeat(1:length(dt_names_3d), length(sizes_3d))

	ax_3d = Axis(
		f[4:5, 1:2],
		ylabel = "Time (ns)",
		title = title_3d,
		titlesize = 25,
		xticks = (1:length(sizes_3d), x_names_3d),
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
		yscale = log10,
		xgridvisible = false,
		ygridvisible = false
	)

	barplot!(
		cat_3d, dt_heights_3d;
		dodge = grp_3d,
		color = colors[grp_3d],
	)

	# X axis label
	Label(f[6, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

	# CPU Legend
	rnge = 1:3
    labels = dt_names_2d[rnge]
    elements = [PolyElement(polycolor = colors[i]) for i in rnge]
    title = "Distance Transform \nAlgorithms (CPU)"
    Legend(f[2:3, 3], elements, labels, title)

	# GPU Legend
	rnge = 4:7
    labels = dt_names_2d[rnge]
    elements = [PolyElement(polycolor = colors[i]) for i in rnge]
    title = "Distance Transform \nAlgorithms (GPU)"
    Legend(f[3:4, 3], elements, labels, title)

	save(joinpath(pwd(), "plots/julia_distance_transforms.png"), f)
	f
end

########################################
## JSON version
########################################

data = JSON3.read(read(datadir("combined_benchmarks.json"), String))
benchmark_groups = data[2][1][2][:data]  # Get the full benchmark data
benchmark_groups

function extract_memory_cpu(data, dimension)
    memory_dict = Dict()
    
    for (thread_key, thread_data) in data
        thread_key_str = string(thread_key)
        
        # Skip if the dimension is not present in the thread data
        if !haskey(thread_data[2][:data], dimension)
            continue
        end
        
        dim_data = thread_data[2][:data][dimension][2][:data]
        
        for (size_key, size_data) in dim_data
            if !haskey(memory_dict, size_key)
                memory_dict[size_key] = Dict()
            end
            
            bench_data = size_data[2][:data]
            
            if thread_key_str == "CPU_1thread"
                # Process Maurer data with 1 thread
                if haskey(bench_data, "Maurer") && haskey(bench_data["Maurer"][2][:data], "CPU")
                    maurer_cpu = bench_data["Maurer"][2][:data]["CPU"][2][:data]
                    for (_, entry) in maurer_cpu
                        memory_dict[size_key]["Maurer"] = entry[2][:memory]
                    end
                end
                
                # Process Felzenszwalb data with 1 thread
                if haskey(bench_data, "Felzenszwalb") && haskey(bench_data["Felzenszwalb"][2][:data], "CPU")
                    felz_cpu = bench_data["Felzenszwalb"][2][:data]["CPU"][2][:data]
                    for (_, entry) in felz_cpu
                        memory_dict[size_key]["Felzenszwalb"] = entry[2][:memory]
                    end
                end
                
            elseif thread_key_str == "CPU_4thread"
                # Process Felzenszwalb_MT data with 4 threads (named "Felzenszwalb MT")
                if haskey(bench_data, "Felzenszwalb_MT") && haskey(bench_data["Felzenszwalb_MT"][2][:data], "CPU")
                    felz_mt_cpu = bench_data["Felzenszwalb_MT"][2][:data]["CPU"][2][:data]
                    for (_, entry) in felz_mt_cpu
                        memory_dict[size_key]["Felzenszwalb MT"] = entry[2][:memory]
                    end
                end
                
            elseif thread_key_str in ["CUDA", "oneAPI", "AMDGPU", "Metal"]
                # Process GPU data
                if haskey(bench_data, "Felzenszwalb") && haskey(bench_data["Felzenszwalb"][2][:data], "GPU")
                    gpu_data = bench_data["Felzenszwalb"][2][:data]["GPU"][2][:data]
                    if haskey(gpu_data, thread_key_str)
                        memory_dict[size_key][thread_key_str] = gpu_data[thread_key_str][2][:memory]
                    end
                end
            end
        end
    end
    
    return memory_dict
end

memory_2d_cpu = extract_memory_cpu(benchmark_groups, "2D")
memory_3d_cpu = extract_memory_cpu(benchmark_groups, "3D")

# Function to create DataFrame from memory dictionary
function create_mem_df(memory)
	# Initialize arrays for each column
	sizes = String[]
	mem_maurer = Float64[]
	mem_fenz = Float64[]
	mem_fenz_multi = Float64[]
	mem_proposed_cuda = Float64[]
	mem_proposed_oneapi = Float64[]
	mem_proposed_metal = Float64[]
	mem_proposed_amdgpu = Float64[]

	# Populate arrays
	for (size_key, size_data) in memory
		push!(sizes, string(size_key))
		
		# Extract memory allocations, defaulting to NaN if not present
		push!(mem_maurer, get(size_data, "Maurer", NaN))
		push!(mem_fenz, get(size_data, "Felzenszwalb", NaN))
		push!(mem_fenz_multi, get(size_data, "Felzenszwalb MT", NaN))
		push!(mem_proposed_cuda, get(size_data, "CUDA", NaN))
		push!(mem_proposed_oneapi, get(size_data, "oneAPI", NaN))
		push!(mem_proposed_metal, get(size_data, "Metal", NaN))
		push!(mem_proposed_amdgpu, get(size_data, "AMDGPU", NaN))
	end

	# Create DataFrame
	df = DataFrame(
		sizes = sizes,
		mem_maurer = mem_maurer,
		mem_fenz = mem_fenz,
		mem_fenz_multi = mem_fenz_multi,
		mem_proposed_cuda = mem_proposed_cuda,
		mem_proposed_oneapi = mem_proposed_oneapi,
		mem_proposed_metal = mem_proposed_metal,
		mem_proposed_amdgpu = mem_proposed_amdgpu
	)
	return df
end

# Create DataFrames
df_mem_2d_cpu = create_mem_df(memory_2d_cpu)
df_mem_3d_cpu = create_mem_df(memory_3d_cpu)

# Process and sort DataFrames (same as before)
df_mem_2d_cpu.sizes = map(extract_size_number, df_mem_2d_cpu.sizes)
sort!(df_mem_2d_cpu, :sizes)

df_mem_3d_cpu.sizes = map(extract_size_number, df_mem_3d_cpu.sizes)
sort!(df_mem_3d_cpu, :sizes)

# Read GPU memory data from JSON files
cuda_memory = JSON3.read(read(datadir("cuda_memory.json"), String))
metal_memory = JSON3.read(read(datadir("metal_memory.json"), String))
amdgpu_memory = JSON3.read(read(datadir("amdgpu_memory.json"), String))
oneapi_memory = JSON3.read(read(datadir("oneapi_memory.json"), String))

function extract_memory_gpu(data, dimension)
    memory_dict = Dict()
    
    # Iterate through all CPU thread variations
    for (thread_key, thread_data) in data
        if !haskey(thread_data[2][:data], dimension)
            continue
        end
        
        dim_data = thread_data[2][:data][dimension][2][:data]
        
        for (size_key, size_data) in dim_data
            if !haskey(memory_dict, size_key)
                memory_dict[size_key] = Dict()
            end
            
            # Get the actual benchmark data
            bench_data = size_data[2][:data]
            
            # Extract Maurer data (CPU memory)
            if haskey(bench_data, "Maurer") && haskey(bench_data["Maurer"][2][:data], "CPU")
                maurer_data = bench_data["Maurer"][2][:data]["CPU"][2][:data]
                for (thread_count, thread_data) in maurer_data
                    memory_dict[size_key]["Maurer"] = thread_data[2][:memory]
                end
            end
            
            # Extract Felzenszwalb data (CPU memory)
            if haskey(bench_data, "Felzenszwalb") && haskey(bench_data["Felzenszwalb"][2][:data], "CPU")
                felz_data = bench_data["Felzenszwalb"][2][:data]
                if haskey(felz_data, "CPU")
                    cpu_data = felz_data["CPU"][2][:data]
                    for (thread_count, thread_data) in cpu_data
                        memory_dict[size_key]["Felzenszwalb"] = thread_data[2][:memory]
                    end
                end
            end
            
            # Extract Felzenszwalb MT data (CPU memory)
            if haskey(bench_data, "Felzenszwalb_MT") && haskey(bench_data["Felzenszwalb_MT"][2][:data], "CPU")
                felz_mt_data = bench_data["Felzenszwalb_MT"][2][:data]
                if haskey(felz_mt_data, "CPU")
                    cpu_data = felz_mt_data["CPU"][2][:data]
                    for (thread_count, thread_data) in cpu_data
                        memory_dict[size_key]["Felzenszwalb MT"] = thread_data[2][:memory]
                    end
                end
            end
            
            # GPU memory from separate JSON files (only need to do this once per size)
            key = "$(dimension)_$(size_key)"
            memory_dict[size_key]["CUDA"] = get(cuda_memory, key, NaN)
            memory_dict[size_key]["Metal"] = get(metal_memory, key, NaN)
            memory_dict[size_key]["AMDGPU"] = get(amdgpu_memory, key, NaN)
            memory_dict[size_key]["oneAPI"] = get(oneapi_memory, key, NaN)
        end
    end
    
    return memory_dict
end


memory_2d_gpu = extract_memory_gpu(benchmark_groups, "2D")
memory_3d_gpu = extract_memory_gpu(benchmark_groups, "3D")

# Create initial DataFrames
df_mem_2d_gpu = create_mem_df(memory_2d_gpu)
df_mem_3d_gpu = create_mem_df(memory_3d_gpu)

# Process and sort DataFrames
df_mem_2d_gpu.sizes = map(extract_size_number, df_mem_2d_gpu.sizes)
sort!(df_mem_2d_gpu, :sizes)

df_mem_3d_gpu.sizes = map(extract_size_number, df_mem_3d_gpu.sizes)
sort!(df_mem_3d_gpu, :sizes)


function plot_benchmarks(df_2d_cpu, df_3d_cpu, df_2d_gpu, df_3d_gpu)
    f = Figure(size = (800, 900))
    dt_names = dt_names_2d[1:end-1]  # Ensure dt_names does not include "oneAPI" in the GPU section

    ### ------------------- 2D PLOT ------------------- ###
    title_2d = "Memory Usage Comparison \nof Julia Distance Transforms (2D)"
    sizes_2d = df_2d_cpu.sizes
    x_names_2d = range_names_2d

    # Exclude oneAPI from the 2D data matrix
    heights_2d = hcat(
        df_2d_cpu.mem_maurer ./ (1024^2),
        df_2d_cpu.mem_fenz ./ (1024^2),
        df_2d_cpu.mem_fenz_multi ./ (1024^2),
        df_2d_gpu.mem_proposed_cuda ./ (1024^2),
        df_2d_gpu.mem_proposed_amdgpu ./ (1024^2),
        df_2d_gpu.mem_proposed_metal ./ (1024^2),
        # Removed: df_2d_gpu.mem_proposed_oneapi
    )

    dt_heights_2d = vec(heights_2d')
    cat_2d = repeat(1:length(sizes_2d), inner=length(dt_names))
    grp_2d = repeat(1:length(dt_names), length(sizes_2d))
    colors = Makie.wong_colors()

    ax_2d = Axis(
        f[1:2, 1:2],
        ylabel = "Memory (MiB)",
        title = title_2d,
        titlesize = 25,
        xticks = (1:length(sizes_2d), x_names_2d),
		yticks = (
			[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
			[L"1 \times 10^{-6}", L"1 \times 10^{-5}", L"1 \times 10^{-4}", L"1 \times 10^{-3}", L"1 \times 10^{-2}", L"1 \times 10^{-1}", L"1 \times 10^{0}", L"1 \times 10^{1}", L"1 \times 10^{2}", L"1 \times 10^{3}"]
		),
        yscale = log10,
        xgridvisible = false,
        ygridvisible = false
    )

    barplot!(ax_2d, cat_2d, dt_heights_2d;
        dodge = grp_2d,
        color = colors[grp_2d],
    )

    ### ------------------- 3D PLOT ------------------- ###
    title_3d = "Memory Usage Comparison \nof Julia Distance Transforms (3D)"
    sizes_3d = df_3d_cpu.sizes
    x_names_3d = range_names_3d

    # Exclude oneAPI from the 3D data matrix
    heights_3d = hcat(
        df_3d_cpu.mem_maurer ./ (1024^2),
        df_3d_cpu.mem_fenz ./ (1024^2),
        df_3d_cpu.mem_fenz_multi ./ (1024^2),
        df_3d_gpu.mem_proposed_cuda ./ (1024^2),
        df_3d_gpu.mem_proposed_amdgpu ./ (1024^2),
        df_3d_gpu.mem_proposed_metal ./ (1024^2),
        # Removed: df_3d_gpu.mem_proposed_oneapi
    )

    dt_heights_3d = vec(heights_3d')
    cat_3d = repeat(1:length(sizes_3d), inner=length(dt_names))
    grp_3d = repeat(1:length(dt_names), length(sizes_3d))

    ax_3d = Axis(
        f[4:5, 1:2],
        ylabel = "Memory (MiB)",
        title = title_3d,
        titlesize = 25,
        xticks = (1:length(sizes_3d), x_names_3d),
		yticks = (
			[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
			[L"1 \times 10^{-5}", L"1 \times 10^{-4}", L"1 \times 10^{-3}", L"1 \times 10^{-2}", L"1 \times 10^{-1}", L"1 \times 10^{0}", L"1 \times 10^{1}", L"1 \times 10^{2}", L"1 \times 10^{3}"]
		),
		yscale = log10,
        xgridvisible = false,
        ygridvisible = false
    )

    barplot!(ax_3d, cat_3d, dt_heights_3d;
        dodge = grp_3d,
        color = colors[grp_3d],
    )

    # X axis labels
    Label(f[3, 1:2], "Array Sizes", fontsize=14, padding=(0, 0, 0, 0))
    Label(f[6, 1:2], "Array Sizes", fontsize=14, padding=(0, 0, 0, 0))

    # Legends
    rnge_cpu = 1:3
    elements_cpu = [PolyElement(polycolor=colors[i]) for i in rnge_cpu]
    Legend(f[2:3, 3], elements_cpu, dt_names[rnge_cpu], "Distance Transform\nAlgorithms (CPU Memory)")

    rnge_gpu = 4:6  # Changed from 4:7 to exclude oneAPI
    elements_gpu = [PolyElement(polycolor=colors[i]) for i in rnge_gpu]
    Legend(f[3:4, 3], elements_gpu, dt_names[rnge_gpu], "Distance Transform\nAlgorithms (GPU Memory)")

	save(joinpath(pwd(), "plots/julia_distance_transforms_memory.png"), f)

    return f
end

f = plot_benchmarks(df_mem_2d_cpu, df_mem_3d_cpu, df_mem_2d_gpu, df_mem_3d_gpu)

heights_2d = hcat(
	df_mem_2d_cpu.mem_maurer ./ (1024^2),
	df_mem_2d_cpu.mem_fenz ./ (1024^2),
	df_mem_2d_cpu.mem_fenz_multi ./ (1024^2),
	df_mem_2d_gpu.mem_proposed_cuda ./ (1024^2),
	df_mem_2d_gpu.mem_proposed_amdgpu ./ (1024^2),
	df_mem_2d_gpu.mem_proposed_metal ./ (1024^2),
	# Removed: df_2d_gpu.mem_proposed_oneapi
)

heights_3d = hcat(
	df_mem_3d_cpu.mem_maurer ./ (1024^2),
	df_mem_3d_cpu.mem_fenz ./ (1024^2),
	df_mem_3d_cpu.mem_fenz_multi ./ (1024^2),
	df_mem_3d_gpu.mem_proposed_cuda ./ (1024^2),
	df_mem_3d_gpu.mem_proposed_amdgpu ./ (1024^2),
	df_mem_3d_gpu.mem_proposed_metal ./ (1024^2),
	# Removed: df_3d_gpu.mem_proposed_oneapi
)

# Define algorithm names (order must match heights_2d/heights_3d columns)
dt_names = [
    "Maurer (Multi-threaded)",
    "Felzenszwalb",
    "Felzenszwalb (Multi-threaded)",
    "Proposed (CUDA)",
    "Proposed (AMDGPU)",
    "Proposed (Metal)"
]

# Generate size labels with ²/³ suffixes
size_labels_2d = [string(s) * "²" for s in df_mem_2d_cpu.sizes]
size_labels_3d = [string(s) * "³" for s in df_mem_3d_cpu.sizes]

# Create DataFrames for 2D and 3D
df_2d = DataFrame(
    [:Size => size_labels_2d; [Symbol(name) => heights_2d[:, i] for (i, name) in enumerate(dt_names)]]
)

df_3d = DataFrame(
    [:Size => size_labels_3d; [Symbol(name) => heights_3d[:, i] for (i, name) in enumerate(dt_names)]]
)

# Combine vertically
df_mem_all = vcat(df_2d, df_3d)

show(df_mem_all; allrows=true, allcols=true, truncate=0)



########################################
# Python DT Benchmarks
########################################

df_py_2d = CSV.read(datadir("dt_py_2D_CUDA.csv"), DataFrame);
df_py_3d = CSV.read(datadir("dt_py_3D_CUDA.csv"), DataFrame);
title_2d_py = "Performance Comparison \nof Python Distance Transforms (2D)"
# dt_names_2d_py = ["Scipy", "Tensorflow", "FastGeodis", "OpenCV", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)"]
dt_names_2d_py = ["Scipy", "Tensorflow", "FastGeodis", "OpenCV", "Felzenszwalb", "Proposed (CUDA)"]
title_3d_py = "Performance Comparison \nof Python Distance Transforms (3D)"
# dt_names_3d_py = ["Scipy", "Tensorflow", "FastGeodis", "Felzenszwalb", "Felzenszwalb (Multi-threaded)", "Proposed (CUDA)"]
dt_names_3d_py = ["Scipy", "Tensorflow", "FastGeodis", "Felzenszwalb", "Proposed (CUDA)"]

########################################
# Combined Barplot
########################################

let
    ### ------------------- 2D PLOT ------------------- ###
    title_2d = title_2d_py
    dt_names_2d = dt_names_2d_py
    sizes_2d = df_py_2d[:, :sizes]
    dt_scipy_2d = df_py_2d[:, :dt_scipy]
    dt_tfa_2d = df_py_2d[:, :dt_tfa]
    dt_fastgeodis_2d = df_py_2d[:, :dt_fastgeodis]
    dt_opencv_2d = df_py_2d[:, :dt_opencv]
    dt_pydt_single_2d = df_py_2d[:, :dt_pydt_single]
    # dt_pydt_multi_2d = df_py_2d[:, :dt_pydt_multi]
    dt_pydt_cuda_2d = df_py_2d[:, :dt_pydt_cuda]
    x_names_2d = range_names_2d
    
    dt_heights_2d = zeros(length(dt_names_2d) * length(sizes_2d))
    
    heights_2d = hcat(
        dt_scipy_2d,
        dt_tfa_2d,
        dt_opencv_2d,
        dt_pydt_single_2d,
        # dt_pydt_multi_2d,
        dt_fastgeodis_2d,
        dt_pydt_cuda_2d
    )

    offset_2d = 1
    for i in eachrow(heights_2d)
        dt_heights_2d[offset_2d:(offset_2d+length(i) - 1)] .= i
        offset_2d += 6
    end

    cat_2d = repeat(1:length(sizes_2d), inner = length(dt_names_2d))
    grp_2d = repeat(1:length(dt_names_2d), length(sizes_2d))
    colors_2d = Makie.wong_colors()

    f = Figure(size = (800, 900))
    ax_2d = Axis(
        f[1:2, 1:2],
        ylabel = "Time (ns)",
        title = title_2d,
        titlesize = 25,
        xticks = (1:length(sizes_2d), x_names_2d),
        yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
        yscale = log10,
        xgridvisible = false,
        ygridvisible = false
    )

    barplot!(
        cat_2d, dt_heights_2d;
        dodge = grp_2d,
        color = colors_2d[grp_2d],
    )

	# X axis label
    Label(f[3, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

	# CPU Legend
	# rnge = [1, 2, 3, 4, 5]
	rnge = [1, 2, 3, 4]
	# labels = dt_names_2d[[1, 2, 4, 5, 6]]
	labels = dt_names_2d[[1, 2, 4, 5]]
	elements = [PolyElement(polycolor = colors_2d[i]) for i in rnge]
	title = "Distance Transform \nAlgorithms (CPU)"
	Legend(f[1, 3], elements, labels, title)

	# GPU Legend
	# rnge = [6, 7]
	rnge = [5, 6]
	# labels = dt_names_2d[[3, 7]]
	labels = dt_names_2d[[3, 6]]
	elements = [PolyElement(polycolor = colors_2d[i]) for i in rnge]
	title = "Distance Transform \nAlgorithms (GPU)"
	Legend(f[2, 3], elements, labels, title)

    ### ------------------- 3D PLOT ------------------- ###
    title_3d = title_3d_py
    dt_names_3d = dt_names_3d_py
    sizes_3d = df_py_3d[:, :sizes_3D]
    dt_scipy_3d = df_py_3d[:, :dt_scipy_3D]
    dt_tfa_3d = df_py_3d[:, :dt_tfa_3D]
    dt_fastgeodis_3d = df_py_3d[:, :dt_fastgeodis_3D]
    dt_pydt_single_3d = df_py_3d[:, :dt_pydt_single_3D]
    # dt_pydt_multi_3d = df_py_3d[:, :dt_pydt_multi_3D]
    dt_pydt_cuda_3d = df_py_3d[:, :dt_pydt_cuda_3D]
    x_names_3d = range_names_3d
    
    dt_heights_3d = zeros(length(dt_names_3d) * length(sizes_3d))
    
    heights_3d = hcat(
        dt_scipy_3d,
        dt_tfa_3d,
        dt_pydt_single_3d,
        # dt_pydt_multi_3d,
        dt_fastgeodis_3d,
        dt_pydt_cuda_3d
    )

    offset_3d = 1
    for i in eachrow(heights_3d)
        dt_heights_3d[offset_3d:(offset_3d+length(i) - 1)] .= i
        offset_3d += 5
    end

    cat_3d = repeat(1:length(sizes_3d), inner = length(dt_names_3d))
    grp_3d = repeat(1:length(dt_names_3d), length(sizes_3d))
    colors_3d = Makie.wong_colors()

    ax_3d = Axis(
        f[4:5, 1:2],
        ylabel = "Time (ns)",
        title = title_3d,
        titlesize = 25,
        xticks = (1:length(sizes_3d), x_names_3d),
        yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
        yscale = log10,
        xgridvisible = false,
        ygridvisible = false
    )

    barplot!(
        cat_3d, dt_heights_3d;
        dodge = grp_3d,
        color = colors_3d[grp_3d],
    )

    # X axis label
    Label(f[6, 1:2], "Array Sizes", fontsize = 14, padding = (0, 0, 0, 0))

	# CPU Legend
	rnge = [1, 2, 3]
	labels = dt_names_3d[[1, 2, 4]]
	elements = [PolyElement(polycolor = colors_3d[i]) for i in rnge]
	title = "Distance Transform \nAlgorithms (CPU)"
	Legend(f[4, 3], elements, labels, title)
	
	# GPU Legend
	rnge = [4, 5]
	labels = dt_names_3d[[3, 5]]
	elements = [PolyElement(polycolor = colors_3d[i]) for i in rnge]
	title = "Distance Transform \nAlgorithms (GPU)"
	Legend(f[5, 3], elements, labels, title)

	save(joinpath(pwd(), "plots/python_distance_transforms.png"), f)

    f
end

begin
	df_clean_py_2d = select(df_py_2d, Not(r"dt_pydt_mult"i))
	rename!(df_clean_py_2d, Dict(
	    "sizes" => "Array Size",
		"dt_scipy" => "Scipy",
		"dt_tfa" => "Tensorflow",
		"dt_fastgeodis" => "FastGeodis",
		"dt_opencv" => "OpenCV",
		"dt_pydt_single" => "Felzenszwalb",
		"dt_pydt_cuda" => "Proposed (CUDA)",
	))
	df_clean_py_2d[!, "Array Size"] .= [
		"8²", "16²", "32²", "64²", "128²", "256²", "512²", "1024²", "2048²", "4096²"
	]

	df_clean_py_2d_cp = copy(df_clean_py_2d)
	for (i, col) in enumerate(eachcol(df_clean_py_2d))
		if eltype(col) == Float64
			df_clean_py_2d_cp[:, i] = round.(col; sigdigits=3)
		end
	end
	df_clean_py_2d_cp
end

begin
	df_clean_py_3d = select(df_py_3d, Not(r"dt_pydt_mult"i))
	rename!(df_clean_py_3d, Dict(
	    "sizes_3D" => "Array Size",
		"dt_scipy_3D" => "Scipy",
		"dt_tfa_3D" => "Tensorflow",
		"dt_fastgeodis_3D" => "FastGeodis",
		"dt_pydt_single_3D" => "Felzenszwalb",
		"dt_pydt_cuda_3D" => "Proposed (CUDA)",
	))
	df_clean_py_3d[!, "Array Size"] .= [
		"1³", "2³", "4³", "8³", "16³", "32³", "64³", "128³", "256³"
	]

	df_clean_py_3d_cp = copy(df_clean_py_3d)
	for (i, col) in enumerate(eachcol(df_clean_py_3d))
		if eltype(col) == Float64
			df_clean_py_3d_cp[:, i] = round.(col; sigdigits=3)
		end
	end
	df_clean_py_3d_cp
end

########################################
# Hausdorff Loss
########################################

df_hd_loss_pure_losses_timings = CSV.read(joinpath(pwd(), "data/hd_loss_pure_losses_timings.csv"), DataFrame);
df_hd_loss_plain_dice_timing = CSV.read(joinpath(pwd(), "data/hd_loss_plain_dice_timing.csv"), DataFrame);
df_hd_loss_hd_dice_scipy_timing = CSV.read(joinpath(pwd(), "data/hd_loss_hd_dice_scipy_timing.csv"), DataFrame);
df_hd_loss_hd_dice_pydt_timing = CSV.read(joinpath(pwd(), "data/hd_loss_hd_dice_pydt_timing.csv"), DataFrame);

########################################
## Combined Barplot
########################################

let
	df = df_hd_loss_pure_losses_timings
	methods = ["Dice Loss", "HD Loss (Scipy)", "HD Loss (Proposed)"]	
	min_times = df[:, "Minimum Time (s)"]
	std_devs = df[:, "Standard Deviation (s)"]
	
	# Create the barplot
	fig = Figure(size = (800, 800))
	ax = Axis(
		fig[1, 1],
		ylabel = "Time (s)",
		title = "Pure Loss Function Timings",
		xticks = (1:length(methods), methods),
		yticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
		yscale = log10,
		ytickformat = "{:.2f}",  # Format y-axis tick labels as scientific notation with 2 decimal places
		xgridvisible = false,
		ygridvisible = false
	)
	
	colors = [:turquoise3, :mediumorchid3, :mediumseagreen]
	barplot!(
		1:length(methods), min_times;
		color = colors,
		bar_labels = string.(round.(min_times; sigdigits = 3))
	)
	ylims!(ax; high=1e2)

	df1 = df_hd_loss_plain_dice_timing
	df2 = df_hd_loss_hd_dice_scipy_timing
	df3 = df_hd_loss_hd_dice_pydt_timing
	
	methods = ["Dice Loss", "Dice + HD Loss (Scipy)", "Dice + HD Loss (Proposed)"]	
	min_times = [
		df1[:, "Avg Epoch Time (s)"]...,
		df2[:, "Avg Epoch Time (s)"]...,
		df3[:, "Avg Epoch Time (s)"]...
	]
	std_devs = [
		df1[:, "Std Epoch Time (s)"]...,
		df2[:, "Std Epoch Time (s)"]...,
		df3[:, "Std Epoch Time (s)"]...
	]
	
	ax = Axis(
		fig[2, 1],
		ylabel = "Time (s)",
		title = "Average Epoch Timings",
		xticks = (1:length(methods), methods),
		yticks = collect(0:10:50),
		xgridvisible = false,
		ygridvisible = false
	)
	
	barplot!(
		1:length(methods), min_times;
		color = colors,
		bar_labels = string.(round.(min_times; sigdigits = 3))
	)
	ylims!(ax; high=50)

	save(joinpath(pwd(), "plots/hd_loss_timings.png"), fig)
	# Adjust the layout and display the plot
	fig
end

########################################
# Training/Accuracy Metrics
########################################

df_training_results_dice = CSV.read(joinpath(pwd(), "data/training_results_dice.csv"), DataFrame);
df_training_results_hd_pydt = CSV.read(joinpath(pwd(), "data/training_results_hd_pydt.csv"), DataFrame);
df_hd_loss_metrics_dice = CSV.read(joinpath(pwd(), "data/hd_loss_metrics_dice.csv"), DataFrame);
df_hd_loss_metrics_dice = CSV.read(joinpath(pwd(), "data/hd_loss_metrics_dice.csv"), DataFrame);
df_hd_loss_metrics_hd_pydt = CSV.read(joinpath(pwd(), "data/hd_loss_metrics_hd_pydt.csv"), DataFrame);

begin
	df_dice = df_hd_loss_metrics_dice
	df_hd_dice = df_hd_loss_metrics_hd_pydt
	
	metrics = ["Dice Score", "IoU Score", "Hausdorff Distance", "95 Percentile Hausdorff Distance", "Total Training Time (s)"]
	
	# Extract the metric values for each model
	dice_values = [df_dice[1, metric] for metric in metrics]
	hd_dice_values = [df_hd_dice[1, metric] for metric in metrics]
end;

# Create a DataFrame with the metrics for each model
df_metrics = DataFrame(
	"Metric" => metrics,
	"Dice Loss" => dice_values,
	"HD + Dice Loss" => hd_dice_values
)

########################################
# Contour
########################################

img_dir = joinpath(pwd(), "plots/hd_contour_raw")
contour_imgs = readdir(img_dir)
img1 = load(joinpath(img_dir, contour_imgs[1]))[149:end-149, :];
img2 = load(joinpath(img_dir, contour_imgs[2]))[149:end-149, :];
img3 = load(joinpath(img_dir, contour_imgs[3]))[149:end-149, :];

let
	f = Figure(size = (700, 700))

	stp = 1
	ax = Axis(
		f[1, 1:stp],
		aspect = DataAspect()
	)
	hidespines!(ax)
	image!(ax, rotr90(img1))
	hidedecorations!(ax)

	ax = Axis(
		f[2, 1:stp],
		aspect = DataAspect()
	)
	hidespines!(ax)
	image!(ax, rotr90(img2))
	hidedecorations!(ax)

	
	ax = Axis(
		f[3, 1:stp],
		aspect = DataAspect()
	)
	hidespines!(ax)
	image!(ax, rotr90(img3))
	hidedecorations!(ax)

	save(joinpath(pwd(), "plots/hd_contours_raw.png"), f)
	f
end

########################################
# Skeletonization
########################################

df_skeleton = CSV.read(datadir("skeleton.csv"), DataFrame);

let
	sizes = df_skeleton[:, :sizes]
	cpu_timings = df_skeleton[:, "cpu timings"]
	gpu_timings = df_skeleton[:, "gpu timings"]
	
	f = Figure()
	ax = Axis(
		f[1, 1],
		ylabel = "Time (ns)",
		title = "Skeletonization Timings",
		yticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
		yscale = log10,
		xticks = (1:length(sizes), range_names_2d),
		xlabel = "Array Sizes"
	)
	scatterlines!(cpu_timings; label = "CPU")
	scatterlines!(gpu_timings; label = "Proposed (CUDA)")

	axislegend(ax; position = :rb)

	save(joinpath(pwd(), "plots/skeletonization.png"), f)
	
	f
end

df_skeleton[:, "cpu timings"][end] / 1e9, df_skeleton[:, "gpu timings"][end] / 1e9
df_skeleton[:, "cpu timings"][end] / df_skeleton[:, "gpu timings"][end]


