"""
    load_data_and_model(data_dir, dice_path, dice_hd_path; batch_size=4)

#### Inputs
- `data_dir`: path to original dataset
- `dice_path`: path to pure dice loss model
- `dice_hd_path`: path to hybrid dice/hd loss model
- `batch_size`: batch size

#### Returns
- `tdl`: train dataloader
- `vdl`: validation dataloader
- `model_dice`: pure dice loss model
- `model_dice_hd`: hybrid dice/hd loss model
"""
function load_data_and_model(data_dir, dice_path, dice_hd_path; batch_size=4)
    function loadfn_label(p)
        a = NIfTI.niread(string(p)).raw
        convert_a = convert(Array{UInt8}, a)
        convert_a = convert_a .+ 1
        return convert_a
    end
    function loadfn_image(p)
        a = NIfTI.niread(string(p)).raw
        convert_a = convert(Array{Float32}, a)
        convert_a = convert_a / max(convert_a...)
        return convert_a
    end
    function presize(files, image_size)
        container_images = Array{Float32,4}(undef, image_size..., numobs(files))
        container_masks = Array{Int64,4}(undef, image_size..., numobs(files))
        for i in 1:numobs(files)
            image, mask = FastAI.getobs(files, i)
            img = imresize(image, image_size)
            msk = round.(imresize(mask, image_size))
            container_images[:, :, :, i] = img
            container_masks[:, :, :, i] = msk
        end
        return container_images, container_masks
    end

    @info "Starting"
    task, model_dice = loadtaskmodel(dice_path)
    _, model_dice_hd = loadtaskmodel(dice_hd_path)
    @info "Model loaded"

    images(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
    masks(dir) = mapobs(loadfn_label, Glob.glob("*.nii*", dir))
    pre_data = (
        images(joinpath(data_dir, "imagesTr")),
        masks(joinpath(data_dir, "labelsTr")),
    )
    @info "Data loaded"

    image_size = (96, 96, 96)
    img_container, mask_container = presize(pre_data, image_size)
    data_resized = (img_container, mask_container)
    @info "Data Prepared"

    a, b = FastVision.imagedatasetstats(img_container, Gray{N0f8})
    means, stds = SVector{1,Float32}(a[1]), SVector{1,Float32}(b[1])
    train_files, val_files = MLDataPattern.splitobs(data_resized, 0.8)
    tdl, vdl = FastAI.taskdataloaders(train_files, val_files, task, batch_size)
    @info "Dataloader Created --> Finished!"

    return tdl, vdl, model_dice, model_dice_hd
end

function loadfn_label(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{UInt8}, a)
    convert_a = convert_a .+ 1
    return convert_a
end

function loadfn_image(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{Float32}, a)
    convert_a = convert_a / max(convert_a...)
    return convert_a
end

function presize(files)
    container_images = Array{Float32,4}(undef, image_size..., numobs(files))
    container_masks = Array{Int64,4}(undef, image_size..., numobs(files))
    for i in 1:numobs(files)
        image, mask = FastAI.getobs(files, i)
        img = imresize(image, image_size)
        msk = round.(imresize(mask, image_size))
        container_images[:, :, :, i] = img
        container_masks[:, :, :, i] = msk
    end
    return container_images, container_masks
end