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