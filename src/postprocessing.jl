function argmax_2ch(pred)
	img1, img2 = pred[:, :, :, 1, :], pred[:, :, :, 2, :]
	rslt = similar(img1)
	for i in CartesianIndices(rslt)
		rslt[i] = img1[i] > img2[i] ? 0 : 1
	end
	return rslt
end

function keep_largest_component(array)
	label = label_components(array)
	for i in eachindex(label)
	if label[i] > 1
			label[i] = 0
		end
	end
	return label
end

function find_edge_idxs(mask)
	edge = erode(mask) .⊻ mask
	return Tuple.(findall(isone, edge))
end

# function get_mask_edges(seg1::AbstractMatrix, seg2::AbstractMatrix, seg3::AbstractMatrix)
# 	seg1, seg2, seg3 = Bool.(seg1), Bool.(seg2), Bool.(seg3)
	
#     # Do binary erosion and use XOR to get edges
#     edges1 = erode(seg1) .⊻ seg1
#     edges2 = erode(seg2) .⊻ seg2
#     edges3 = erode(seg3) .⊻ seg3
	
# 	return findall(isone, edges1), findall(isone, edges2), findall(isone, edges3)
# end