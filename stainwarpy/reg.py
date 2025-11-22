import random
from skimage.transform import resize, estimate_transform, warp, AffineTransform
from skimage.feature import SIFT, match_descriptors
from skimage import measure


def transform_seg_mask(mask, transformation_maps, output_shape):

    moved_mask = warp(mask, transformation_maps.inverse, output_shape=output_shape, order=0, preserve_range=True)

    return moved_mask



def features_with_SIFT(fixed, moving, max_ratio=0.6, n_octaves=3, n_scales=5):
    fixedX, fixedY = fixed.shape
    movingX, movingY = moving.shape
    scale_factor = 4

    if fixedX > 2000 or fixedY > 2000:
        scale_factor = max(fixedX // 2000, fixedY // 2000) * 4

    # Resize the images to reduce memory usage
    fixed_scaled = resize(fixed, (fixedX // scale_factor, fixedY // scale_factor), anti_aliasing=True)
    moving_scaled = resize(moving, (movingX // scale_factor, movingY // scale_factor), anti_aliasing=True)

    descriptor_extractor = SIFT(n_octaves=n_octaves, n_scales=n_scales)

    descriptor_extractor.detect_and_extract(moving_scaled)
    keypoints1, descriptors1 = descriptor_extractor.keypoints, descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(fixed_scaled)
    keypoints2, descriptors2 = descriptor_extractor.keypoints, descriptor_extractor.descriptors

    matches12 = match_descriptors(
        descriptors1, descriptors2, max_ratio=max_ratio, cross_check=True
    )

    if matches12.shape[0] < 3:
        raise ValueError("Not enough matching points found between images for reliable registration.")

    # Extract matched keypoints
    src, dst = keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]]

    dst, src = dst * scale_factor, src * scale_factor

    # Compute inliers using RANSAC 
    _, inliers = measure.ransac((dst, src),
                               AffineTransform, min_samples=4,
                               residual_threshold=2, max_trials=1000)
    
    movingtemp_matches = src[inliers] 
    fixedtemp_matches = dst[inliers] 
    
    moving_matches = movingtemp_matches[:, [1, 0]].copy()
    fixed_matches = fixedtemp_matches[:, [1, 0]].copy()

    return [moving_matches, fixed_matches]



def register_feature_based(fixed, moving, feature_tform):


    [moving_matches, fixed_matches] = features_with_SIFT(fixed, moving)

    num_matches = moving_matches.shape[0]

    if num_matches < 3:
        raise ValueError(f"At least three matching points are required for initial feature based registration, only {num_matches} found.")
    
    num_tre_points = min(6, num_matches - 3, num_matches // 2)
    all_idx = set(range(num_matches))
    tre_idx = random.sample(range(num_matches), num_tre_points)
    other_idx = list(all_idx - set(tre_idx))
    moving_pts_for_reg, fixed_pts_for_reg = moving_matches[other_idx], fixed_matches[other_idx]

    tform = estimate_transform(feature_tform, src=moving_pts_for_reg, dst=fixed_pts_for_reg)
    aligned_moving = warp(moving, tform.inverse, output_shape=fixed.shape)

    return tform, aligned_moving, [moving_matches[tre_idx], fixed_matches[tre_idx]], [moving_pts_for_reg, fixed_pts_for_reg]



def register_DAPI_HnE(fixed, moving, feature_tform='similarity'):

    tform_map, moving_img_aligned, [moving_tre_pts, fixed_tre_pts], [moving_reg_pts, fixed_reg_pts] = register_feature_based(fixed, moving, feature_tform)

    print('Feature based registration completed.')

    return tform_map, moving_img_aligned, [moving_tre_pts, fixed_tre_pts]



    