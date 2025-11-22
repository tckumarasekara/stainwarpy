import numpy as np
from scipy.stats import entropy

def compute_TRE(tform, tre_points, fixed):

    src_points, dst_points = tre_points
    tre = {}
    h, w = fixed.shape   
    diagonal = np.sqrt(h**2 + w**2)

    if len(src_points) != len(dst_points):
        raise ValueError("Same number of source and destination points must be provided.")
    
    if len(src_points) < 3:
        raise ValueError("At least three points are required to compute TRE.")
    
    tre_temp = np.mean(np.linalg.norm(np.array(src_points) - np.array(dst_points), axis=1))
    tre_temp /= diagonal    
    tre['before registration'] = tre_temp
    print("rTRE before registration: ", tre_temp)
    
    transformed_src = np.array(tform(src_points), dtype=float)
    tre_temp = np.mean(np.linalg.norm(transformed_src - dst_points, axis=1))
    tre_temp /= diagonal
    tre['after feature based'] = tre_temp
    print("rTRE after feature based registration: ", tre_temp)

    return tre



def mutual_information_metric(fixed, moving, bins):
    fy, fx = fixed.shape
    my, mx = moving.shape

    if fy != my or fx != mx:
        min_y = min(fy, my)
        min_x = min(fx, mx)

        fixed = fixed[:min_y, :min_x]
        moving = moving[:min_y, :min_x]

    # Compute joint histogram
    hist_2d, _, _ = np.histogram2d(fixed.ravel(), moving.ravel(), bins=bins)
    
    # Normalize to get joint probabilities
    pxy = hist_2d / np.sum(hist_2d)
    
    # Marginal probabilities
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # Entropies
    Hx = entropy(px)
    Hy = entropy(py)
    Hxy = entropy(pxy.ravel())
    
    # Mutual Information
    mutual_info = Hx + Hy - Hxy
    n_mutual_info = mutual_info / np.mean([Hx, Hy]) 
    
    return n_mutual_info



def compute_mutual_information(fixed, moving, tform_img, bins = 50):
    mi_scores = {}
    
    mi_before = mutual_information_metric(fixed, moving, bins)
    mi_scores['before registration'] = mi_before
    print("normalized MI before registration: ", mi_before)

    mi_after_init = mutual_information_metric(fixed, tform_img, bins)
    mi_scores['after feature based'] = mi_after_init
    print("normalized MI after feature based registration: ", mi_after_init)

    return mi_scores