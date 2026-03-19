import cv2
import numpy as np
import json

def load_points_from_json(json_path):
    """Expects JSON format: {'left': [[x,y], ...], 'right': [[x,y], ...]}"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['right'], dtype=np.float32), np.array(data['left'], dtype=np.float32)

def laplacian_blend(img1, img2, mask):
    """Blends two images using a 5-level Laplacian Pyramid."""
    # Ensure dimensions are divisible by 2^5 (32)
    h, w = img1.shape[:2]
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Build Gaussian Pyramids
    gp1, gp2, gpM = [img1], [img2], [mask]
    for i in range(5):
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        mask = cv2.pyrDown(mask)
        gp1.append(img1)
        gp2.append(img2)
        gpM.append(mask)

    # Build Laplacian Pyramids
    lp1 = [gp1[5]]
    lp2 = [gp2[5]]
    for i in range(5, 0, -1):
        L1 = cv2.subtract(gp1[i-1], cv2.pyrUp(gp1[i]))
        L2 = cv2.subtract(gp2[i-1], cv2.pyrUp(gp2[i]))
        lp1.append(L1)
        lp2.append(L2)

    # Blend and Reconstruct
    LS = []
    for l1, l2, m in zip(lp1, lp2, gpM[::-1]):
        LS.append(l1 * m + l2 * (1.0 - m))

    res = LS[0]
    for i in range(1, 6):
        res = cv2.add(cv2.pyrUp(res), LS[i])
    
    return np.clip(res, 0, 255).astype(np.uint8)

def stitch_sensors(left_path, right_path, json_path):
    img_l = cv2.imread(left_path)
    img_r = cv2.imread(right_path)
    pts_r, pts_l = load_points_from_json(json_path)

    # 1. Calculate Rigid Transformation (Rotation + Translation + Scale)
    # This is more robust for deep racks than cv2.findHomography
    matrix, _ = cv2.estimateAffinePartial2D(pts_r, pts_l)

    # 2. Prepare Canvas (Double the width for stitching)
    h, w = img_l.shape[:2]
    canvas_w = w * 2 
    warped_r = cv2.warpAffine(img_r, matrix, (canvas_w, h))
    
    # Place left image on the canvas
    canvas_l = np.zeros((h, canvas_w, 3), dtype=np.uint8)
    canvas_l[:, :w] = img_l

    # 3. Create Gradient Mask for the Overlap
    # We find where both images have content to define the overlap
    mask_l = (np.sum(canvas_l, axis=2) > 0).astype(np.float32)
    mask_r = (np.sum(warped_r, axis=2) > 0).astype(np.float32)
    overlap = (mask_l * mask_r)
    
    # Generate the smooth transition inside the overlap region
    final_mask = mask_l.copy()
    overlap_indices = np.where(overlap[h//2, :] > 0)[0]
    if len(overlap_indices) > 0:
        start, end = overlap_indices[0], overlap_indices[-1]
        alpha_gradient = np.linspace(1, 0, end - start + 1)
        for i, col in enumerate(range(start, end + 1)):
            final_mask[:, col] = alpha_gradient[i]

    # Convert mask to 3 channels
    final_mask_3ch = cv2.merge([final_mask]*3)

    # 4. Perform the Multi-band Blend
    return laplacian_blend(canvas_l, warped_r, final_mask_3ch)

# Usage
# result = stitch_sensors('left.jpg', 'right.jpg', 'points.json')
# cv2.imwrite('stitched_rack.jpg', result)
