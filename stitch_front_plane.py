def stitch_front_plane(left_path, right_path, json_path):
    img_l = cv2.imread(left_path)
    img_r = cv2.imread(right_path)
    pts_r, pts_l = load_points_from_json(json_path)

    # 1. Use Homography for "Planar" alignment
    # RANSAC helps ignore any points that might have been clicked slightly off-plane
    H, mask = cv2.findHomography(pts_r, pts_l, cv2.RANSAC, 5.0)

    # 2. Define Canvas
    h, w = img_l.shape[:2]
    canvas_w = w * 2
    
    # Warp Image B (Right) onto the coordinate system of Image A (Left)
    # This maps the "Front Plane" of B to the "Front Plane" of A
    warped_r = cv2.warpPerspective(img_r, H, (canvas_w, h))
    
    # Prepare Image A on the same sized canvas
    canvas_l = np.zeros((h, canvas_w, 3), dtype=np.uint8)
    canvas_l[:, :w] = img_l

    # 3. Create the Overlap Mask
    # (Same mask logic from the previous code block goes here)
    # ...
    
    # 4. Blend
    return laplacian_blend(canvas_l, warped_r, final_mask_3ch)
