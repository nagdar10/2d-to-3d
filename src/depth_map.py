import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import cfg

def get_distinct_contours(image: np.ndarray, 
                          canny_thresh1: Optional[int] = None, 
                          canny_thresh2: Optional[int] = None, 
                          blur_size: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Pre-processes an image to find distinct external contours.

    Args:
        image (np.array): The input image.
        canny_thresh1 (int): First threshold for the Canny edge detector.
        canny_thresh2 (int): Second threshold for the Canny edge detector.
        blur_size (int): The kernel size for the Gaussian blur.

    Returns:
        tuple: A tuple containing the contours and hierarchy.
    """
    if canny_thresh1 is None: canny_thresh1 = cfg.get("edge_detection", "canny_thresh1")
    if canny_thresh2 is None: canny_thresh2 = cfg.get("edge_detection", "canny_thresh2")
    if blur_size is None: blur_size = cfg.get("edge_detection", "blur_size")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (blur_size, blur_size))
    dilated = cv2.dilate(blurred, None)
    canny_output = cv2.Canny(dilated, canny_thresh1, canny_thresh2)
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def get_bottom_point(contour: np.ndarray) -> np.ndarray:
    """Finds the lowest point (max y-coordinate) in a contour."""
    return max(contour, key=lambda point: point[0][1])[0]

def generate_depth_map(contours: List[np.ndarray], labels: List[int], num_clusters: int, width: int, height: int) -> np.ndarray:
    """
    Generates a depth map by clustering contours and assigning depth based on vertical position.
    """
    # Create a background with a vertical linear gradient. This will serve as our depth map.
    # Objects at the top will be "further" (darker), and objects at the bottom
    # will be "closer" (brighter).
    gradient_map = np.zeros((height, width), dtype=np.uint8)
    
    g_min = cfg.get("depth_map", "gradient_min")
    g_max = cfg.get("depth_map", "gradient_max")
    g_diff = g_max - g_min
    
    for r in range(height):
        # Create a gradient from g_min to g_max
        color_val = int((g_diff * r) / height) + g_min
        gradient_map[r, :] = color_val

    # Merge contours that belong to the same cluster
    merged_contours = [[] for _ in range(num_clusters)]
    for i, label in enumerate(labels):
        if label != -1:  # Ignore noise
            merged_contours[label].extend(contours[i])
    
    # Filter out empty cluster lists. Note: This creates a list of arrays.
    # We keep them as arrays for convHull.
    merged_contours = [np.array(mc) for mc in merged_contours if mc]

    # Create the final depth map by drawing the objects onto the gradient
    depth_map = gradient_map.copy()
    for i, contour in enumerate(merged_contours):
        if len(contour) > 0:
            # Determine the object's "color" (depth) by finding its lowest point
            # and sampling the color from the gradient map at that location.
            bottom_point = get_bottom_point(contour)
            # Ensure the point is within bounds
            y = min(bottom_point[1], height - 1)
            depth_color = int(gradient_map[y, 0])
            
            # Draw the convex hull of the merged contour onto the depth map
            hull = cv2.convexHull(contour)
            cv2.drawContours(depth_map, [hull], -1, (depth_color,), thickness=cv2.FILLED)

    return depth_map
