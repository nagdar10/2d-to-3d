import cv2
import numpy as np
import sys
import math

class DbScan:
    """
    A Python implementation of the DBSCAN clustering algorithm tailored for clustering
    bounding boxes (cv2.Rect). It groups rectangles that are close to each other.
    """
    def __init__(self, data, eps, min_pts):
        """
        Initializes the DBSCAN algorithm.

        Args:
            data (list): A list of rectangles [x, y, w, h] to cluster.
            eps (float): The maximum distance between two samples for one to be 
                         considered as in the neighborhood of the other.
            min_pts (int): The number of samples in a neighborhood for a point 
                           to be considered as a core point.
        """
        self.data = data
        self.eps = eps
        self.min_pts = min_pts
        self.labels = [-99] * len(data)  # -99: unvisited, -1: noise
        self.cluster_id = -1
        # Memoization table for distances to avoid re-computation
        self.dist_cache = np.full((len(data), len(data)), -1.0)

    def run(self):
        """
        Executes the DBSCAN clustering algorithm.
        """
        for i in range(len(self.data)):
            if not self.is_visited(i):
                neighbors = self.region_query(i)
                if len(neighbors) < self.min_pts:
                    self.labels[i] = -1  # Mark as noise
                else:
                    self.cluster_id += 1
                    self.expand_cluster(i, neighbors)
        return self.labels

    def expand_cluster(self, p_index, neighbors):
        """
        Expands a cluster from a core point.

        Args:
            p_index (int): The index of the core point.
            neighbors (list): The list of neighbors of the core point.
        """
        self.labels[p_index] = self.cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_index = neighbors[i]
            if not self.is_visited(neighbor_index):
                self.labels[neighbor_index] = self.cluster_id
                new_neighbors = self.region_query(neighbor_index)
                if len(new_neighbors) >= self.min_pts:
                    # Add new neighbors to the list to be processed
                    neighbors.extend(n for n in new_neighbors if n not in neighbors)
            i += 1

    def is_visited(self, index):
        """Checks if a point has been visited (i.e., assigned a cluster or noise)."""
        return self.labels[index] != -99

    def region_query(self, p_index):
        """
        Finds all points within the epsilon distance of a given point.

        Args:
            p_index (int): The index of the point to query around.

        Returns:
            list: A list of indices of neighboring points.
        """
        neighbors = []
        for i in range(len(self.data)):
            if self.distance_func(p_index, i) <= self.eps:
                neighbors.append(i)
        return neighbors

    def distance_func(self, i, j):
        """
        Calculates the minimum distance between the corners of two rectangles.
        Uses a cache to store and retrieve already computed distances.
        """
        if self.dist_cache[i, j] != -1:
            return self.dist_cache[i, j]
        if i == j:
            self.dist_cache[i, j] = 0.0
            return 0.0

        rect_a = self.data[i]
        rect_b = self.data[j]

        corners_a = [
            (rect_a[0], rect_a[1]),
            (rect_a[0] + rect_a[2], rect_a[1]),
            (rect_a[0], rect_a[1] + rect_a[3]),
            (rect_a[0] + rect_a[2], rect_a[1] + rect_a[3]),
        ]
        corners_b = [
            (rect_b[0], rect_b[1]),
            (rect_b[0] + rect_b[2], rect_b[1]),
            (rect_b[0], rect_b[1] + rect_b[3]),
            (rect_b[0] + rect_b[2], rect_b[1] + rect_b[3]),
        ]

        min_dist = float('inf')
        for p1 in corners_a:
            for p2 in corners_b:
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < min_dist:
                    min_dist = dist
        
        self.dist_cache[i, j] = min_dist
        self.dist_cache[j, i] = min_dist
        return min_dist

def get_distinct_contours(image, canny_thresh1=100, canny_thresh2=200, blur_size=3):
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (blur_size, blur_size))
    dilated = cv2.dilate(blurred, None)
    canny_output = cv2.Canny(dilated, canny_thresh1, canny_thresh2)
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def get_bottom_point(contour):
    """Finds the lowest point (max y-coordinate) in a contour."""
    return max(contour, key=lambda point: point[0][1])[0]

def generate_red_cyan(image, depth_map):
    """
    Generates a red-cyan 3D anaglyph image from an image and its depth map.

    Args:
        image (np.array): The original color image.
        depth_map (np.array): A grayscale depth map (darker is further).

    Returns:
        np.array: The resulting red-cyan anaglyph image.
    """
    # Start with a copy of the original image
    anaglyph_image = image.copy()
    rows, cols, _ = image.shape

    # Shift the blue and green channels to the right based on the depth map
    for i in range(rows):
        for j in range(cols):
            # Calculate pixel shift amount (m) based on depth
            depth_val = depth_map[i, j]
            m = int((15 * depth_val) / 255)
            
            if j < cols - m:
                # For the current pixel (i, j), get the blue and green values
                # from a pixel to the right (i, j + m).
                # This creates the cyan channel for the right eye's view.
                blue_val = image[i, j + m, 0]
                green_val = image[i, j + m, 1]
                anaglyph_image[i, j, 0] = blue_val
                anaglyph_image[i, j, 1] = green_val
                # The red channel at (i, j) remains unchanged, representing
                # the left eye's view.

    return anaglyph_image

def main():
    """
    Main execution function.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        return -1

    image_path = sys.argv[1]
    im = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if im is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return -1

    height, width, _ = im.shape

    # 1. Find initial contours in the image
    contours, _ = get_distinct_contours(im)

    # 2. Get the bounding box for each contour
    boxes = [cv2.boundingRect(c) for c in contours]

    # 3. Cluster the bounding boxes using DBSCAN to group parts of the same object
    # The distance threshold is proportional to the image size.
    dbscan_distance = ((height + width) / 2) * 0.02
    dbscan = DbScan(boxes, dbscan_distance, 2)
    labels = dbscan.run()
    num_clusters = dbscan.cluster_id + 1

    # 4. Create a background with a vertical linear gradient. This will serve as our depth map.
    # Objects at the top will be "further" (darker), and objects at the bottom
    # will be "closer" (brighter).
    gradient_map = np.zeros((height, width), dtype=np.uint8)
    for r in range(height):
        # Create a gradient from 32 to 223 (191+32)
        color_val = int((191 * r) / height) + 32
        gradient_map[r, :] = color_val

    # 5. Merge contours that belong to the same cluster
    merged_contours = [[] for _ in range(num_clusters)]
    for i, label in enumerate(labels):
        if label != -1:  # Ignore noise
            merged_contours[label].extend(contours[i])
    
    # Filter out empty cluster lists
    merged_contours = [np.array(mc) for mc in merged_contours if mc]

    # 6. Create the final depth map by drawing the objects onto the gradient
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

    # 7. Generate the red-cyan anaglyph image
    red_cyan_image = generate_red_cyan(im, depth_map)

    # 8. Display the results
    cv2.imshow("Original Image", im)
    cv2.imshow("Generated Depth Map", depth_map)
    cv2.imshow("Red-Cyan 3D Anaglyph", red_cyan_image)
    
    # Optional: Save the output
    cv2.imwrite("depth_map_output.jpg", depth_map)
    cv2.imwrite("red_cyan_anaglyph_output.jpg", red_cyan_image)
    print("Output images 'depth_map_output.jpg' and 'red_cyan_anaglyph_output.jpg' have been saved.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
