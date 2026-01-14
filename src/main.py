import cv2
import numpy as np
from typing import Optional
from clustering import DbScan, KMeansClustering
from depth_map import get_distinct_contours, generate_depth_map
from anaglyph import generate_red_cyan
import ui
from config import cfg

def main() -> Optional[int]:
    """
    Main execution function.
    """
    args = ui.parse_arguments()

    if args.image_path:
        image_path = args.image_path
    else:
        # Use UI if no path provided
        image_path = ui.select_image_file()

    if not image_path:
        print("No image selected.")
        return

    im = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if im is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return -1

    height, width, _ = im.shape

    # 1. Find initial contours in the image
    contours, _ = get_distinct_contours(im)

    # 2. Get the bounding box for each contour
    boxes = [cv2.boundingRect(c) for c in contours]

    # 3. Cluster the bounding boxes
    labels = []
    num_clusters = 0

    if args.algo == 'kmeans':
        print(f"Using K-Means clustering with k={args.k}")
        kmeans = KMeansClustering(boxes, args.k)
        labels = kmeans.run()
        num_clusters = kmeans.cluster_id + 1
    else:
        # Default to DBSCAN
        # The distance threshold is proportional to the image size.
        dbscan_distance = ((height + width) / 2) * args.eps_factor
        print(f"Using DBSCAN clustering with eps={dbscan_distance:.2f} (factor={args.eps_factor})")
        
        min_pts = cfg.get("clustering", "dbscan_min_pts")
        dbscan = DbScan(boxes, dbscan_distance, min_pts)
        labels = dbscan.run()
        num_clusters = dbscan.cluster_id + 1

    # 4-6. Generate Depth Map
    depth_map = generate_depth_map(contours, labels, num_clusters, width, height)

    # 7. Generate the red-cyan anaglyph image
    red_cyan_image = generate_red_cyan(im, depth_map)

    # Output defaults
    depth_out = cfg.get("output", "default_depth_map_filename")
    anaglyph_out = cfg.get("output", "default_anaglyph_filename")
    
    cv2.imwrite(depth_out, depth_map)
    cv2.imwrite(anaglyph_out, red_cyan_image)
    print(f"Output images '{depth_out}' and '{anaglyph_out}' have been saved.")

    # 8. Display the results if not in test mode
    if not args.test_mode:
        window_names = ["Original Image", "Generated Depth Map", "Red-Cyan 3D Anaglyph"]
        cv2.imshow(window_names[0], im)
        cv2.imshow(window_names[1], depth_map)
        cv2.imshow(window_names[2], red_cyan_image)
        
        print("Press 's' to save images, 'q' or Esc to quit, or close all windows.")
        refresh_rate = cfg.get("output", "window_refresh_rate_ms")
        while True:
            # Wait for key press
            key = cv2.waitKey(refresh_rate) & 0xFF
            
            # If 'q' or Esc (27) is pressed, break
            if key == ord('q') or key == 27:
                break
            
            # If 's' is pressed, save the images
            if key == ord('s'):
                ui.save_images(depth_map, red_cyan_image)
                
            # Check if windows are still open
            if not ui.check_windows_open(window_names):
                break
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
