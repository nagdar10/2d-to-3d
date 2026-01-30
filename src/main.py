import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from clustering import DbScan, KMeansClustering
from depth_map import get_distinct_contours, generate_depth_map
from anaglyph import generate_red_cyan
import ui
from config import cfg

def run_clustering(boxes: List[Tuple[int, int, int, int]], 
                   params: Dict[str, Any], 
                   width: int, height: int) -> Tuple[List[int], int]:
    """
    Runs the selected clustering algorithm based on params.
    """
    algo = params.get('algo', 'dbscan')
    
    if algo == 'kmeans':
        k = params.get('k', 3)
        # Avoid spamming console during tuning
        # print(f"Using K-Means clustering with k={k}")
        kmeans = KMeansClustering(boxes, k)
        labels = kmeans.run()
        num_clusters = kmeans.cluster_id + 1
    else:
        # Default to DBSCAN
        eps_factor = params.get('eps_factor', 0.1)
        min_pts = params.get('min_pts', cfg.get("clustering", "dbscan_min_pts"))
        
        # The distance threshold is proportional to the image size.
        dbscan_distance = ((height + width) / 2) * eps_factor
        # print(f"Using DBSCAN clustering with eps={dbscan_distance:.2f} (factor={eps_factor})")
        
        dbscan = DbScan(boxes, dbscan_distance, min_pts)
        labels = dbscan.run()
        num_clusters = dbscan.cluster_id + 1
        
    return labels, num_clusters

def perform_processing(im: np.ndarray, 
                       boxes: List[Tuple[int, int, int, int]], 
                       contours: List[np.ndarray], 
                       params: Dict[str, Any], 
                       width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes the full pipeline: clustering -> depth map -> anaglyph.
    """
    labels, num_clusters = run_clustering(boxes, params, width, height)
    
    # Generate Depth Map
    depth_map = generate_depth_map(contours, labels, num_clusters, width, height)

    # Generate the red-cyan anaglyph image
    red_cyan_image = generate_red_cyan(im, depth_map)
    
    return depth_map, red_cyan_image

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

    # Initial parameters
    current_params = {
        'algo': args.algo,
        'k': args.k,
        'eps_factor': args.eps_factor,
        'min_pts': cfg.get("clustering", "dbscan_min_pts")
    }
    
    # Store images to save later
    current_depth_map = None
    current_red_cyan = None

    window_names = ["Original Image", "Generated Depth Map", "Red-Cyan 3D Anaglyph"]

    # Callback function for parameter updates
    def update_callback(params):
        nonlocal current_depth_map, current_red_cyan
        # print(f"Updating with params: {params}") # Debug
        
        try:
            current_depth_map, current_red_cyan = perform_processing(
                im, boxes, contours, params, width, height
            )
            
            # Update display
            cv2.imshow(window_names[1], current_depth_map)
            cv2.imshow(window_names[2], current_red_cyan)
        except Exception as e:
            print(f"Error processing: {e}")

    # Initial processing
    # print("Running initial processing...")
    current_depth_map, current_red_cyan = perform_processing(
        im, boxes, contours, current_params, width, height
    )
    
    # Save default output as requested by original logic
    depth_out = cfg.get("output", "default_depth_map_filename")
    anaglyph_out = cfg.get("output", "default_anaglyph_filename")
    cv2.imwrite(depth_out, current_depth_map)
    cv2.imwrite(anaglyph_out, current_red_cyan)
    print(f"Initial output images '{depth_out}' and '{anaglyph_out}' have been saved.")

    # 8. Display the results if not in test mode
    if not args.test_mode:
        cv2.namedWindow(window_names[0]) # Create window first
        cv2.imshow(window_names[0], im)
        cv2.imshow(window_names[1], current_depth_map)
        cv2.imshow(window_names[2], current_red_cyan)
        
        # Attach parameter tuning controls to the main window
        tuner = ui.create_tuning_controls(window_names[0], current_params, update_callback)
        
        print("Press 's' to save current result, 'q' or Esc to quit.")
        refresh_rate = cfg.get("output", "window_refresh_rate_ms")
        
        while True:
            # Wait for key press
            key = cv2.waitKey(refresh_rate) & 0xFF
            
            # If 'q' or Esc (27) is pressed, break
            if key == ord('q') or key == 27:
                break
            
            # If 's' is pressed, save the images
            if key == ord('s'):
                ui.save_images(current_depth_map, current_red_cyan)
                
            # Check if windows are still open
            if not ui.check_windows_open(window_names):
                break
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
