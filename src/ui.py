import argparse
import tkinter as tk
from tkinter import filedialog
import cv2

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="2D to 3D Image Converter")
    parser.add_argument('--algo', choices=['dbscan', 'kmeans'], default='dbscan', help='Clustering algorithm to use')
    parser.add_argument('--k', type=int, default=3, help='Number of clusters for K-Means')
    parser.add_argument('--eps-factor', type=float, default=0.02, help='Epsilon factor for DBSCAN (proportion of image size)')
    parser.add_argument('--image_path', type=str, help='Path to the input image file')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode, disabling GUI elements')
    return parser.parse_args()

def select_image_file():
    """
    Opens a file dialog to select an image file.
    """
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename()
    return image_path

def save_images(depth_map, red_cyan_image):
    """
    Opens file dialogs to save the depth map and anaglyph image.
    """
    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
        title="Save Depth Map As"
    )
    if file_path:
        cv2.imwrite(file_path, depth_map)
        print(f"Depth map saved to {file_path}")
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
        title="Save Anaglyph Image As"
    )
    if file_path:
        cv2.imwrite(file_path, red_cyan_image)
        print(f"Anaglyph image saved to {file_path}")

def check_windows_open(window_names):
    """
    Checks if any of the specified windows are still open.
    Returns True if at least one window is open, False otherwise.
    """
    for name in window_names:
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
            return True
    return False
