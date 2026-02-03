import argparse
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from typing import List
from config import cfg

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    defaults = cfg.get("clustering")
    
    parser = argparse.ArgumentParser(description="2D to 3D Image Converter")
    parser.add_argument('--algo', choices=['dbscan', 'kmeans'], default=defaults['algorithm'], help='Clustering algo to use')
    parser.add_argument('--k', type=int, default=defaults['kmeans_k'], help='Number of clusters for K-Means')
    parser.add_argument('--eps-factor', type=float, default=defaults['dbscan_eps_factor'], help='Epsilon factor for DBSCAN (proportion of image size)')
    parser.add_argument('--image_path', type=str, help='Path to the input image file')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode, disabling GUI elements')
    return parser.parse_args()

def select_image_file() -> str:
    """
    Opens a file dialog to select an image file.
    """
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename()
    return image_path

def prompt_save_paths() -> tuple[str, str]:
    """
    Opens file dialogs to select paths for saving the depth map and anaglyph image.
    Returns:
        tuple: (depth_map_path, anaglyph_path) - Empty string if cancelled.
    """
    filetypes = [
        ("JPEG files", "*.jpg"), 
        ("PNG files", "*.png"), 
        ("TIFF files", "*.tiff;*.tif"),
        ("All files", "*.*")
    ]
    
    depth_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=filetypes,
        title="Save Depth Map As"
    )
    
    if not depth_path:
        return "", ""
        
    anaglyph_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=filetypes,
        title="Save Anaglyph Image As"
    )
    
    return depth_path, anaglyph_path

def check_windows_open(window_names: List[str]) -> bool:
    """
    Checks if any of the specified windows are still open.
    Returns True if at least one window is open, False otherwise.
    """
    for name in window_names:
        # Check if the window is closed (property returns -1)
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            return False
    return True

class ParameterTuner:
    """
    Manages trackbars for tuning clustering parameters.
    """
    def __init__(self, window_name: str, initial_params: dict, on_change_callback):
        self.window_name = window_name
        self.params = initial_params.copy()
        self.callback = on_change_callback
        
        # Algorithm: 0=DBSCAN, 1=KMeans
        self.algo_map = {0: 'dbscan', 1: 'kmeans'}
        inv_algo_map = {v: k for k, v in self.algo_map.items()}
        
        # Set up trackbars
        cv2.createTrackbar('Algorithm (0:DBSCAN, 1:KMeans)', window_name, inv_algo_map[self.params['algo']], 1, self._on_algo_change)
        
        # DBSCAN params
        # eps_factor: 1-100 (represents 0.01-1.00)
        eps_int = int(self.params.get('eps_factor', 0.1) * 100)
        cv2.createTrackbar('DBSCAN Eps (x0.01)', window_name, eps_int, 100, self._on_eps_change)
        
        # min_samples: 1-20
        min_pts = self.params.get('min_pts', 5)
        cv2.createTrackbar('DBSCAN Min Pts', window_name, min_pts, 20, self._on_min_pts_change)
        
        # KMeans params
        # k: 1-20
        k = self.params.get('k', 3)
        cv2.createTrackbar('K-Means K', window_name, k, 20, self._on_k_change)

    def _on_algo_change(self, val):
        self.params['algo'] = self.algo_map.get(val, 'dbscan')
        self.callback(self.params)

    def _on_eps_change(self, val):
        if val < 1: val = 1
        self.params['eps_factor'] = val / 100.0
        self.callback(self.params)

    def _on_min_pts_change(self, val):
        if val < 1: val = 1
        self.params['min_pts'] = val
        self.callback(self.params)

    def _on_k_change(self, val):
        if val < 1: val = 1
        self.params['k'] = val
        self.callback(self.params)

def create_tuning_controls(window_name: str, initial_params: dict, on_change_callback) -> ParameterTuner:
    """
    Creates trackbars on the specified window for parameter tuning.
    """
    return ParameterTuner(window_name, initial_params, on_change_callback)
