import json
import os
from typing import Dict, Any, Optional

# Default configuration dictionary
DEFAULT_CONFIG: Dict[str, Any] = {
    "clustering": {
        "algorithm": "dbscan",
        "kmeans_k": 3,
        "dbscan_eps_factor": 0.02,
        "dbscan_min_pts": 2,
        "kmeans_max_iter": 10,
        "kmeans_epsilon": 1.0,
        "kmeans_attempts": 10
    },
    "edge_detection": {
        "canny_thresh1": 100,
        "canny_thresh2": 200,
        "blur_size": 3
    },
    "depth_map": {
        "gradient_min": 32,
        "gradient_max": 223
    },
    "anaglyph": {
        "shift_factor": 15
    },
    "output": {
        "default_depth_map_filename": "output/depth_map_output.jpg",
        "default_anaglyph_filename": "output/red_cyan_anaglyph_output.jpg",
        "window_refresh_rate_ms": 100
    }
}

class Config:
    def __init__(self, config_path: str = "config.json"):
        self.config: Dict[str, Any] = DEFAULT_CONFIG
        self.load_config(config_path)

    def load_config(self, path: str) -> None:
        """Loads configuration from a JSON file, overriding defaults."""
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    user_config = json.load(f)
                    self._update_recursive(self.config, user_config)
                print(f"Loaded configuration from {path}")
            except Exception as e:
                print(f"Error loading config file {path}: {e}")
                print("Using default configuration.")
        else:
            print(f"Config file {path} not found. Using defaults.")

    def _update_recursive(self, original: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively updates a dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                self._update_recursive(original[key], value)
            else:
                original[key] = value

    def get(self, section: str, key: Optional[str] = None) -> Any:
        """Retrieves a configuration value."""
        if section not in self.config:
            return None
        if key is None:
            return self.config[section]
        return self.config[section].get(key)

# Global instance
cfg = Config()
