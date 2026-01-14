import numpy as np
import math
import cv2
from typing import List, Tuple
from config import cfg

class DbScan:
    """
    A Python implementation of the DBSCAN clustering algorithm tailored for clustering
    bounding boxes (cv2.Rect). It groups rectangles that are close to each other.
    """
    def __init__(self, data: List[Tuple[int, int, int, int]], eps: float, min_pts: int):
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
        self.labels: List[int] = [-99] * len(data)  # -99: unvisited, -1: noise
        self.cluster_id = -1
        # Memoization table for distances to avoid re-computation
        self.dist_cache: np.ndarray = np.full((len(data), len(data)), -1.0)

    def run(self) -> List[int]:
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

    def expand_cluster(self, p_index: int, neighbors: List[int]):
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

    def is_visited(self, index: int) -> bool:
        """Checks if a point has been visited (i.e., assigned a cluster or noise)."""
        return self.labels[index] != -99

    def region_query(self, p_index: int) -> List[int]:
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

    def distance_func(self, i: int, j: int) -> float:
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

class KMeansClustering:
    """
    K-Means clustering implementation for bounding boxes.
    Clusters boxes based on their centroids.
    """
    def __init__(self, data: List[Tuple[int, int, int, int]], k: int):
        """
        Initializes K-Means clustering.

        Args:
            data (list): List of rectangles [x, y, w, h].
            k (int): Number of clusters.
        """
        self.data = data
        self.k = k
        self.labels: List[int] = []
        self.cluster_id = -1

    def run(self) -> List[int]:
        """
        Executes K-Means clustering.
        """
        if not self.data:
            return []

        # Convert rectangles to centroids for K-Means
        points = []
        for rect in self.data:
            x, y, w, h = rect
            cx = x + w / 2.0
            cy = y + h / 2.0
            points.append([cx, cy])
        
        points_np = np.array(points, dtype=np.float32)
        
        # Ensure k is not greater than the number of samples
        real_k = min(len(self.data), self.k)
        if real_k < 1:
            return [-1] * len(self.data)

        # Define criteria = ( type, max_iter = 10, epsilon = 1.0 )
        max_iter = cfg.get("clustering", "kmeans_max_iter")
        epsilon = cfg.get("clustering", "kmeans_epsilon")
        attempts = cfg.get("clustering", "kmeans_attempts")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
        
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(points_np, real_k, None, criteria, attempts, flags)
        
        self.labels = labels.flatten().tolist()
        self.cluster_id = real_k - 1
        return self.labels
