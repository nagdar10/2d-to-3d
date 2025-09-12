# TODO

This file contains a list of potential improvements and features for the 2D to 3D conversion project.

## Core Functionality

- [ ] **3D Model Generation:** Use the generated depth map to create a 3D model, such as a point cloud or a mesh.
- [ ] **3D Visualization:** Implement a 3D viewer to display the generated 3D model.
- [ ] **Save/Export:** Allow the user to save the generated depth map and/or the 3D model to a file.

## User Interface

- [ ] **Graphical User Interface (GUI):** Create a GUI to make the application more user-friendly.
- [ ] **File Dialog:** Implement a file dialog to allow the user to select an image file instead of using a hardcoded filename.
- [ ] **Parameter Tuning:** Add controls to the GUI to allow the user to tune the DBSCAN parameters (`eps` and `min_samples`) and other settings.

## Algorithm Improvements

- [ ] **Alternative Clustering Algorithms:** Explore other clustering algorithms (e.g., K-Means, Mean Shift) to see if they produce better results.
- [ ] **Image Preprocessing:** Add more advanced image preprocessing steps to improve the quality of the depth map (e.g., noise reduction, edge detection).
- [ ] **Refine Depth Map Generation:** Improve the logic for assigning depth values to clusters to create a more realistic depth map.
