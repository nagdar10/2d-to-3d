# TODO

This file contains a list of potential improvements and features for the 2D to 3D conversion project.

## Code improvements

- [x] **Tests:** Add e2e or some kind of tests to compare or score the results.
- [x] **Code documentation:** Improve readme with run instructions and details
- [x] **Code modularization:** Split main.py into separate modules (clustering.py, depth_map.py, anaglyph.py, ui.py)
- [x] **Type hints:** Add proper type annotations throughout the codebase
- [ ] **Error handling:** Add try-catch blocks for file operations and image processing
- [x] **Configuration management:** Create a config file for default parameters
- [ ] **Unit tests:** Test clustering algorithms and depth map generation functions
- [ ] **Integration tests:** Test end-to-end workflow with sample images
- [ ] **Performance benchmarks:** Measure processing time for different image sizes

## Core Functionality

- [ ] **3D Model Generation:** Use the generated depth map to create a 3D model, such as a point cloud or a mesh.
- [ ] **3D Visualization:** Implement a 3D viewer to display the generated 3D model.
- [x] **Save/Export:** Allow the user to save the generated depth map and/or the 3D model to a file.
- [ ] **Multiple output formats:** Support PNG, TIFF for better quality and 16-bit depth maps
- [ ] **3D format export:** Export to point cloud (PLY) or mesh (OBJ) formats
- [ ] **Batch processing:** Handle multiple images at once

## User Interface

- [ ] **Graphical User Interface (GUI):** Create a GUI to make the application more user-friendly.
- [x] **File Dialog:** Implement a file dialog to allow the user to select an image file instead of using a hardcoded filename.
- [x] **Parameter Tuning:** Added controls to the GUI to allow the user to tune the DBSCAN parameters (`eps` and `min_samples`) and other settings.
- [x] **Issue with closing** When you close all image windows, the program does not terminate. (Also now closes on any window close)
- [ ] **Progress indicators:** Show processing status for large images
- [x] **Preview mode:** Allow parameter adjustment before final processing (integrated with parameter tuning)

## Algorithm Improvements

- [x] **Alternative Clustering Algorithms:** Implemented K-Means clustering. (Original: Explore other clustering algorithms (e.g., K-Means, Mean Shift) to see if they produce better results.)
- [ ] **Image Preprocessing:** Add more advanced image preprocessing steps to improve the quality of the depth map (e.g., noise reduction, edge detection).
- [ ] **Refine Depth Map Generation:** Improve the logic for assigning depth values to clusters to create a more realistic depth map.

## Documentation

- [ ] **API documentation:** Generate docs from code comments
- [ ] **Examples gallery:** Showcase before/after results with sample images
- [ ] **Algorithm explanation:** Detail the depth generation process in documentation
- [ ] **Adaptive parameters:** Auto-tune clustering parameters based on image characteristics
- [ ] **Multiple depth cues:** Incorporate additional depth indicators (shading, texture gradients)
- [ ] **Post-processing:** Add smoothing filters to reduce artifacts in depth maps
