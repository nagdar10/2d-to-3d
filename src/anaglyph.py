
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
