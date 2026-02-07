import cv2
import numpy as np

def apply_blur(image: np.ndarray, method: str = 'gaussian', kernel_size: int = 3) -> np.ndarray:
    """
    Applies blur to the image to reduce noise.
    
    Args:
        image: Input image.
        method: 'gaussian' or 'median'.
        kernel_size: Size of the kernel (must be odd).
        
    Returns:
        Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    else:
        return image

def apply_edge_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Enhances edges in the image using unsharp masking.
    Bypasses if image is not valid or empty.
    
    Args:
        image: Input image.
        
    Returns:
        Image with enhanced edges.
    """
    if image is None or image.size == 0:
        return image

    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
    return unsharp_image
