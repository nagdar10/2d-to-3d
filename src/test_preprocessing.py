import unittest
import numpy as np
import cv2
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a dummy image (100x100 grayscale) with some noise
        self.image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(self.image, (20, 20), (80, 80), 255, -1)
        # Add random noise
        noise = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
        self.noisy_image = cv2.add(self.image, noise)

    def test_apply_blur_gaussian(self):
        blurred = preprocessing.apply_blur(self.noisy_image, method='gaussian', kernel_size=3)
        self.assertEqual(blurred.shape, self.noisy_image.shape)
        # Blurring should reduce variance in homogeneous regions, but exact check is tricky.
        # Check if it runs without error and returns valid image.
        self.assertIsNotNone(blurred)

    def test_apply_blur_median(self):
        blurred = preprocessing.apply_blur(self.noisy_image, method='median', kernel_size=3)
        self.assertEqual(blurred.shape, self.noisy_image.shape)
        self.assertIsNotNone(blurred)

    def test_apply_edge_enhancement(self):
        enhanced = preprocessing.apply_edge_enhancement(self.image)
        self.assertEqual(enhanced.shape, self.image.shape)
        self.assertIsNotNone(enhanced)
        # Check that edges are actually different (sharpened)
        # In a simple rectangle, the corners might have different values.
        # But here we just want to ensure it runs and modifies the image potentially.
        # If the image is flat 0, enhancement might do nothing.
        # But our image has a rectangle.
        self.assertFalse(np.array_equal(enhanced, self.image), "Enhanced image should check differ from original")

    def test_invalid_input(self):
        res = preprocessing.apply_edge_enhancement(None)
        self.assertIsNone(res)
        
if __name__ == '__main__':
    unittest.main()
