import unittest
import os
import sys
from unittest.mock import patch

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import main

class TestMain(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        self.test_image_path = 'images/f1.jpg'
        self.depth_map_output = 'output/depth_map_output.jpg'
        self.anaglyph_output = 'output/red_cyan_anaglyph_output.jpg'
        # Ensure the output directory exists
        os.makedirs('output', exist_ok=True)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        for file_path in [self.depth_map_output, self.anaglyph_output]:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_main_e2e(self):
        """
        Test the main function in an end-to-end fashion.
        It runs the script with a test image and checks if the output files are created.
        """
        # We patch sys.argv to simulate command-line arguments
        with patch('sys.argv', ['src/main.py', '--image_path', self.test_image_path, '--test_mode']):
            main()
        
        # Check if the output files were created
        self.assertTrue(os.path.exists(self.depth_map_output), f"{self.depth_map_output} was not created.")
        self.assertTrue(os.path.exists(self.anaglyph_output), f"{self.anaglyph_output} was not created.")

if __name__ == '__main__':
    unittest.main()
