import unittest
from unittest.mock import patch

from math import sqrt
import webcolors

def color_distance_from(from_color, to_rgb_tuple):
    if from_color == 'white':
        green_diff = 255 - to_rgb_tuple[0]
        blue_diff = 255 - to_rgb_tuple[1]
        red_diff = 255 - to_rgb_tuple[2]
        color_distance = sqrt(green_diff**2 + blue_diff**2 + red_diff**2)
    elif from_color == 'black':
        color_distance = sqrt(to_rgb_tuple[0]**2 + to_rgb_tuple[1]**2 + to_rgb_tuple[2]**2)
    else:
        rbg_tuple = tuple(webcolors.hex_to_rgb(from_color))
        green_diff = rbg_tuple[0] - to_rgb_tuple[0]
        blue_diff = rbg_tuple[1] - to_rgb_tuple[1]
        red_diff = rbg_tuple[2] - to_rgb_tuple[2]
        color_distance = sqrt(green_diff**2 + blue_diff**2 + red_diff**2)

    return color_distance

def get_text_color(text_color='white', bar_color_rgb=(0, 0, 0)):
    if bar_color_rgb != (0, 0, 0):
        text_colors_list = []
        for color in ['white', '#404040', 'black']:
            color_tuple = (color_distance_from(color, bar_color_rgb), color)
            text_colors_list.append(color_tuple)
        text_color = sorted(text_colors_list, key=lambda x: x[0])[-1][1]
        try: text_color = webcolors.name_to_hex(text_color)
        except: pass

    return text_color

class TestColorDistanceFrom(unittest.TestCase):

    @patch('webcolors.hex_to_rgb')
    def test_white_to_black(self, mock_hex_to_rgb):
        # Mock the hex_to_rgb function to avoid external dependency
        mock_hex_to_rgb.return_value = (255, 255, 255)  # White

        distance = color_distance_from('white', (0, 0, 0))  # Black

        # We expect the distance to be the diagonal of the RGB cube (all 255)
        expected_distance = sqrt(255**2 + 255**2 + 255**2)
        self.assertEqual(distance, expected_distance)

    @patch('webcolors.hex_to_rgb')
    def test_black_to_white(self, mock_hex_to_rgb):
        # Mock the hex_to_rgb function to avoid external dependency
        mock_hex_to_rgb.return_value = (0, 0, 0)  # Black

        distance = color_distance_from('black', (255, 255, 255))  # White

        # We expect the distance to be the diagonal of the RGB cube (all 255)
        expected_distance = sqrt(255**2 + 255**2 + 255**2)
        self.assertEqual(distance, expected_distance)

    @patch('webcolors.hex_to_rgb')
    def test_colored_distance(self, mock_hex_to_rgb):
        # Mock the hex_to_rgb function to simulate a colored input
        mock_hex_to_rgb.return_value = (100, 50, 150)  # Sample color

        distance = color_distance_from('#ffaa99', (0, 255, 0))  # Green

        # You'll need to calculate the expected distance based on the chosen color
        # This would involve calculating the difference between each color channel
        # and applying the distance formula (sqrt of squared differences)
        expected_distance = 272.9926738943739

        self.assertEqual(distance, expected_distance)

class TestGetTextColor(unittest.TestCase):

    @patch('test_color_functions.color_distance_from')  # Patching color_distance_from
    def test_default_white_text(self, mock_color_distance_from):
        # Mock color_distance_from to avoid complex calculations during test
        mock_color_distance_from.side_effect = [[10, 'white'], [20, '#404040'], [30, 'black']]

        text_color = get_text_color()
        self.assertEqual(text_color, 'white')

    @patch('test_color_functions.color_distance_from')  # Patching color_distance_from
    def test_black_bar_prefers_white_text(self, mock_color_distance_from):
        # Mock color_distance_from to control color distances
        mock_color_distance_from.side_effect = [[100, 'white'], [50, '#404040'], [0, 'black']]

        text_color = get_text_color(bar_color_rgb=(0, 0, 0))
        self.assertEqual(text_color, 'white')

    @patch('test_color_functions.color_distance_from')  # Patching color_distance_from
    def test_light_gray_bar_prefers_black_text(self, mock_color_distance_from):
        # Mock color_distance_from to control color distances
        mock_color_distance_from.side_effect = [(109.11920087683927, 'white'), (221.70250336881628, '#404040'), (332.55375505322445, 'black')]

        text_color = get_text_color(bar_color_rgb=(192, 192, 192))
        self.assertEqual(text_color, '#000000')

if __name__ == "__main__":
    unittest.main()