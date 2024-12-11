
from datetime import timedelta
from unittest.mock import patch
import os
import pandas as pd
import unittest

# Import the class containing the functions
import sys
if (osp.join('..', 'py') not in sys.path): sys.path.insert(1, osp.join('..', 'py'))
from FRVRS import fu, nu

class TestFRVRSResponderInit(unittest.TestCase):

    @patch('os.makedirs')
    def test_init_default_paths(self, mock_makedirs):
        # Test with default data and saves folder paths
        responder = FRVRSResponder()

        # Assert data folder path is set correctly
        self.assertEqual(responder.data_folder, '../data')
        # Assert saves folder path is set correctly
        self.assertEqual(responder.saves_folder, '../saves')

        # Assert os.makedirs was called for both folders
        mock_makedirs.assert_has_calls([
            patch.call(responder.data_folder, exist_ok=True),
            patch.call(responder.saves_folder, exist_ok=True)
        ])

    @patch('os.makedirs')
    def test_init_custom_paths(self, mock_makedirs):
        # Test with custom data and saves folder paths
        custom_data_folder = '/path/to/custom/data'
        custom_saves_folder = '/path/to/custom/saves'
        responder = FRVRSResponder(custom_data_folder, custom_saves_folder)

        # Assert data folder path is set correctly
        self.assertEqual(responder.data_folder, custom_data_folder)
        # Assert saves folder path is set correctly
        self.assertEqual(responder.saves_folder, custom_saves_folder)

        # Assert os.makedirs was called for both folders with custom paths
        mock_makedirs.assert_has_calls([
            patch.call(custom_data_folder, exist_ok=True),
            patch.call(custom_saves_folder, exist_ok=True)
        ])

    def test_init_verbose_enabled(self):
        # Test with verbose flag set to True
        responder = FRVRSResponder(verbose=True)

        # Assert verbose flag is set correctly
        self.assertTrue(responder.verbose)

    def test_init_verbose_disabled(self):
        # Test with verbose flag set to False (default)
        responder = FRVRSResponder()

        # Assert verbose flag is set correctly (default: False)
        self.assertFalse(responder.verbose)

### String Functions ###
    
    
class TestFormatTimedelta(unittest.TestCase):
    
    def setUp(self):
        # Test cases with different values and expected outputs
        self.test_cases = [
            ('zero seconds', timedelta(seconds=0), '0 sec', '0:00'),
            ('thirty seconds', timedelta(seconds=30), '30 sec', '0:30'),
            ('one minute', timedelta(minutes=1, seconds=0), '1 min', '1:00'),
            ('a minute and a half', timedelta(minutes=1, seconds=30), '1:30', '1:30'),
            ('two minutes', timedelta(minutes=2, seconds=0), '2 min', '2:00'),
            ('two and a half minutes', timedelta(minutes=2, seconds=30), '2:30', '2:30'),
            ('ten minutes', timedelta(minutes=10), '10 min', '10:00'),
        ]

    def test_format_seconds(self):
        """
        Tests formatting timedelta with minimum unit as seconds.
        """
        for description, delta, expected_string, _ in self.test_cases:
            formatted_string = fu.format_timedelta_lambda(delta)
            self.assertEqual(formatted_string, expected_string)

    def test_format_minutes(self):
        """
        Tests formatting timedelta with minimum unit as minutes.
        """
        for description, delta, _, expected_string in self.test_cases:
            formatted_string = fu.format_timedelta_lambda(delta, minimum_unit="minutes")
            self.assertEqual(formatted_string, expected_string)

    def test_invalid_minimum_unit(self):
        """
        Tests handling of invalid minimum_unit argument.
        """
        with self.assertRaises(ValueError):
            fu.format_timedelta_lambda(timedelta(minutes=1), minimum_unit="invalid_unit")

    
### List Functions ###
    
    
class TestReplaceConsecutiveElements(unittest.TestCase):
    
    def test_empty_list(self):
        self.assertEqual(nu.replace_consecutive_elements([]), [])
    
    def test_no_consecutive_elements(self):
        self.assertEqual(nu.replace_consecutive_elements([1, 2, 3]), [1, 2, 3])
    
    def test_consecutive_elements_at_start(self):
        self.assertEqual(nu.replace_consecutive_elements(['PATIENT_ENGAGED', 'PATIENT_ENGAGED', 2, 3]),
                         ['PATIENT_ENGAGED x2', 2, 3])
    
    def test_consecutive_elements_in_middle(self):
        self.assertEqual(nu.replace_consecutive_elements([1, 'PATIENT_ENGAGED', 'PATIENT_ENGAGED', 3, 4]),
                         [1, 'PATIENT_ENGAGED x2', 3, 4])
    
    def test_consecutive_elements_at_end(self):
        self.assertEqual(nu.replace_consecutive_elements([1, 2, 'PATIENT_ENGAGED', 'PATIENT_ENGAGED']),
                         [1, 2, 'PATIENT_ENGAGED x2'])
    
    def test_all_consecutive_elements(self):
        self.assertEqual(nu.replace_consecutive_elements(['PATIENT_ENGAGED', 'PATIENT_ENGAGED', 'PATIENT_ENGAGED']),
                         ['PATIENT_ENGAGED x3'])
    
    def test_different_element(self):
        self.assertEqual(nu.replace_consecutive_elements(['OTHER_ELEMENT', 'OTHER_ELEMENT', 1, 2], element='OTHER_ELEMENT'),
                         ['OTHER_ELEMENT x2', 1, 2])
    
    
### File Functions ###
    
    
class TestGetNewFileName(unittest.TestCase):

    @patch('builtins.open')
    @patch('csv.reader')
    def test_generate_new_file_name_with_am_date(self, mock_reader, mock_open):
        old_file_name = "my_log_file.csv"
        expected_date_str = "03/05/2024 09:45"
        expected_new_file_name = "24.03.05.0945.csv"

        mock_file = mock_open.return_value.__enter__.return_value
        mock_reader.return_value = [["data1", "data2", expected_date_str]]

        new_file_path = get_new_file_name(old_file_name)

        self.assertEqual(new_file_path, "../data/logs/" + expected_new_file_name)

    @patch('builtins.open')
    @patch('csv.reader')
    def test_generate_new_file_name_with_pm_date(self, mock_reader, mock_open):
        old_file_name = "another_log_file.csv"
        expected_date_str = "03/05/2024 05:30:00 PM"
        expected_new_file_name = "24.03.05.1730.csv"

        mock_file = mock_open.return_value.__enter__.return_value
        mock_reader.return_value = [["data1", "data2", expected_date_str]]

        new_file_path = get_new_file_name(old_file_name)

        self.assertEqual(new_file_path, "../data/logs/" + expected_new_file_name)

    @patch('builtins.open')
    def test_file_not_found(self, mock_open):
        mock_open.side_effect = FileNotFoundError("File not found")
        old_file_name = "missing_file.csv"

        with self.assertRaises(FileNotFoundError) as cm:
            get_new_file_name(old_file_name)

        self.assertEqual(str(cm.exception), "File not found")

if __name__ == "__main__":
    unittest.main()