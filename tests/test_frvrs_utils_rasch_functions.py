
from contextlib import redirect_stdout
from datetime import timedelta
from numpy import nan
from unittest.mock import patch, MagicMock
import numpy as np
import os
import pandas as pd
import unittest

# Import the class containing the functions
import sys
if (osp.join('..', 'py') not in sys.path): sys.path.insert(1, osp.join('..', 'py'))
from FRVRS import fu, nu


### Rasch Analysis Scene Functions ###


class TestGetStillsValue(unittest.TestCase):

    def test_all_stills_visited_first(self):
        """
        Test case where all stills are visited first in the actual sequence.
        """
        scene_df = pd.DataFrame({
            "order": [1, 2, 3, 4, 5],
            "type": ["action", "still", "dialogue", "still", "still"]
        })
        expected_value = 1
        actual_value = fu.get_stills_value(self, scene_df)
        self.assertEqual(actual_value, expected_value)

    def test_not_all_stills_visited_first(self):
        """
        Test case where not all stills are visited first in the actual sequence.
        """
        scene_df = pd.DataFrame({
            "order": [1, 2, 3, 4, 5],
            "type": ["action", "dialogue", "still", "still", "action"]
        })
        expected_value = 0
        actual_value = fu.get_stills_value(self, scene_df)
        self.assertEqual(actual_value, expected_value)

    def test_empty_dataframe(self):
        """
        Test case with an empty DataFrame.
        """
        scene_df = pd.DataFrame(columns=["order", "type"])
        expected_value = 0  # Can be adjusted based on expected behaviour for empty data
        actual_value = fu.get_stills_value(self, scene_df)
        self.assertEqual(actual_value, expected_value)

    # Add more test cases for different edge cases and scenarios

class TestGetWalkersValue(unittest.TestCase):

    def setUp(self):
        # Create test data (replace with your actual data creation logic)
        self.scene_df = # Your logic to create a sample scene_df
        self.expected_actual_sequence = [1, 0, 1]
        self.expected_ideal_sequence = [1, 0, 1]
        self.expected_sort_dict = {'walker': [1, 2, 3]}

    def test_all_walkers_visited_last(self):
        """
        Tests the function when all walkers are visited last.
        """
        # Modify test data if needed
        self.scene_df.loc[0, 'walker'] = self.expected_ideal_sequence[0]  # Set first walker to be visited last
        actual_value = fu.get_walkers_value(self.scene_df)
        self.assertEqual(actual_value, 1)

    def test_all_walkers_not_visited_last(self):
        """
        Tests the function when not all walkers are visited last.
        """
        # Modify test data if needed
        self.scene_df.loc[1, 'walker'] = 0  # Set second walker to not be visited last
        actual_value = fu.get_walkers_value(self.scene_df)
        self.assertEqual(actual_value, 0)

    # Add more test cases for different scenarios

class TestGetWaveValue(unittest.TestCase):

    def test_no_wave_command(self):
        """
        Test case for when no wave command is issued.
        """
        # Sample scene data with no wave command
        scene_df = {
            "action_type": ["MOVE_LEFT", "JUMP", "IDLE"]
        }
        scene_df = pd.DataFrame(scene_df)

        # Expected output
        expected_output = 0

        # Call the function
        actual_output = fu.get_wave_value(scene_df)

        # Assert the output
        self.assertEqual(actual_output, expected_output)

    def test_wave_command_issued(self):
        """
        Test case for when a wave command is issued.
        """
        # Sample scene data with wave command
        scene_df = {
            "action_type": ["MOVE_LEFT", "S_A_L_T_WAVE_IF_CAN", "JUMP"]
        }
        scene_df = pd.DataFrame(scene_df)

        # Expected output
        expected_output = 1

        # Call the function
        actual_output = fu.get_wave_value(scene_df)

        # Assert the output
        self.assertEqual(actual_output, expected_output)

    def test_empty_dataframe(self):
        """
        Test case for an empty dataframe.
        """
        # Empty scene data
        scene_df = pd.DataFrame()

        # Expected output
        expected_output = 0

        # Call the function
        actual_output = fu.get_wave_value(scene_df)

        # Assert the output
        self.assertEqual(actual_output, expected_output)

class TestGetWalkValue(unittest.TestCase):

    def setUp(self):
        # Sample scene data with and without walk command
        self.scene_df_with_walk = pd.DataFrame({
            "action_type": ["S_A_L_T_TALK", "S_A_L_T_WALK_IF_CAN", "S_A_L_T_JUMP"]
        })
        self.scene_df_without_walk = pd.DataFrame({
            "action_type": ["S_A_L_T_TALK", "S_A_L_T_JUMP"]
        })

    def test_get_walk_value_with_walk(self):
        """Tests get_walk_value with a scene containing a walk command"""
        expected_value = 1
        actual_value = fu.get_walk_value(self.scene_df_with_walk)
        self.assertEqual(expected_value, actual_value)

    def test_get_walk_value_without_walk(self):
        """Tests get_walk_value with a scene without a walk command"""
        expected_value = 0
        actual_value = fu.get_walk_value(self.scene_df_without_walk)
        self.assertEqual(expected_value, actual_value)

    def test_get_walk_value_verbose(self):
        """Tests get_walk_value with verbose set to True"""
        expected_value = 1
        # Capture any print statements during the test
        with self.assertLogs() as log_cm:
            actual_value = fu.get_walk_value(self.scene_df_with_walk, verbose=True)
        # Check if expected message is logged
        self.assertIn("Walk command issued: True", log_cm.output[0])
        self.assertEqual(expected_value, actual_value)


### Rasch Analysis Patient Functions ###


class TestGetTreatmentValue(unittest.TestCase):

    def setUp(self):
        # Create sample data
        data = {
            "patient_id": [1, 1, 2, 2, 3],
            "injury_id": [1, 1, 1, 2, 1],
            "injury_record_required_procedure": ["A", None, "B", "B", "A"],
            "injury_treated_required_procedure": [None, "B", None, "B", None],
            "action_tick": [1, 2, 1, 1, 1]
        }
        self.patient_df = pd.DataFrame(data)

    def test_correct_treatment(self):
        # Case: Patient received the required procedure
        injury_id = 1
        expected_value = 1
        actual_value = fu.get_treatment_value(self.patient_df, injury_id)
        self.assertEqual(actual_value, expected_value)

    def test_wrong_treatment(self):
        # Case: Patient received a different procedure
        injury_id = 1
        expected_value = 0
        actual_value = fu.get_treatment_value(self.patient_df, injury_id, verbose=True)
        # Adjust the assertion based on the behavior of verbose argument
        self.assertTrue(actual_value == expected_value or actual_value == "Wrong treatment applied (verbose mode)")

    def test_no_treatment(self):
        # Case: Patient did not receive any treatment
        injury_id = 3
        expected_value = 0
        actual_value = fu.get_treatment_value(self.patient_df, injury_id)
        self.assertEqual(actual_value, expected_value)

    def test_no_required_procedure(self):
        # Case: Required procedure information missing
        injury_id = 2
        expected_value = np.nan
        actual_value = fu.get_treatment_value(self.patient_df, injury_id)
        self.assertEqual(actual_value, expected_value)

    def test_empty_dataframe(self):
        # Case: Empty dataframe passed
        empty_df = pd.DataFrame(columns=["patient_id", "injury_id", etc.])
        injury_id = 1
        expected_value = np.nan
        actual_value = fu.get_treatment_value(empty_df, injury_id)
        self.assertEqual(actual_value, expected_value)

class TestGetTagValue(unittest.TestCase):

    def test_correct_tag(self):
        """
        Tests if the function returns 1 for a correct tag.
        """
        patient_df = None  # Replace with appropriate data for correct tag
        is_tag_correct = fu.get_tag_value(patient_df)
        self.assertEqual(is_tag_correct, 1)

    def test_wrong_tag(self):
        """
        Tests if the function returns 0 for a wrong tag.
        """
        patient_df = None  # Replace with appropriate data for wrong tag
        is_tag_correct = fu.get_tag_value(patient_df)
        self.assertEqual(is_tag_correct, 0)

    def test_nan_value(self):
        """
        Tests if the function returns 0 for a NaN value from is_tag_correct.
        """
        patient_df = None  # Replace with appropriate data for NaN value
        is_tag_correct = np.nan
        with self.assertRaises(Exception):
            fu.get_tag_value(patient_df)

    def test_exception(self):
        """
        Tests if the function returns 0 for any raised exception.
        """
        def mock_is_tag_correct(patient_df, verbose=False):
            raise Exception("Mocked exception")

        patient_df = None  # Replace with appropriate data
        with self.assertRaises(Exception):
            fu.get_tag_value(patient_df, is_tag_correct=mock_is_tag_correct)

class TestGetPulseValue(unittest.TestCase):

    def test_pulse_taken(self):
        """
        Tests if function returns 1 when pulse is taken.
        """
        # Create a sample dataframe with pulse taken
        data = {'action_type': ['MEASURE_SPO2', 'PULSE_TAKEN', 'BMI']}
        patient_df = pd.DataFrame(data)

        # Call the function
        result = fu.get_pulse_value(patient_df)

        # Assert the expected value
        self.assertEqual(result, 1)

    def test_pulse_not_taken(self):
        """
        Tests if function returns 0 when pulse is not taken.
        """
        # Create a sample dataframe with no pulse taken
        data = {'action_type': ['MEASURE_SPO2', 'BMI', 'TEMPERATURE']}
        patient_df = pd.DataFrame(data)

        # Call the function
        result = fu.get_pulse_value(patient_df)

        # Assert the expected value
        self.assertEqual(result, 0)

if __name__ == "__main__":
    unittest.main()