
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
sys.path.insert(1, '../py')
from FRVRS import fu, nu


### Scene Functions ###


class TestGetSceneStart(unittest.TestCase):

    def test_get_scene_start_simple(self):
        scene_df = pd.DataFrame({'action_tick': [1500, 2000, 2500]})
        expected_start = 1500
        actual_start = get_scene_start(scene_df)
        self.assertEqual(actual_start, expected_start)

    def test_get_scene_start_empty_df(self):
        scene_df = pd.DataFrame({'action_tick': []})
        with self.assertRaises(ValueError):
            get_scene_start(scene_df)

    def test_get_scene_start_missing_column(self):
        scene_df = pd.DataFrame({'other_column': [10, 20, 30]})
        with self.assertRaises(KeyError):
            get_scene_start(scene_df)

    def test_get_scene_start_verbose(self):
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            scene_df = pd.DataFrame({'action_tick': [1425.3654, 1437.2546]})
            get_scene_start(scene_df, verbose=True)
            self.assertEqual(mock_stdout.getvalue().strip(), 'Scene start: 1425.3654')

class TestGetLastEngagement(unittest.TestCase):

    def test_empty_dataframe(self):
        """Tests behavior with an empty DataFrame."""
        empty_df = pd.DataFrame({'action_type': []})
        result = get_last_engagement(empty_df)
        self.assertEqual(result, None)

    def test_no_patient_engagements(self):
        """Tests behavior with no PATIENT_ENGAGED actions."""
        df = pd.DataFrame({'action_type': ['OTHER_ACTION'] * 5})
        result = get_last_engagement(df)
        self.assertEqual(result, None)

    def test_single_patient_engagement(self):
        """Tests behavior with a single PATIENT_ENGAGED action."""
        df = pd.DataFrame({'action_type': ['PATIENT_ENGAGED'], 'action_tick': [1000]})
        result = get_last_engagement(df)
        self.assertEqual(result, 1000)

    def test_multiple_patient_engagements(self):
        """Tests behavior with multiple PATIENT_ENGAGED actions."""
        df = pd.DataFrame({
            'action_type': ['OTHER_ACTION', 'PATIENT_ENGAGED', 'OTHER_ACTION', 'PATIENT_ENGAGED'],
            'action_tick': [500, 1000, 1500, 2000]
        })
        result = get_last_engagement(df)
        self.assertEqual(result, 2000)

    def test_verbose_output(self):
        """Tests verbose output functionality."""
        with self.assertLogs() as captured_logs:
            df = pd.DataFrame({
                'action_type': ['PATIENT_ENGAGED'],
                'action_tick': [12345]
            })
            get_last_engagement(df, verbose=True)
            self.assertEqual(captured_logs.output[0], 'INFO:root:Last engagement time: 12345')

class TestGetPlayerLocation(unittest.TestCase):

    def setUp(self):
        # Sample scene data
        data = {
            "action_type": ["PLAYER_LOCATION", "ATTACK", "HEAL"],
            "action_tick": [10, 20, 15],
            "location_id": ["'1,2'", "'3,4'", "'5,6'"]
        }
        self.scene_df = pd.DataFrame(data)

    def test_player_location_found(self):
        action_tick = 15
        expected_location = (1, 2)

        player_location = get_player_location(self.scene_df, action_tick)

        self.assertEqual(player_location, expected_location)

    def test_player_location_not_found(self):
        action_tick = 30  # No PLAYER_LOCATION action at this tick
        expected_error_message = "no player location found"

        with self.assertRaises(ValueError) as cm:
            get_player_location(self.scene_df, action_tick)

        self.assertEqual(str(cm.exception), expected_error_message)

class TestGetSceneType(unittest.TestCase):

    def test_scene_type_column_exists(self):
        # Create a DataFrame with a 'scene_type' column
        data = {'scene_type': ['Triage']}
        scene_df = pd.DataFrame(data)

        # Call the function and assert the returned scene type
        scene_type = get_scene_type(scene_df)
        self.assertEqual(scene_type, 'Triage')

    def test_scene_type_column_multiple_values(self):
        # Create a DataFrame with a 'scene_type' column containing multiple values
        data = {'scene_type': ['Triage', 'Transport']}
        scene_df = pd.DataFrame(data)

        # Call the function with an expected exception
        with self.assertRaises(Exception) as cm:
            get_scene_type(scene_df)

        # Assert the expected error message
        self.assertEqual(str(cm.exception), "scene_types=['Triage' 'Transport'] has more than one entry")

    def test_scene_type_column_missing(self):
        # Create a DataFrame without a 'scene_type' column
        data = {'patient_id': ['123', '456']}
        scene_df = pd.DataFrame(data)

        # Call the function and assert the returned scene type
        scene_type = get_scene_type(scene_df)
        self.assertEqual(scene_type, 'Triage')

    def test_orientation_scene_type(self):
        # Create a DataFrame with patient IDs containing 'mike'
        data = {'patient_id': ['123 Mike', '456 Smith']}
        scene_df = pd.DataFrame(data)

        # Call the function and assert the returned scene type
        scene_type = get_scene_type(scene_df)
        self.assertEqual(scene_type, 'Orientation')

    def test_other_scene_type(self):
        # Create a DataFrame with patient IDs not containing 'mike'
        data = {'patient_id': ['123 Johnson', '456 Williams']}
        scene_df = pd.DataFrame(data)

        # Call the function and assert the returned scene type
        scene_type = get_scene_type(scene_df)
        self.assertEqual(scene_type, 'Triage')

class TestGetSceneEnd(unittest.TestCase):

    def test_get_scene_end_basic(self):
        # Create a sample DataFrame with scene data
        data = {
            "action_tick": [100, 250, 175, 300]
        }
        scene_df = pd.DataFrame(data)

        # Call the function and compare the result with expected value
        expected_end_time = 300
        actual_end_time = get_scene_end(scene_df)
        self.assertEqual(actual_end_time, expected_end_time)

    def test_get_scene_end_verbose(self):
        # Create a sample DataFrame with scene data
        data = {
            "action_tick": [100, 250, 175, 300]
        }
        scene_df = pd.DataFrame(data)

        # Mock the print function to capture output
        with patch('builtins.print') as mock_print:
            get_scene_end(scene_df, verbose=True)

            # Assert that the expected message was printed
            self.assertTrue(mock_print.called)
            mock_print.assert_called_once_with(f'Scene end time calculated: 300')

class TestGetPatientCount(unittest.TestCase):

    def setUp(self):
        # Create sample data with different scenarios
        self.data_1 = {
            "patient_id": [1, 2, 1, 3, 1],
            "other_column": ["A", "B", "C", "D", "E"],
        }
        self.data_2 = {
            "patient_id": [1, 1, 1],
            "other_column": ["X", "Y", "Z"],
        }
        self.df_1 = pd.DataFrame(self.data_1)
        self.df_2 = pd.DataFrame(self.data_2)

    def test_unique_patients_verbose_false(self):
        # Test with unique patients, verbose=False
        expected_count = 3
        actual_count = get_patient_count(self.df_1)
        self.assertEqual(actual_count, expected_count)

    def test_unique_patients_verbose_true(self):
        # Test with unique patients, verbose=True
        expected_count = 3
        # Capture the printed output using a context manager
        with captured_output() as (stdout, stderr):
            actual_count = get_patient_count(self.df_1, verbose=True)
        # Check for expected output and count
        self.assertEqual(actual_count, expected_count)
        self.assertIn(f"Number of unique patients: {expected_count}", stdout.getvalue())

    def test_duplicate_patients_verbose_false(self):
        # Test with duplicate patients, verbose=False
        expected_count = 1
        actual_count = get_patient_count(self.df_2)
        self.assertEqual(actual_count, expected_count)

    def test_duplicate_patients_verbose_true(self):
        # Test with duplicate patients, verbose=True
        expected_count = 1
        # Capture the printed output using a context manager
        with captured_output() as (stdout, stderr):
            actual_count = get_patient_count(self.df_2, verbose=True)
        # Check for expected output and count
        self.assertEqual(actual_count, expected_count)
        self.assertIn(f"Number of unique patients: {expected_count}", stdout.getvalue())


# Helper function to capture printed output
def captured_output(target_stdout=None):
    """Captures and stores the output printed to stdout during a block of code."""
    with redirect_stdout(target_stdout):
        # Run the code that generates the output
        yield

class TestGetInjuryTreatmentsCount(unittest.TestCase):

    def setUp(self):
        # Create sample data
        data = {
            "scene_id": [1, 2, 3, 4],
            "injury_treated_injury_treated": [True, False, True, False]
        }
        self.scene_df = pd.DataFrame(data)

    def test_injury_treatments_count(self):
        # Call the function with default parameters
        count = get_injury_treatments_count(self.scene_df)

        # Assert the expected count
        self.assertEqual(count, 2)

    def test_injury_treatments_count_verbose(self):
        # Call the function with verbose set to True
        count = get_injury_treatments_count(self.scene_df, verbose=True)

        # Assert the expected count
        self.assertEqual(count, 2)

class TestGetInjuryNotTreatedCount(unittest.TestCase):

   def test_empty_dataframe(self):
       """Test with an empty DataFrame."""
       scene_df = pd.DataFrame()
       count = get_injury_not_treated_count(scene_df)
       self.assertEqual(count, 0)

   def test_no_untreated_injuries(self):
       """Test with a DataFrame where all injuries have been treated."""
       data = {'injury_treated_injury_treated': [True, True, True]}
       scene_df = pd.DataFrame(data)
       count = get_injury_not_treated_count(scene_df)
       self.assertEqual(count, 0)

   def test_some_untreated_injuries(self):
       """Test with a DataFrame containing a mix of treated and untreated injuries."""
       data = {'injury_treated_injury_treated': [True, False, True, False]}
       scene_df = pd.DataFrame(data)
       count = get_injury_not_treated_count(scene_df)
       self.assertEqual(count, 2)

   def test_verbose_output(self):
       """Test with verbose output enabled."""
       data = {'injury_treated_injury_treated': [True, False, True]}
       scene_df = pd.DataFrame(data)
       try:
           count = get_injury_not_treated_count(scene_df, verbose=True)  # Capture any printed output
           self.fail("Expected a print statement when verbose=True")
       except Exception as e:
           self.assertIn("Number of records where injuries were not treated: 1", str(e))

class TestGetInjuryCorrectlyTreatedCount(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame with different treatment scenarios
        data = {
            "injury_id": [1, 1, 2, 3, 3],
            "injury_treated_required_procedure": ["Bandage", None, "Stitches", "Bandage", None],
            "injury_treated_injury_treated_with_wrong_treatment": [False, True, False, False, True]
        }
        self.scene_df = pd.DataFrame(data)

    def test_injury_correctly_treated(self):
        # Test with expected correctly treated cases
        expected_count = 2
        actual_count = self.scene_df.apply(
            lambda x: x.get_injury_correctly_treated_count(verbose=False),
            axis=1
        ).values[0]
        self.assertEqual(actual_count, expected_count)

    def test_no_injury_treated(self):
        # Test with no correctly treated cases
        data = {
            "injury_id": [1, 2, 3],
            "injury_treated_required_procedure": [None, "Stitches", None],
            "injury_treated_injury_treated_with_wrong_treatment": [True, False, True]
        }
        df = pd.DataFrame(data)
        expected_count = 0
        actual_count = df.apply(
            lambda x: x.get_injury_correctly_treated_count(verbose=False),
            axis=1
        ).values[0]
        self.assertEqual(actual_count, expected_count)

    def test_verbose_output(self):
        # Test with verbose mode enabled (no assertion, just checking output)
        self.scene_df.apply(
            lambda x: x.get_injury_correctly_treated_count(verbose=True),
            axis=1
        )

class TestGetInjuryWronglyTreatedCount(unittest.TestCase):

    @staticmethod
    def create_test_dataframe(injury_treated_values, injury_treated_wrongly_values):
        """Helper function to create a DataFrame for testing"""
        data = {
            "injury_treated_injury_treated": injury_treated_values,
            "injury_treated_injury_treated_with_wrong_treatment": injury_treated_wrongly_values
        }
        return pd.DataFrame(data)

    def test_no_wrong_treatments(self):
        """Test when there are no injuries treated incorrectly"""
        scene_df = self.create_test_dataframe([True, False, True], [False, False, False])
        count = get_injury_wrongly_treated_count(scene_df)
        self.assertEqual(count, 0)

    def test_one_wrong_treatment(self):
        """Test when there's one injury treated incorrectly"""
        scene_df = self.create_test_dataframe([True, True, False], [True, False, False])
        count = get_injury_wrongly_treated_count(scene_df)
        self.assertEqual(count, 1)

    def test_multiple_wrong_treatments(self):
        """Test when there are multiple injuries treated incorrectly"""
        scene_df = self.create_test_dataframe([True, True, True], [True, True, True])
        count = get_injury_wrongly_treated_count(scene_df)
        self.assertEqual(count, 3)

    def test_verbose_output(self):
        """Test the verbose output"""
        scene_df = self.create_test_dataframe([True], [True])
        with self.assertLogs() as captured_logs:
            count = get_injury_wrongly_treated_count(scene_df, verbose=True)
            self.assertEqual(count, 1)
            self.assertIn('Injury wrongly treated count: 1', captured_logs.output[0])

class TestGetPulseTakenCount(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            "scene_id": [1, 1, 2, 3],
            "action_type": ["MOVE", "PULSE_TAKEN", "TALK", "PULSE_TAKEN"]
        }
        self.scene_df = pd.DataFrame(data)

    def test_get_pulse_taken_count_no_verbose(self):
        """
        Test get_pulse_taken_count with verbose=False.
        """
        count = self.scene_df.get_pulse_taken_count(verbose=False)

        self.assertEqual(count, 2)
        self.assertFalse(len(self.scene_df) > 0)  # No additional data printed in verbose=False

    def test_get_pulse_taken_count_verbose(self):
        """
        Test get_pulse_taken_count with verbose=True.
        """
        with self.assertRaises(AssertionError):  # We don't directly capture printed output
            count = self.scene_df.get_pulse_taken_count(verbose=True)
            self.assertEqual(count, 2)  # This assertion would fail if the count is incorrect

class TestGetTeleportCount(unittest.TestCase):

    def setUp(self):
        # Create sample scene data
        data = {
            "action_type": ["MOVE", "TELEPORT", "TALK", "TELEPORT", "FIGHT"],
            "other_data": ["data1", "data2", "data3", "data4", "data5"]
        }
        self.scene_df = pd.DataFrame(data)

    def test_get_teleport_count_success(self):
        # Call the function with default verbose (False)
        teleport_count = self.get_teleport_count(self.scene_df)

        # Assert the expected teleport count
        self.assertEqual(teleport_count, 2)

    def test_get_teleport_count_verbose(self):
        # Call the function with verbose set to True
        teleport_count = self.get_teleport_count(self.scene_df, verbose=True)

        # Assert the expected teleport count (same as previous test)
        self.assertEqual(teleport_count, 2)

        # Check if a print statement containing "Number of TELEPORT actions" was executed
        # (Mocking print statements is generally not recommended, but in this case, it can 
        # help verify if the verbose functionality works as intended)
        # with patch('builtins.print') as mock_print:
        #     self.get_teleport_count(self.scene_df, verbose=True)
        #     mock_print.assert_called_once_with(f'Number of TELEPORT actions: {teleport_count}')

class TestGetVoiceCaptureCount(unittest.TestCase):

    def setUp(self):
        # Create sample scene data
        data = {
            "action_type": ["MOVE", "VOICE_CAPTURE", "ROTATE", "VOICE_CAPTURE"],
            # Other columns can be added here if needed
        }
        self.scene_df = pd.DataFrame(data)

    def test_get_voice_capture_count(self):
        """
        Tests the get_voice_capture_count function with a sample DataFrame.
        """
        expected_count = 2  # Expected number of VOICE_CAPTURE actions

        # Call the function without verbosity
        voice_capture_count = get_voice_capture_count(self.scene_df)

        # Assert that the returned count matches the expected value
        self.assertEqual(voice_capture_count, expected_count)

    def test_get_voice_capture_count_verbose(self):
        """
        Tests the get_voice_capture_count function with verbosity enabled.
        """
        # Patch the `display` function to avoid actual display during testing
        with patch('get_voice_capture_count.display') as mock_display:
            voice_capture_count = get_voice_capture_count(self.scene_df, verbose=True)

            # Assert that the function was called with the expected parameters
            get_voice_capture_count.assert_called_once_with(self.scene_df, True)

            # Assert that the mock display function was not called (since we don't want actual display during testing)
            mock_display.assert_not_called()

class TestGetWalkCommandCount(unittest.TestCase):

    # Define sample DataFrames for testing
    def setUp(self):
        self.test_data = [
            {'voice_command_message': 'walk to the safe area', 'other_column': 1},
            {'voice_command_message': 'stop', 'other_column': 2},
            {'voice_command_message': 'walk to the safe area', 'other_column': 3},
            {'voice_command_message': 'turn left', 'other_column': 4}
        ]
        self.scene_df = pd.DataFrame(self.test_data)

    # Test with a DataFrame containing "walk to the safe area" commands
    def test_walk_command_count_valid(self):
        count = get_walk_command_count(self.scene_df)
        self.assertEqual(count, 2)

    # Test with a DataFrame without any "walk to the safe area" commands
    def test_walk_command_count_zero(self):
        scene_df_no_command = self.scene_df.loc[self.scene_df['voice_command_message'] != 'walk to the safe area']
        count = get_walk_command_count(scene_df_no_command)
        self.assertEqual(count, 0)

    # Test with verbose mode enabled
    def test_walk_command_count_verbose(self):
        with patch('builtins.print') as mock_print:
            get_walk_command_count(self.scene_df, verbose=True)
            mock_print.assert_called_once_with(f"Number of 'walk to the safe area' voice command events: 2")

class TestGetWaveCommandCount(unittest.TestCase):

    def setUp(self):
        # Create sample scene data
        data = {
            "scene_id": [1, 2, 3],
            "voice_command_message": ["open the door", "wave if you can", "play music"],
        }
        self.scene_df = pd.DataFrame(data)

    def test_wave_command_count(self):
        # Test with expected wave command
        count = self.scene_df.get_wave_command_count()
        self.assertEqual(count, 1)

        # Test with no wave command
        data = {"scene_id": [1], "voice_command_message": ["open the door"]}
        df = pd.DataFrame(data)
        count = df.get_wave_command_count()
        self.assertEqual(count, 0)

    def test_verbose_output(self):
        # Capture output using context manager
        with patch('builtins.print') as mock_print:
            self.scene_df.get_wave_command_count(verbose=True)
            mock_print.assert_called_once_with(
                f"Number of 'wave if you can' voice command events: 1"
            )

class TestGetFirstEngagement(unittest.TestCase):

    def test_empty_dataframe(self):
        """
        Tests if the function returns None with an empty DataFrame.
        """
        empty_df = pd.DataFrame(columns=["action_type", "action_tick"])
        first_engagement = get_first_engagement(empty_df)
        self.assertIsNone(first_engagement)

    def test_no_patient_engaged_action(self):
        """
        Tests if the function returns None when no "PATIENT_ENGAGED" action exists.
        """
        no_engagement_df = pd.DataFrame(
            {
                "action_type": ["MOVE", "LOOK", "GRASP"],
                "action_tick": [10, 20, 30],
            }
        )
        first_engagement = get_first_engagement(no_engagement_df)
        self.assertIsNone(first_engagement)

    def test_single_patient_engaged_action(self):
        """
        Tests if the function returns the correct action tick with a single "PATIENT_ENGAGED" action.
        """
        single_engagement_df = pd.DataFrame(
            {
                "action_type": ["MOVE", "PATIENT_ENGAGED", "GRASP"],
                "action_tick": [10, 20, 30],
            }
        )
        first_engagement = get_first_engagement(single_engagement_df)
        self.assertEqual(first_engagement, 20)

    def test_multiple_patient_engaged_actions(self):
        """
        Tests if the function returns the action tick of the first "PATIENT_ENGAGED" action with multiple occurrences.
        """
        multiple_engagement_df = pd.DataFrame(
            {
                "action_type": ["MOVE", "PATIENT_ENGAGED", "GRASP", "PATIENT_ENGAGED"],
                "action_tick": [10, 20, 30, 40],
            }
        )
        first_engagement = get_first_engagement(multiple_engagement_df)
        self.assertEqual(first_engagement, 20)

    def test_verbose_output(self):
        """
        Tests if verbose output is printed when verbose argument is True.
        """
        # Patch printing function to capture output in a test
        with patch('builtins.print') as mock_print:
            verbose_df = pd.DataFrame(
                {
                    "action_type": ["MOVE", "PATIENT_ENGAGED", "GRASP"],
                    "action_tick": [10, 20, 30],
                }
            )
            get_first_engagement(verbose_df, verbose=True)

            # Check if expected messages were printed
            mock_print.assert_called_once_with(
                f'Action tick of the first PATIENT_ENGAGED action: {20}'
            )

class TestGetFirstTreatment(unittest.TestCase):

    # Define a consistent setup method to create a sample DataFrame
    def setUp(self):
        self.sample_df = pd.DataFrame({
            'action_type': ['MOVE', 'INJURY_TREATED', 'ATTACK', 'INJURY_TREATED'],
            'action_tick': [10, 25, 30, 45]
        })

    # Test for a DataFrame with INJURY_TREATED actions
    def test_first_treatment_found(self):
        first_treatment = get_first_treatment(self.sample_df)
        self.assertEqual(first_treatment, 25)

    # Test for a DataFrame without INJURY_TREATED actions
    def test_no_treatment_found(self):
        no_treatment_df = self.sample_df[self.sample_df['action_type'] != 'INJURY_TREATED']
        first_treatment = get_first_treatment(no_treatment_df)
        self.assertIsNone(first_treatment)

    # Test verbose mode
    def test_verbose_output(self):
        with self.assertLogs(level='INFO') as cm:
            get_first_treatment(self.sample_df, verbose=True)
        self.assertEqual(cm.output, ['INFO:root:Action tick of the first INJURY_TREATED action: 25'])

# Mock your external dependencies (nu.get_nearest_neighbor)
def mock_get_nearest_neighbor(player_location, locations_list):
    # Implement your mock logic here to return a nearest neighbor based on your needs
    # for testing. You can return different values to test various scenarios.
    return locations_list[0]  # Example: always return the first element

class TestGetIdealEngagementOrder(unittest.TestCase):

    def setUp(self):
        # Mock dependencies (replace with your actual implementations)
        self.mock_get_engagement_starts_order = MagicMock()
        self.mock_nu = MagicMock()

    @patch('your_module.nu.get_nearest_neighbor', side_effect=mock_get_nearest_neighbor)
    def test_empty_scene_df(self, mock_get_nearest_neighbor):
        # Create an empty DataFrame for scene_df
        scene_df = pd.DataFrame(columns=['action_type', 'location_id', 'action_tick'])

        # Call the function and check the returned value
        ideal_order = self.get_ideal_engagement_order(scene_df)

        self.assertEqual(ideal_order, [])
    
    @patch('your_module.nu.get_nearest_neighbor', side_effect=mock_get_nearest_neighbor)
    def test_single_patient_high_severity(self, mock_get_nearest_neighbor):
        # Set up scene_df
        scene_df = pd.DataFrame({
            'action_type': ['PATIENT_ARRIVAL', 'PLAYER_LOCATION'],
            'location_id': [str((1.0, 2.0, 3.0)), str((4.0, 5.0, 6.0))],
            'action_tick': [1, 2]
        })

        # Set mock sort_category_order and severity_category_order
        self.sort_category_order = ["low", "medium", "high"]
        self.severity_category_order = ["low", "medium", "high"]

        # Set up mock return values
        mock_get_engagement_starts_order.return_value = pd.DataFrame({
            'patient_id': [1],
            'engagement_start': [2],
            'location_tuple': ['(2.0, 3.0, 4.0)'],
            'patient_sort': [1],
            'predicted_priority': [1],
            'injury_severity': ['high']
        })
        mock_nu.get_nearest_neighbor.side_effect = lambda x, y: y[0]

        # Call the function
        ideal_order = self.get_ideal_engagement_order(scene_df)

        # Assertions
        self.assertEqual(ideal_order, [tuple(mock_get_engagement_starts_order.return_value.iloc[0])])

    @patch('your_module.nu.get_nearest_neighbor', side_effect=mock_get_nearest_neighbor)
    def test_multiple_patients_mixed_severity(self, mock_get_nearest_neighbor):
        # Create a DataFrame with multiple patients and mixed severity
        scene_df = pd.DataFrame({
            'action_type': ['PATIENT_ARRIVAL', 'PATIENT_ARRIVAL', 'PLAYER_LOCATION'],
            'location_id': [str((1.0, 2.0, 3.0)), str((4.0, 5.0, 6.0)), str((3.0, 2.0, 1.0))],
            'action_tick': [1, 2, 3]
        })

        # Set mock sort_category_order and severity_category_order
        self.sort_category_order = ["low", "medium", "high"]
        self.severity_category_order = ["low", "medium", "high"]

        # Call the function with verbose set to True
        ideal_order = self.get_ideal_engagement_order(scene_df, verbose=True)

        # Expected order: [('high', 4.0, 5.0, 6.0, 'low', 'low'), ('medium', 1.0, 2.0, 3.0, 'low', 'low')]
        expected_order = [('high', 4.0, 5.0, 6.0, 'low', 'low'), ('medium', 1.0, 2.0, 3.0, 'low', 'low')]
        self.assertEqual(ideal_order, expected_order)

    @patch("your_module.nu.get_nearest_neighbor")  # Replace "your_module" with the actual module name
    def test_ideal_engagement_order_high_severity_first(self, mock_get_nearest_neighbor):
        # Set up mock data for scene_df, get_engagement_starts_order, and nu.get_nearest_neighbor
        scene_df = pd.DataFrame(...)  # Create a DataFrame with appropriate test data
        self.instance.get_engagement_starts_order = MagicMock(return_value=[...])  # Set up the mock for get_engagement_starts_order
        mock_get_nearest_neighbor.side_effect = [...]  # Set up a sequence of nearest neighbors for testing

        # Call the function and make assertions on the result
        ideal_engagement_order = self.instance.get_ideal_engagement_order(scene_df)

        # Assert that patients with high severity are prioritized correctly
        self.assertEqual(ideal_engagement_order[0][5], 'high')  # Assuming 'injury_severity' is at index 5 in the tuple

        # Assert that patients are ordered based on nearest neighbor within each priority group
        # ... (add assertions to check the order based on your expected logic)

    # Add more test cases to cover different scenarios and edge cases, such as:
    # - Empty scene_df
    @patch("your_module.nu.get_nearest_neighbor", side_effect=mock_get_nearest_neighbor)
    def test_ideal_engagement_order_empty_dataframe(self, mock_get_nearest_neighbor, mock_get_engagement_starts_order):
        # Set up empty scene_df
        scene_df = pd.DataFrame(columns=['action_type', 'location_id', 'action_tick'])

        # Mock return values
        mock_get_engagement_starts_order.return_value = pd.DataFrame(columns=[])
        mock_nu.get_nearest_neighbor.return_value = None

        # Call the function
        ideal_order = self.get_ideal_engagement_order(scene_df)

        # Assertions
        self.assertEqual(ideal_order, [])

    # - Different patient sorts
    # - Varying player locations
    # - Handling of exceptions or errors

class TestGetIsSceneAborted(unittest.TestCase):

    def test_scene_aborted_column_exists(self):
        # Create a mock DataFrame with the 'is_scene_aborted' column
        scene_df = pd.DataFrame({'is_scene_aborted': [True]})

        # Call the function
        is_scene_aborted = self.get_is_scene_aborted(scene_df)

        # Assert the expected return value
        self.assertTrue(is_scene_aborted)

    def test_scene_aborted_column_not_exists(self):
        # Create a mock DataFrame without the 'is_scene_aborted' column
        scene_df = pd.DataFrame({'other_column': [1]})

        # Call the function
        is_scene_aborted = self.get_is_scene_aborted(scene_df)

        # Assert the expected return value (False) and that get_scene_start and get_last_engagement weren't called
        self.assertFalse(is_scene_aborted)
        self.assertFalse(hasattr(self, 'get_scene_start_called'))
        self.assertFalse(hasattr(self, 'get_last_engagement_called'))

    def test_scene_aborted_exceeds_threshold(self):
        # Mock the scene_start and last_engagement functions to return specific values
        def mock_get_scene_start(scene_df, verbose):
            self.get_scene_start_called = True
            return pd.Timestamp('2023-01-01 10:00:00')

        def mock_get_last_engagement(scene_df, verbose):
            self.get_last_engagement_called = True
            return pd.Timestamp('2023-01-01 10:20:00')

        self.get_scene_start = mock_get_scene_start
        self.get_last_engagement = mock_get_last_engagement

        # Create a mock DataFrame (doesn't matter for this test)
        scene_df = pd.DataFrame({})

        # Call the function
        is_scene_aborted = self.get_is_scene_aborted(scene_df)

        # Assert the expected return value and that get_scene_start and get_last_engagement were called
        self.assertTrue(is_scene_aborted)
        self.assertTrue(hasattr(self, 'get_scene_start_called'))
        self.assertTrue(hasattr(self, 'get_last_engagement_called'))

    def test_scene_not_aborted_below_threshold(self):
        # Mock the scene_start and last_engagement functions to return specific values
        def mock_get_scene_start(scene_df, verbose):
            self.get_scene_start_called = True
            return pd.Timestamp('2023-01-01 10:00:00')

        def mock_get_last_engagement(scene_df, verbose):
            self.get_last_engagement_called = True
            return pd.Timestamp('2023-01-01 10:14:00')

        self.get_scene_start = mock_get_scene_start
        self.get_last_engagement = mock_get_last_engagement

        # Create a mock DataFrame (doesn't matter for this test)
        scene_df = pd.DataFrame({})

        # Call the function
        is_scene_aborted = self.get_is_scene_aborted(scene_df)

        # Assert the expected return value and that get_scene_start and get_last_engagement were called
        self.assertFalse(is_scene_aborted)
        self.assertTrue(hasattr(self, 'get_scene_start_called'))
        self.assertTrue(hasattr(self, 'get_last_engagement_called'))

class TestGetTriageTime(unittest.TestCase):

    def setUp(self):
        # Create a mock DataFrame with scene start and end time columns
        self.scene_df = pd.DataFrame({
            "start_time": pd.to_datetime(["01:00:00", "02:00:00"]),
            "end_time": pd.to_datetime(["01:30:00", "02:15:00"])
        })

    def test_get_triage_time(self):
        # Test with first scene
        triage_time = self.get_triage_time(self.scene_df.iloc[0])
        self.assertEqual(triage_time, 1800000)  # 1800 seconds (30 minutes)

    def test_get_triage_time_verbose(self):
        # Test with verbose output (optional, modify if your function prints details)
        with self.assertLogs(level="INFO") as cm:
            self.get_triage_time(self.scene_df.iloc[0], verbose=True)
        self.assertIn("Scene triage time:", cm.output[0])

class TestGetDeadPatients(unittest.TestCase):

    # Placeholder for the self.salt_columns_list attribute
    salt_columns_list = ['column1', 'column2']  # Replace with actual column names

    def test_single_dead_patient(self):
        scene_df = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'column1': ['ALIVE', 'DEAD', 'ALIVE'],
            'column2': ['ALIVE', 'ALIVE', 'EXPECTANT']
        })

        expected_dead_list = [2, 3]
        actual_dead_list = self.get_dead_patients(scene_df)

        self.assertEqual(actual_dead_list, expected_dead_list)

    def test_multiple_dead_patients(self):
        scene_df = pd.DataFrame({
            'patient_id': [1, 2, 3, 4, 5],
            'column1': ['DEAD', 'EXPECTANT', 'ALIVE', 'DEAD', 'ALIVE'],
            'column2': ['ALIVE', 'ALIVE', 'EXPECTANT', 'EXPECTANT', 'DEAD']
        })

        expected_dead_list = [1, 2, 4, 5]
        actual_dead_list = self.get_dead_patients(scene_df)

        self.assertEqual(actual_dead_list, expected_dead_list)

    def test_no_dead_patients(self):
        scene_df = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'column1': ['ALIVE', 'ALIVE', 'ALIVE'],
            'column2': ['ALIVE', 'ALIVE', 'ALIVE']
        })

        expected_dead_list = []
        actual_dead_list = self.get_dead_patients(scene_df)

        self.assertEqual(actual_dead_list, expected_dead_list)

class TestGetStillPatients(unittest.TestCase):

    # Placeholder for the actual list of sort columns
    sort_columns_list = ['column1', 'column2']  # Replace with actual column names

    def test_empty_dataframe(self):
        """Test with an empty DataFrame."""
        scene_df = pd.DataFrame()
        result = self.get_still_patients(scene_df)
        self.assertEqual(result, [])

    def test_various_still_cases(self):
        """Test with various scenarios for patients marked as 'still'."""
        data = {
            'patient_id': [1, 2, 2, 3],
            'column1': ['still', 'moving', 'still', 'still'],
            'column2': ['moving', 'still', 'moving', 'still']
        }
        scene_df = pd.DataFrame(data)
        result = self.get_still_patients(scene_df)
        self.assertEqual(result, [1, 2, 3])

    def test_verbose_output(self):
        """Test with verbose output enabled."""
        # ... (setup data for verbose output test)
        result = self.get_still_patients(scene_df, verbose=True)
        # ... (assert expected output, including printed messages)

# ... (other test cases as needed)

class TestGetTotalActions(unittest.TestCase):

    def setUp(self):
        # Sample scene DataFrame
        self.scene_df = pd.DataFrame({
            "action_type": ["MOVE", "VOICE_COMMAND", "LOOK", "VOICE_COMMAND"],
            "voice_command_message": ["", "hello", "", "open door"]
        })

        # Example action and command lists
        self.action_types_list = ["MOVE", "LOOK"]
        self.command_messages_list = ["hello", "open door"]

    def test_get_total_actions_default(self):
        # Test with default verbose (False)
        total_actions = self.get_total_actions(self.scene_df)
        self.assertEqual(total_actions, 2)

    def test_get_total_actions_verbose(self):
        # Test with verbose set to True
        captured_output = StringIO()
        sys.stdout = captured_output
        total_actions = self.get_total_actions(self.scene_df, verbose=True)
        sys.stdout = sys.__stdout__  # Restore original stdout

        self.assertEqual(total_actions, 2)
        self.assertIn("Total number of user actions: 2", captured_output.getvalue())

    def test_get_total_actions_with_voice_command(self):
        # Test including specific voice commands
        total_actions = self.get_total_actions(self.scene_df)
        self.assertEqual(total_actions, 3)

class TestGetActualAndIdealSequences(unittest.TestCase):

    def setUp(self):
        
        # Sample scene dataframe
        self.scene_df = pd.DataFrame({
            'patient_sort': ['waver', 'waver', 'still', 'waver', 'waver', 'walker'],
            'patient_id': ['Gloria_6 Root', 'Lily_2 Root', 'Gary_3 Root', 'Mike_5 Root', 'Lily_4 Root', 'Gloria_8 Root'],
            'action_type': ['PATIENT_ENGAGED', 'PATIENT_ENGAGED', 'PATIENT_ENGAGED', 'PATIENT_ENGAGED', 'PATIENT_ENGAGED', 'PATIENT_ENGAGED'],
            'action_tick': [384722, 409276, 336847, 438270, 607365, 346066],
        })

    def test_get_actual_and_ideal_sequences(self):
        actual_sequence, ideal_sequence, sort_dict = fu.get_actual_and_ideal_sequences(self.scene_df)
        
        # Expected results
        expected_actual_sequence = pd.Series(data=[336847, 346066, 384722, 409276, 438270, 607365], index=[0, 5, 1, 2, 3, 4])
        expected_ideal_sequence = pd.Series(data=[336847, 384722, 409276, 438270, 607365, 346066])
        expected_sort_dict = {'still': [336847], 'walker': [346066], 'waver': [384722, 409276, 438270, 607365]}

        # Assert the results
        self.assertTrue(expected_actual_sequence.equals(actual_sequence))
        self.assertTrue(expected_ideal_sequence.equals(ideal_sequence))
        self.assertEqual(expected_sort_dict, sort_dict)

class TestGetMeasureOfRightOrdering(unittest.TestCase):

    @patch('choropleth_utils.get_actual_and_ideal_sequences')  # Patch the dependency
    def test_empty_dataframe(self, mock_get_sequences):
        # Mock the return values of the dependency
        mock_get_sequences.return_value = pd.Series(), pd.Series(), None

        # Call the function with an empty DataFrame
        empty_df = pd.DataFrame()
        measure = get_measure_of_right_ordering(self, empty_df)

        # Assert the expected behavior
        self.assertEqual(measure, np.nan)
        self.assertFalse(mock_get_sequences.called)

    @patch('choropleth_utils.get_actual_and_ideal_sequences')
    def test_successful_calculation(self, mock_get_sequences):
        # Mock the return values of the dependency
        ideal_sequence = pd.Series([1, 2, 3, 4])
        actual_sequence = pd.Series([2, 1, 4, 3])
        mock_get_sequences.return_value = actual_sequence, ideal_sequence, None

        # Call the function with a sample DataFrame
        data = {'SORT_category': [1, 2, 3, 4], 'elapsed_time': [10, 5, 20, 15]}
        df = pd.DataFrame(data)
        measure = get_measure_of_right_ordering(self, df)

        # Assert the expected behavior
        self.assertAlmostEqual(measure, 0.75)  # Assuming some calculated R-squared adjusted value
        mock_get_sequences.assert_called_once_with(df, False)

    @patch('choropleth_utils.get_actual_and_ideal_sequences')
    def test_exception_handling(self, mock_get_sequences):
        # Mock the dependency to raise an exception
        mock_get_sequences.side_effect = Exception("Error during sequence generation")

        # Call the function and expect an exception
        df = pd.DataFrame({'SORT_category': [1], 'elapsed_time': [10]})
        with self.assertRaises(Exception) as cm:
            get_measure_of_right_ordering(self, df)

        # Assert the expected exception message
        self.assertEqual(str(cm.exception), "Error during sequence generation")
        mock_get_sequences.assert_called_once_with(df, False)

class TestGetPercentHemorrhageControlled(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create sample data (adjust values and columns as needed)
        data = {
            'injury_record_required_procedure': ['Procedure A', 'Procedure B', 'Procedure A', 'Hemorrhage Control 1', 'Hemorrhage Control 2'],
            'injury_treated_required_procedure': ['Procedure A', '', 'Procedure C', 'Hemorrhage Control 2', 'Hemorrhage Control 2 (Wrong)'],
            'injury_treated_injury_treated_with_wrong_treatment': [False, True, False, False, True]
        }
        cls.scene_df = pd.DataFrame(data)
        cls.hemorrhage_control_procedures_list = ['Hemorrhage Control 1', 'Hemorrhage Control 2']

    def test_all_controlled(self):
        """
        Tests the function with all hemorrhage cases controlled.
        """
        percent_controlled = self.scene_df.apply(
            lambda row: self.get_percent_hemorrhage_controlled(row, verbose=False), axis=1
        ).iloc[0]
        self.assertEqual(percent_controlled, 100.0)

    def test_no_hemorrhage_cases(self):
        """
        Tests the function with no hemorrhage cases.
        """
        filtered_df = self.scene_df[~self.scene_df['injury_record_required_procedure'].isin(
            self.hemorrhage_control_procedures_list)]
        percent_controlled = self.get_percent_hemorrhage_controlled(filtered_df, verbose=False)
        self.assertTrue(pd.isna(percent_controlled))

    def test_wrong_treatment(self):
        """
        Tests the function with a case where a hemorrhage control treatment is marked as wrong.
        """
        percent_controlled = self.get_percent_hemorrhage_controlled(self.scene_df, verbose=False)
        self.assertEqual(percent_controlled, 50.0)

    def test_duplicate_treatments(self):
        """
        Tests the function with duplicate treatment entries for the same injury.
        """
        # Create additional row with duplicate treatment
        duplicate_df = self.scene_df.append(self.scene_df.iloc[3])
        percent_controlled = self.get_percent_hemorrhage_controlled(duplicate_df, verbose=False)
        self.assertEqual(percent_controlled, 50.0)

class TestGetTimeToLastHemorrhageControlled(unittest.TestCase):

    def setUp(self):
        
        # Create mock data for the scene DataFrame
        self.scene_df = pd.DataFrame({
            "patient_id": [1, 1, 2],
            "hemorrhage_control_procedures_list": [["a", "b"], ["c"], ["d", "e"]]
        })

    def test_no_hemorrhage(self):
        # Mock the is_patient_hemorrhaging function to return False for all patients
        self.patch_is_patient_hemorrhaging(return_value=False)

        # Call the function with verbose set to False
        last_controlled_time = self.get_time_to_last_hemorrhage_controlled(self.scene_df, verbose=False)

        # Assert that the returned time is 0
        self.assertEqual(last_controlled_time, 0)

    def test_single_patient_hemorrhage(self):
        # Mock the get_time_to_hemorrhage_control function to return a specific value
        self.patch_get_time_to_hemorrhage_control(return_value=1000)

        # Mock the is_patient_hemorrhaging function to return True for the first patient only
        def is_patient_hemorrhaging(patient_df):
            return patient_df["patient_id"].iloc[0] == 1

        self.patch_is_patient_hemorrhaging(side_effect=is_patient_hemorrhaging)

        # Call the function with verbose set to False
        last_controlled_time = self.get_time_to_last_hemorrhage_controlled(self.scene_df, verbose=False)

        # Assert that the returned time is the controlled time from the mocked function
        self.assertEqual(last_controlled_time, 1000)

    def test_multiple_patients_hemorrhage(self):
        # Mock the get_time_to_hemorrhage_control function to return different values for each patient
        controlled_times = [500, 1200]
        def get_time_to_hemorrhage_control(patient_df, scene_start, verbose):
            patient_id = patient_df["patient_id"].iloc[0]
            return controlled_times[patient_id - 1]

        self.patch_get_time_to_hemorrhage_control(side_effect=get_time_to_hemorrhage_control)

        # Mock the is_patient_hemorrhaging function to return True for all patients
        self.patch_is_patient_hemorrhaging(return_value=True)

        # Call the function with verbose set to False
        last_controlled_time = self.get_time_to_last_hemorrhage_controlled(self.scene_df, verbose=False)

        # Assert that the returned time is the maximum controlled time from the mocked function
        self.assertEqual(last_controlled_time, max(controlled_times))

    # Helper methods for patching functions
    def patch_is_patient_hemorrhaging(self, **kwargs):
        patcher = patch.object(self, "is_patient_hemorrhaging", **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def patch_get_time_to_hemorrhage_control(self, **kwargs):
        patcher = patch.object(self, "get_time_to_hemorrhage_control", **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

class TestGetTriagePriorityDataFrame(unittest.TestCase):

    def setUp(self):
        # Create sample scene_df
        self.scene_df = pd.DataFrame({
            'injury_id': [1, 2, 3, 4, 5],
            'injury_severity': ['Minor', 'Critical', 'Moderate', 'Critical', 'Severe'],
            'injury_required_procedure': [False, True, False, True, True],
            # ... other columns
        })
        self.scene_groupby_columns = ['scene_id']  # Example groupby column

    def test_get_triage_priority_data_frame(self):
        # Test with default parameters
        triage_df = self.get_triage_priority_data_frame(self.scene_df)

        # Assert expected behavior
        self.assertEqual(list(triage_df.columns), self.scene_groupby_columns + ['injury_id', 'injury_severity', 'injury_required_procedure', 'patient_salt', 'patient_sort', 'patient_pulse', 'patient_breath', 'patient_hearing', 'patient_mood', 'patient_pose'])
        self.assertTrue(triage_df['injury_severity'].iloc[0] == 'Critical')  # Check sorting by injury severity
        self.assertTrue(triage_df['patient_sort'].iloc[0] == 1)  # Check sorting by patient_sort within severity

    def test_get_triage_priority_data_frame_verbose(self):
        # Test with verbose set to True
        triage_df = self.get_triage_priority_data_frame(self.scene_df, verbose=True)

        # No additional assertions needed, just testing verbose behavior

class TestGetEngagementStartsOrder(unittest.TestCase):

    def setUp(self):
        # Sample scene data
        self.scene_data = {
            "patient_id": [1, 1, 2, 3, 3],
            "action_type": ["MOVE", "TALK", "TALK", "TREAT", "MOVE"],
            "action_tick": [100, 200, 150, 300, 400],
            "location_id": ["(1,2,3)", None, "(4,5,6)", None, "(7,8,9)"],
            "patient_sort": ["RED", None, "GREEN", None, "BLUE"],
            "dtr_triage_priority_model_prediction": [0.7, None, 0.5, None, 0.9],
            # Add a column for injury severity (replace with your actual implementation)
            "injury_severity": ["Critical", "Minor", "Moderate", "Critical", "Minor"]
        }
        
        # Define responder negotiation list
        self.responder_negotiations_list = ["TALK", "TREAT"]

        # Convert data to a DataFrame
        self.scene_df = pd.DataFrame(self.scene_data)

    def test_empty_dataframe(self):
        # Test with an empty scene DataFrame
        empty_df = pd.DataFrame(columns=self.scene_data.keys())
        engagement_order = self.get_engagement_starts_order(empty_df)
        self.assertEqual(engagement_order, [])

    def test_no_responder_interaction(self):
        # Test with a patient with no responder interactions
        patient_df = self.scene_df[self.scene_df["patient_id"] == 2]
        engagement_order = self.get_engagement_starts_order(patient_df)
        self.assertEqual(engagement_order, [])

    def test_single_engagement(self):
        # Test with a single engagement
        patient_df = self.scene_df[self.scene_df["patient_id"] == 3]
        engagement_order = self.get_engagement_starts_order(patient_df)
        expected_order = [(3, 300, (7, 8, 9), "BLUE", 0.9, "Critical")]
        self.assertEqual(engagement_order, expected_order)

    def test_multiple_engagements(self):
        # Test with multiple engagements for a patient
        patient_df = self.scene_df[self.scene_df["patient_id"] == 1]
        engagement_order = self.get_engagement_starts_order(patient_df)
        expected_order = [
            (1, 200, (1, 2, 3), "RED", None, "Critical"),
            (1, 300, (0.0, 0.0), None, None, "Critical"),
        ]
        self.assertEqual(engagement_order, expected_order)

    def test_missing_location(self):
        # Test with a case where the first engagement has a missing location
        self.scene_data["location_id"][1] = None
        self.scene_df = pd.DataFrame(self.scene_data)
        patient_df = self.scene_df[self.scene_df["patient_id"] == 1]
        engagement_order = self.get_engagement_starts_order(patient_df)
        expected_order = [
            (1, 100, (0.0, 0.0), None, 0.7, "Critical"),
            (1, 300, (0.0, 0.0), None, None, "Critical"),
        ]
        self.assertEqual(engagement_order, expected_order)

class TestGetDistractedEngagementOrder(unittest.TestCase):

    def setUp(self):
        self.mock_get_engagement_starts_order = MagicMock()
        self.mock_scene_df = MagicMock()

    @patch("distracted_engagement_utils.nu.get_nearest_neighbor")
    def test_get_distracted_engagement_order_with_tuples_list(self, mock_get_nearest_neighbor):
        # Mock data
        mock_tuples_list = [
            (1, 2, (1.0, 2.0)),
            (2, 3, (3.0, 4.0)),
            (3, 4, (2.0, 3.0)),
        ]
        mock_player_location = (1.0, 2.0)
        mock_nearest_neighbor = (2.0, 3.0)
        mock_get_nearest_neighbor.side_effect = [mock_nearest_neighbor]

        # Mock return values
        self.mock_get_engagement_starts_order.return_value = mock_tuples_list
        self.mock_scene_df.__getitem__.return_value = self.mock_scene_df

        # Call the function
        distracted_order = get_distracted_engagement_order(self, self.mock_scene_df, mock_tuples_list, verbose=True)

        # Expected output
        expected_order = [mock_tuples_list[2], mock_tuples_list[0]]

        # Assert calls and results
        self.mock_get_engagement_starts_order.assert_called_once_with(self.mock_scene_df, verbose=True)
        self.assertEqual(distracted_order, expected_order)

    @patch("distracted_engagement_utils.nu.get_nearest_neighbor")
    def test_get_distracted_engagement_order_no_tuples_list(self, mock_get_nearest_neighbor):
        # Mock data
        mock_player_location = (1.0, 2.0)
        mock_nearest_neighbor = (2.0, 3.0)
        mock_get_nearest_neighbor.side_effect = [mock_nearest_neighbor]

        # Mock return values
        self.mock_get_engagement_starts_order.return_value = []
        self.mock_scene_df.__getitem__.return_value = self.mock_scene_df

        # Call the function
        distracted_order = get_distracted_engagement_order(self, self.mock_scene_df, verbose=True)

        # Expected output (empty list)
        expected_order = []

        # Assert calls and results
        self.mock_get_engagement_starts_order.assert_called_once_with(self.mock_scene_df, verbose=True)
        self.assertEqual(distracted_order, expected_order)

if __name__ == "__main__":
    unittest.main()