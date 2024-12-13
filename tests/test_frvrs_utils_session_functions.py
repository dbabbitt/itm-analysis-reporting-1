
from datetime import timedelta
from unittest.mock import patch, MagicMock
import numpy as np
import os
import pandas as pd
import unittest

# Import the class containing the functions
import sys
if (osp.join(os.pardir, 'py') not in sys.path): sys.path.insert(1, osp.join(os.pardir, 'py'))
from FRVRS import fu, nu


### Session Functions ###


class TestGetSessionGroupby(unittest.TestCase):

    # Define sample DataFrame for testing
    grouped_df = pd.DataFrame({
        'session_uuid': [1, 1, 2, 2, 3],
        'action_tick': [10, 20, 30, 40, 50],
        'other_column': ['A', 'B', 'A', 'B', 'A']
    })

    def test_groupby_no_mask_no_extra_column(self):
        gb = get_session_groupby(self.grouped_df)
        expected_groups = self.grouped_df.groupby('session_uuid')
        self.assertEqual(gb.groups, expected_groups.groups)

    def test_groupby_no_mask_with_extra_column(self):
        extra_column = 'other_column'
        gb = get_session_groupby(self.grouped_df, extra_column=extra_column)
        expected_groups = self.grouped_df.groupby(['session_uuid', extra_column])
        self.assertEqual(gb.groups, expected_groups.groups)

    def test_groupby_with_mask_no_extra_column(self):
        mask_series = self.grouped_df['action_tick'] > 25
        gb = get_session_groupby(self.grouped_df, mask_series=mask_series)
        expected_groups = self.grouped_df[mask_series].groupby('session_uuid')
        self.assertEqual(gb.groups, expected_groups.groups)

    def test_groupby_with_mask_with_extra_column(self):
        mask_series = self.grouped_df['action_tick'] > 25
        extra_column = 'other_column'
        gb = get_session_groupby(self.grouped_df, mask_series=mask_series, extra_column=extra_column)
        expected_groups = self.grouped_df[mask_series].groupby(['session_uuid', extra_column])
        self.assertEqual(gb.groups, expected_groups.groups)

class TestGetIsAOneTriageFile(unittest.TestCase):

    def test_single_triage_run(self):
        # Create a DataFrame with a single triage run
        session_df = pd.DataFrame({
            'scene_type': ['Triage', 'Other'],
            'is_scene_aborted': [False, True],
            'file_name': ['file1.txt', 'file1.txt']
        })

        # Call the function and assert the result
        is_one_triage = get_is_a_one_triage_file(session_df.copy())
        self.assertTrue(is_one_triage)

    def test_multiple_triage_runs(self):
        # Create a DataFrame with multiple triage runs
        session_df = pd.DataFrame({
            'scene_type': ['Triage', 'Triage', 'Other'],
            'is_scene_aborted': [False, False, True],
            'file_name': ['file1.txt', 'file1.txt', 'file1.txt']
        })

        # Call the function and assert the result
        is_one_triage = get_is_a_one_triage_file(session_df.copy())
        self.assertFalse(is_one_triage)

    def test_column_exists_in_session_df(self):
        # Create a DataFrame with the 'is_a_one_triage_file' column already present
        session_df = pd.DataFrame({
            'is_a_one_triage_file': [True],
            'scene_type': ['Triage'],
            'is_scene_aborted': [False],
            'file_name': ['file1.txt']
        })

        # Call the function and assert it reads the value from the column
        is_one_triage = get_is_a_one_triage_file(session_df.copy())
        self.assertTrue(is_one_triage)

    def test_missing_columns_merged_correctly(self):
        # Create DataFrames for logs_df, file_stats_df, and scene_stats_df with necessary columns
        # ... (code for creating these DataFrames)

        # Create a session_df with missing columns
        session_df = pd.DataFrame({
            'scene_type': ['Triage', 'Other'],
            'is_scene_aborted': [False, True],
            'file_name': ['file1.txt', 'file1.txt']
        })

        # Call the function with logs_df, file_stats_df, and scene_stats_df
        is_one_triage = get_is_a_one_triage_file(session_df.copy(), logs_df=logs_df,
                                                    file_stats_df=file_stats_df, scene_stats_df=scene_stats_df)
        self.assertEqual(is_one_triage, True)  # Assert the result based on the merged DataFrame

class TestGetFile(unittest.TestCase):

    # Define a sample DataFrame with a unique file name
    test_df = pd.DataFrame({'file_name': ['test_file.txt']})

    def test_get_file_name_unique(self):
        """
        Tests that the correct file name is returned when a unique file name exists.
        """
        file_name = self.test_df.__class__.get_file_name(self.test_df)
        self.assertEqual(file_name, 'test_file.txt')

    def test_get_file_name_verbose(self):
        """
        Tests that the verbose output is printed when verbose=True.
        """
        with self.assertLogs() as captured:
            self.test_df.__class__.get_file_name(self.test_df, verbose=True)
        self.assertEqual(captured.records[0].getMessage(), 'File name: test_file.txt')

    def test_get_file_name_multiple_values(self):
        """
        Tests that an error is raised when the DataFrame has multiple file names.
        """
        multiple_file_df = pd.DataFrame({'file_name': ['test1.txt', 'test2.txt']})
        with self.assertRaises(ValueError) as cm:
            multiple_file_df.__class__.get_file_name(multiple_file_df)
        self.assertEqual(str(cm.exception), 'DataFrame contains multiple file names.')

class TestGetLoggerVersion(unittest.TestCase):

   @patch("builtins.open")
   def test_get_logger_version_from_file_1_3(self, mock_open):
       mock_file = mock_open.return_value.__enter__.return_value
       mock_file.read.return_value = 'data with version,1.3,'

       result = your_class_name.get_logger_version("test_file.txt")

       self.assertEqual(result, "1.3")

   @patch("builtins.open")
   def test_get_logger_version_from_file_1_0(self, mock_open):
       mock_file = mock_open.return_value.__enter__.return_value
       mock_file.read.return_value = 'data with version,1.0,'

       result = your_class_name.get_logger_version("test_file.txt")

       self.assertEqual(result, "1.0")

   def test_get_logger_version_from_dataframe(self):
       mock_df = pd.DataFrame({"logger_version": ["1.3"]})

       result = your_class_name.get_logger_version(mock_df)

       self.assertEqual(result, "1.3")

   @patch("sys.stdout", new_callable=io.StringIO)
   def test_verbose_output(self, mock_stdout):
       your_class_name.get_logger_version("test_file.txt", verbose=True)

       self.assertIn("Logger version: 1.0", mock_stdout.getvalue())

class TestGetIsDuplicateFile(unittest.TestCase):

    def test_duplicate_file(self):
        """Test for a DataFrame with multiple unique file names"""
        data = {'session_uuid': ['uuid1', 'uuid1', 'uuid1'],
                'file_name': ['file1.txt', 'file1.txt', 'file2.txt']}
        session_df = pd.DataFrame(data)

        result = fu.get_is_duplicate_file(session_df)
        self.assertTrue(result)

    def test_single_file(self):
        """Test for a DataFrame with a single unique file name"""
        data = {'session_uuid': ['uuid1', 'uuid1'],
                'file_name': ['file1.txt', 'file1.txt']}
        session_df = pd.DataFrame(data)

        result = fu.get_is_duplicate_file(session_df)
        self.assertFalse(result)

    def test_empty_dataframe(self):
        """Test for an empty DataFrame"""
        session_df = pd.DataFrame()

        result = fu.get_is_duplicate_file(session_df)
        self.assertFalse(result)

class TestGetSessionFileDate(unittest.TestCase):

    def test_get_file_date_from_logs_df(self):
        # Create a sample logs DataFrame
        logs_df = pd.DataFrame({
            'session_uuid': ['uuid1', 'uuid1', 'uuid2'],
            'event_time': pd.to_datetime(['2023-11-21 10:00', '2023-11-21 11:00', '2023-11-22 12:00'])
        })
        uuid = 'uuid1'

        file_date_str = get_session_file_date(logs_df, uuid)

        self.assertEqual(file_date_str, 'November 21, 2023')

    def test_get_file_date_from_session_df(self):
        # Create a sample session DataFrame
        session_df = pd.DataFrame({
            'event_time': pd.to_datetime(['2023-12-05 09:00', '2023-12-05 10:00'])
        })
        uuid = 'uuid3'  # Doesn't matter for this test

        file_date_str = get_session_file_date(None, uuid, session_df=session_df)

        self.assertEqual(file_date_str, 'December 05, 2023')

class TestGetDistanceDeltasDataFrame(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.logs_df = pd.DataFrame({
            'session_uuid': ['uuid1', 'uuid1', 'uuid2', 'uuid2'],
            'scene_id': [1, 1, 2, 2],
            'patient_id': [1, 2, 3, 4],
            'engagement_start': [10, 20, 30, 40],
            'location_tuple': [('A', 1, 1), ('B', 2, 2), ('C', 3, 3), ('D', 4, 4)],
            'patient_sort': ['moving', 'still', 'moving', 'distracted']
        })
        self.scene_groupby_columns = ['session_uuid', 'scene_id']

        # Mock functions (replace with actual implementations in your tests)
        def mock_get_engagement_starts_order(self, df, verbose=False):
            # Implement your logic to return the engagement order
            return [('A', 10), ('B', 20), ('C', 30)]

        def mock_get_order_of_ideal_engagement(self, df, verbose=False):
            # Implement your logic to return the ideal order
            return [('C', 30), ('A', 10), ('B', 20)]

        def mock_get_order_of_distracted_engagement(self, df, tuples_list=None, verbose=False):
            # Implement your logic to return the distracted order
            return [('B', 20), ('A', 10), ('C', 30)]

        def mock_get_scene_start(self, df):
            # Implement your logic to get the scene start time
            return 0

        def mock_get_measure_of_right_ordering(self, df, verbose=False):
            # Implement your logic to get the measure of right ordering
            return 0.5

        self.get_order_of_actual_engagement = mock_get_engagement_starts_order
        self.get_order_of_ideal_engagement = mock_get_order_of_ideal_engagement
        self.get_order_of_distracted_engagement = mock_get_order_of_distracted_engagement
        self.get_scene_start = mock_get_scene_start
        self.get_measure_of_right_ordering = mock_get_measure_of_right_ordering

    def test_get_distance_deltas_dataframe(self):
        # Call the function with the mock functions
        distance_delta_df = self.get_distance_deltas_dataframe(self.logs_df)

        # Assertions (modify these based on your expected output)
        self.assertEqual(distance_delta_df.shape, (2, 12))
        self.assertAlmostEqual(distance_delta_df['actual_engagement_distance'].iloc[0], 2.8284, places=4)
        self.assertAlmostEqual(distance_delta_df['measure_of_ideal_ordering'].iloc[0], 0.5, places=4)
        self.assertTrue(distance_delta_df['adherence_to_salt'].iloc[0] == False)

if __name__ == "__main__":
    unittest.main()