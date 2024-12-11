
from contextlib import redirect_stdout
from datetime import timedelta
from numpy import nan
from pandas import DataFrame
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import numpy as np
import os
import pandas as pd
import unittest

# Import the class containing the functions
import sys
if (osp.join('..', 'py') not in sys.path): sys.path.insert(1, osp.join('..', 'py'))
from FRVRS import fu, nu


### Pandas Functions ###


class TestGetStatistics(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame
        self.describable_df = DataFrame({
            'triage_time': [461207, 615663, 649185, 19615, 488626],
            'last_controlled_time': [377263, 574558, 462280, 0, 321956]
        })
        self.expected_df = DataFrame(
            {
                'triage_time': [446859.2, 19615.0, 488626.0, 251951.57452415334, 19615.0, 461207.0, 488626.0, 615663.0, 649185.0],
                'last_controlled_time': [347211.4, 0.0, 377263.0, 216231.3283379631, 0.0, 321956.0, 377263.0, 462280.0, 574558.0]
            },
            index=['mean', 'mode', 'median', 'SD', 'min', '25%', '50%', '75%', 'max']
        )

    def test_get_statistics_all_columns(self):
        # Test with all columns
        columns_list = self.describable_df.columns.tolist()
        actual_df = nu.get_statistics(self.describable_df, columns_list)
        self.assertTrue(self.expected_df.equals(actual_df))

    def test_get_statistics_subset_columns(self):
        # Test with a subset of columns
        columns_list = ['triage_time']
        actual_df = nu.get_statistics(self.describable_df, columns_list)
        self.assertTrue(self.expected_df.drop(columns=['last_controlled_time']).equals(actual_df))


class TestShowTimeStatistics(unittest.TestCase):

    def setUp(self):
        # Create sample data
        data = {'date_col': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'time_col': ['00:00:01', '00:00:02', '00:00:03']}
        self.describable_df = pd.DataFrame(data)
        self.columns_list = ['date_col', 'time_col']

    # Mock methods for get_statistics and format_timedelta_lambda
    @staticmethod
    def mock_get_statistics(df, columns):
        # Simulate get_statistics function to return a DataFrame with specific values
        return pd.DataFrame({'mean': ['00:00:02', pd.Timedelta('0 days 00:00:01')],
                             'mode': ['00:00:01', '00:00:01'],
                             'median': ['00:00:02', pd.Timedelta('0 days 00:00:01')],
                             'std': ['0 days 00:00:00.5773502691896257', '0 days 00:00:00.5773502691896257']},
                            index=columns)

    @staticmethod
    def mock_format_timedelta_lambda(td):
        # Simulate format_timedelta_lambda function to return a formatted string
        return td.strftime('%H:%M:%S')

    def test_show_time_statistics(self):
        # Patch get_statistics and format_timedelta_lambda methods with mocks
        with unittest.mock.patch.object(fu, 'show_time_statistics.get_statistics', self.mock_get_statistics), \
             unittest.mock.patch.object(fu, 'show_time_statistics.format_timedelta_lambda', self.mock_format_timedelta_lambda):

            # Call the function
            self.show_time_statistics(self.describable_df, self.columns_list)

            # Assertions (modify as needed based on expected output)
            # ...


class TestSetSceneIndices(unittest.TestCase):

    def setUp(self):
        # Sample data with session starts and ends
        data = {
            "action_tick": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "action_type": ["MOVE", "ATTACK", "SESSION_START", "HEAL", "MOVE", "SESSION_END", "ATTACK", "DEFEND", "SESSION_START", "MOVE", "ATTACK", "SESSION_END", "CAST_SPELL"]
        }
        self.file_df = pd.DataFrame(data)
        self.file_df = self.file_df.reset_index(drop=True)

    def test_set_scene_indices(self):
        expected_df = self.file_df.copy()
        expected_df["scene_id"] = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]
        expected_df["is_scene_aborted"] = [False, False, False, False, False, False, False, False, False, False, False, False, True]

        actual_df = fu.set_scene_indices(self.file_df.copy())

        # Assert that DataFrames are equal
        pd.testing.assert_frame_equal(expected_df, actual_df)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["action_tick", "action_type"])
        expected_df = empty_df.copy()
        expected_df["scene_id"] = []
        expected_df["is_scene_aborted"] = []

        actual_df = fu.set_scene_indices(empty_df.copy())

        # Assert that DataFrames are equal
        pd.testing.assert_frame_equal(expected_df, actual_df)

class TestSetMcivrMetricsTypes(unittest.TestCase):

    def setUp(self):
        # Create an example DataFrame
        self.df = pd.DataFrame({
            "action_type": ["BAG_ACCESS", "INJURY_RECORD", "TOOL_APPLIED", "PLAYER_LOCATION"],
            # Add more rows and columns as needed for your test cases
            "row_series_1": [
                ["BAG_ACCESS", 1, 2, 3, "location"],
                ["INJURY_RECORD", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                ["TOOL_APPLIED", 1, 2, 3, "patient_id", "tool_type", "attachment_point", "tool_location", "some message", "sender"],
                ["PLAYER_LOCATION", 1, 2, 3, "location", None, None]
            ]
        })

    def test_bag_access(self):
        # Test BAG_ACCESS action type
        expected_df = self.df.copy()
        expected_df.loc[0, "bag_access_location"] = "location"
        actual_df = self.df.apply(lambda row: fu.set_mcivr_metrics_types(row["action_type"], self.df, row.name, row), axis=1)
        self.assertEqual(actual_df.to_string(), expected_df.to_string())

    def test_injury_record(self):
        # Test INJURY_RECORD action type
        expected_df = self.df.copy()
        expected_df.loc[1, "injury_record_id"] = 1
        expected_df.loc[1, "injury_record_patient_id"] = 2
        # ... add expected values for all INJURY_RECORD columns
        actual_df = self.df.apply(lambda row: fu.set_mcivr_metrics_types(row["action_type"], self.df, row.name, row), axis=1)
        self.assertEqual(actual_df.to_string(), expected_df.to_string())

    def test_tool_applied(self):
        # Test TOOL_APPLIED action type
        expected_df = self.df.copy()
        expected_df.loc[2, "tool_applied_patient_id"] = "patient_id"
        expected_df.loc[2, "tool_applied_type"] = "tool_type"
        # ... add expected values for all TOOL_APPLIED columns
        actual_df = self.df.apply(lambda row: fu.set_mcivr_metrics_types(row["action_type"], self.df, row.name, row), axis=1)
        self.assertEqual(actual_df.to_string(), expected_df.to_string())

    def test_player_location(self):
        # Test PLAYER_LOCATION action type
        expected_df = self.df.copy()
        expected_df.loc[3, "player_location_location"] = "location"
        actual_df = self.df.apply(lambda row: fu.set_mcivr_metrics_types(row["action_type"], self.df, row.name, row), axis=1)
        self.assertEqual(actual_df.to_string(), expected_df.to_string())

    # Add more test cases for other action types as needed

class TestProcessFiles(unittest.TestCase):

    def setUp(self):
        # Mock data for testing
        self.sub_directory_df = pd.DataFrame(columns=['col1', 'col2'])
        self.sub_directory = '/path/to/subdirectory'
        self.file_name = 'test_file.csv'
        self.data_logs_folder = '/path/to/data/logs'

    def test_process_small_file(self):
        # Test case for ignoring small files
        small_file_df = pd.DataFrame(columns=['col1'])
        with self.assertRaises(AssertionError):
            fu.process_files(self.sub_directory_df, self.sub_directory, self.file_name, verbose=True)

    def test_process_csv_file(self):
        # Test case for successful CSV parsing
        expected_df = pd.DataFrame([[1, 2, 3, 'data'], [4, 5, 6, 'more data']], columns=list(range(4)))
        mock_file_path = os.path.join(self.sub_directory, self.file_name)
        with open(mock_file_path, 'w') as f:
            f.write(','.join([str(x) for x in list(range(4))]) + '\n')
            f.write(','.join(['data', 'more data']))
        result_df = fu.process_files(self.sub_directory_df.copy(), self.sub_directory, self.file_name)
        # Drop unnecessary columns for easier comparison
        result_df = result_df.drop(columns=['file_name', 'logger_version'])
        expected_df = expected_df.drop(columns=['file_name', 'logger_version'])
        self.assertTrue(result_df.equals(expected_df))

    def test_process_non_csv_file(self):
        # Test case for handling non-csv files
        mock_file_path = os.path.join(self.sub_directory, self.file_name)
        with open(mock_file_path, 'w') as f:
            f.write('This is not a CSV file')
        result_df = fu.process_files(self.sub_directory_df.copy(), self.sub_directory, self.file_name)
        self.assertEqual(result_df.shape[0], 0)

    # Add more test cases for different functionalities of the process_files function

class TestConcatonateLogs(unittest.TestCase):

   def setUp(self):
       # Set up any necessary test fixtures, such as creating temporary files or folders
       pass

   @patch('os.walk')  # Mock the os.walk function for controlled testing
   def test_concatonate_logs_default_folder(self, mock_walk):
       mock_walk.return_value = [
           ('logs_folder', ['subdir1', 'subdir2'], ['file1.csv', 'file2.csv']),
           ('logs_folder/subdir1', [], []),
           ('logs_folder/subdir2', [], ['file3.csv'])
       ]
       fu.data_logs_folder = 'logs_folder'  # Set the default logs folder
       
       with patch.object(fu, 'process_files') as mock_process_files:
           mock_process_files.side_effect = [pd.DataFrame({'data1': [1]}), pd.DataFrame({'data2': [2]}), pd.DataFrame({'data3': [3]})]
           
           result = fu.concatonate_logs()

       expected_df = pd.DataFrame({'data1': [1], 'data2': [2], 'data3': [3]})
       pd.testing.assert_frame_equal(result, expected_df)

   # Add more test cases to cover different scenarios and edge cases, such as:
   # - Specific logs folder
   # - Empty logs folder
   # - Different file extensions
   # - Data validation
   # - Error handling

class TestSplitDfByTeleport(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame
        self.df = pd.DataFrame({'col1': [1, 2, 3, 4, 5],
                                'col2': ['a', 'b', 'c', 'd', 'e']})
        # Sample teleport locations
        self.teleport_rows = [2, 4]

    def test_split_df_by_teleport_empty_df(self):
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = fu.split_df_by_indices(empty_df)
        self.assertEqual(result, [])

    def test_split_df_by_teleport_no_teleports(self):
        # Test with no teleport locations
        result = fu.split_df_by_indices(self.df.copy())
        self.assertEqual(result, [self.df.copy()])

    def test_split_df_by_teleport_single_teleport(self):
        # Test with single teleport location
        result = fu.split_df_by_indices(self.df.copy())
        expected_dfs = [self.df.iloc[:2, :], self.df.iloc[2:, :]]
        self.assertEqual(result, expected_dfs)

    def test_split_df_by_teleport_multiple_teleports(self):
        # Test with multiple teleport locations
        result = fu.split_df_by_indices(self.df.copy())
        expected_dfs = [self.df.iloc[:2, :], self.df.iloc[2:4, :], self.df.iloc[4:, :]]
        self.assertEqual(result, expected_dfs)

class TestShowLongRuns(unittest.TestCase):

    def setUp(self):
        # Sample data
        self.data = {
            "session_uuid": ["uuid1", "uuid1", "uuid2", "uuid3"],
            "column_name": [1000, 5000, 2000, 8000],
            "file_name": ["file1.csv", "file1.csv", "file2.csv", "file3.csv"]
        }
        self.df = pd.DataFrame(self.data)
        self.logs_df = pd.DataFrame({"session_uuid": ["uuid2", "uuid3"], "file_name": ["file2.csv", "file3.csv"]})

        # Mock functions
        self.mock_delta_fn = lambda x: pd.Timedelta(milliseconds=x)

    def test_show_long_runs_greater(self):
        # Test case for files with duration greater than threshold
        description = "longer"
        milliseconds = 3000
        expected_output = f"\nThese files have {description} than {self.mock_delta_fn(milliseconds)}:\nfile1.csv (or file1.csv)\n"

        # Capture output in a StringIO object to simulate printing
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output

        # Call the function
        show_long_runs.show_long_runs(self.df, "column_name", milliseconds, self.mock_delta_fn, description, self.logs_df)

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Assert the captured output
        self.assertEqual(captured_output.getvalue(), expected_output)

    def test_show_long_runs_empty(self):
        # Test case for no files exceeding threshold
        description = "shorter"
        milliseconds = 10000
        expected_output = f"\nThese files have {description} than {self.mock_delta_fn(milliseconds)}:\n"

        # Capture output in a StringIO object to simulate printing
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output

        # Call the function
        show_long_runs.show_long_runs(self.df, "column_name", milliseconds, self.mock_delta_fn, description, self.logs_df)

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Assert the captured output
        self.assertEqual(captured_output.getvalue(), expected_output)

    def test_show_long_runs_logs_df_none(self):
        # Test case with None for logs_df
        description = "longer"
        milliseconds = 2000
        expected_output = f"\nThese files have {description} than {self.mock_delta_fn(milliseconds)}:\n"

        # Capture output in a StringIO object to simulate printing
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output

        # Call the function
        show_long_runs.show_long_runs(self.df, "column_name", milliseconds, self.mock_delta_fn, description)

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Assert the captured output
        self.assertEqual(captured_output.getvalue(), expected_output)


class TestReplaceConsecutiveRows(unittest.TestCase):

    def test_replace_no_consecutive(self):
        """
        Tests the function on a DataFrame with no consecutive elements.
        """
        data = {'element': ['A', 'B', 'C', 'D', 'E'], 'time_diff': [1000, 500, 1000, 800, 1200]}
        df = pd.DataFrame(data)
        expected_df = df.copy()

        result_df = nu.replace_consecutive_rows(df.copy(), 'element', 'A', 'time_diff', 1000)

        self.assertTrue(result_df.equals(expected_df))

    def test_replace_consecutive_within_cutoff(self):
        """
        Tests the function on a DataFrame with consecutive elements within the cutoff.
        """
        data = {'element': ['A', 'A', 'A', 'B', 'C', 'A', 'A', 'D'], 'time_diff': [1000, 200, 300, 800, 1200, 400, 500, 1500]}
        df = pd.DataFrame(data)
        expected_df = pd.DataFrame({'element': ['A x3', 'B', 'C', 'A x2', 'D']})

        result_df = nu.replace_consecutive_rows(df.copy(), 'element', 'A', 'time_diff', 500)

        self.assertTrue(result_df.equals(expected_df))

    def test_replace_consecutive_outside_cutoff(self):
        """
        Tests the function on a DataFrame with consecutive elements outside the cutoff.
        """
        data = {'element': ['A', 'A', 'A', 'B', 'C', 'A', 'A', 'D'], 'time_diff': [1000, 600, 1200, 800, 1200, 1500, 2000, 1500]}
        df = pd.DataFrame(data)
        expected_df = df.copy()
        expected_df.loc[1:3, 'element'] = 'A'

        result_df = nu.replace_consecutive_rows(df.copy(), 'element', 'A', 'time_diff', 500)

        self.assertTrue(result_df.equals(expected_df))

    def test_replace_consecutive_last_element(self):
        """
        Tests the function on a DataFrame where the last element is consecutive.
        """
        data = {'element': ['A', 'B', 'C', 'A', 'A'], 'time_diff': [1000, 500, 1000, 800, 200]}
        df = pd.DataFrame(data)
        expected_df = pd.DataFrame({'element': ['A', 'B', 'C', 'A x2']})

        result_df = nu.replace_consecutive_rows(df.copy(), 'element', 'A', 'time_diff', 1000)

        self.assertTrue(result_df.equals(expected_df))

class TestGetElevensDataFrame(unittest.TestCase):

    def setUp(self):
        # Create sample DataFrames for testing
        data_frames_dict = nu.load_data_frames(
            verbose=False, first_responder_master_registry_df='', first_responder_master_registry_file_stats_df='', first_responder_master_registry_scene_stats_df=''
        )
        self.logs_df = data_frames_dict['first_responder_master_registry_df']
        self.file_stats_df = data_frames_dict['first_responder_master_registry_file_stats_df']
        self.scene_stats_df = data_frames_dict['first_responder_master_registry_scene_stats_df']

    def test_all_columns_in_logs_df(self):
        """
        Test case where all needed columns are present in the logs DataFrame.
        """
        needed_columns = ['session_uuid', 'scene_id']
        elevens_df = fu.get_elevens_dataframe(self.logs_df, self.file_stats_df, self.scene_stats_df, needed_columns)
        self.assertEqual(elevens_df.shape[1], self.logs_df.shape[1]+2)  # All rows from logs_df should be present

    def test_missing_columns_from_all_dfs(self):
        """
        Test case where a needed column is missing from all DataFrames.
        """
        needed_columns = ['missing_column']
        with self.assertRaises(KeyError):
            fu.get_elevens_dataframe(self.logs_df, self.file_stats_df, self.scene_stats_df, needed_columns)

    def test_missing_column_from_logs_df_only(self):
        """
        Test case where a needed column is missing only from the logs DataFrame.
        """
        needed_columns = ['scene_type', 'is_scene_aborted', 'is_a_one_triage_file', 'responder_category']
        elevens_df = fu.get_elevens_dataframe(self.logs_df, self.file_stats_df, self.scene_stats_df, needed_columns)
        self.assertEqual(elevens_df.shape[1], self.logs_df.shape[1] + len(needed_columns))  # Merged DataFrames with additional columns

    def test_empty_dataframes(self):
        """
        Test case where all DataFrames are empty.
        """
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            fu.get_elevens_dataframe(empty_df, empty_df, empty_df)

if __name__ == "__main__":
    unittest.main()