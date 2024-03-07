
from contextlib import redirect_stdout
from datetime import timedelta
from humanize import precisedelta
from matplotlib import pyplot as plt
from numpy import nan
from numpy.random import choice
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import unittest

# Import the class containing the functions
import sys
sys.path.insert(1, '../py')
from FRVRS import fu, nu


### Plotting Functions ###


class TestVisualizeOrderOfEngagement(unittest.TestCase):

    @patch('fu.plt.subplots')
    @patch('fu.nu.get_color_cycler')
    @patch('fu.humanize.ordinal')
    @patch('fu.get_wanderings')
    @patch('fu.get_wrapped_label')
    def test_fu.visualize_order_of_engagement(self, 
                                            mock_get_wrapped_label, 
                                            mock_get_wanderings, 
                                            mock_humanize_ordinal, 
                                            mock_nu_get_color_cycler, 
                                            mock_plt_subplots):
        # Mock dependencies
        mock_fig, mock_ax = mock_plt_subplots.return_value
        mock_get_color_cycler.return_value = iter(['red', 'blue', 'green'])
        mock_humanize_ordinal.return_value = 'Second'
        mock_get_wanderings.return_value = ([1, 2, 3], [4, 5, 6])
        mock_get_wrapped_label.return_value = "Patient A"

        # Create sample scene_df
        scene_df = MagicMock()
        scene_df.session_uuid = '1234'
        scene_df.scene_id = 1

        # Call the function
        fig, ax = fu.visualize_order_of_engagement(scene_df, verbose=True)

        # Assertions
        self.assertEqual(mock_plt_subplots.call_count, 1)
        self.assertEqual(mock_get_color_cycler.call_count, 1)
        self.assertEqual(mock_humanize_ordinal.call_count, 1)
        self.assertEqual(mock_get_wanderings.call_count, 1)
        self.assertEqual(mock_get_wrapped_label.call_count, 1)

        # Verify plot elements (using mock objects is more robust)
        mock_ax.plot.assert_called()
        mock_ax.scatter.assert_called()
        mock_ax.annotate.assert_called()
        
        # Verify additional assertions (modify as needed)
        # ...


class TestVisualizePlayerMovement(unittest.TestCase):

    @patch('fu.visualize_player_movement.osp')
    @patch('fu.visualize_player_movement.plt')
    @patch('fu.visualize_player_movement.nu')
    def test_fu.visualize_player_movement(self, mock_nu, mock_plt, mock_osp):
        # Mock dependencies
        mock_get_wanderings = MagicMock(return_value=(None, None))
        mock_get_wrapped_label = MagicMock(return_value="Patient X")
        self.patch_object(type('self', (object,), {}), 'get_wanderings', mock_get_wanderings)
        self.patch_object(type('self', (object,), {}), 'get_wrapped_label', mock_get_wrapped_label)

        # Sample data
        logs_df = pd.DataFrame({
            "patient_id": [1, 1, 2],
            "action_tick": [1, 2, 1],
            "action_type": ["PLAYER_MOVE", "PLAYER_MOVE", "PLAYER_MOVE"],
            # ... other columns
        })
        scene_mask = pd.Series([True, True, False])
        title = "Test Visualization"
        save_only = False
        verbose = False

        # Call the function
        fu.visualize_player_movement(logs_df, scene_mask, title, save_only, verbose)

        # Assertions
        # Assert calls to mocked methods
        mock_get_wanderings.assert_called_once_with(logs_df.iloc[0])
        mock_get_wrapped_label.assert_called_once_with(logs_df.iloc[0])

        # Assert plot configuration (modify as needed)
        mock_plt.subplots.assert_called_once_with(figsize=(18, 9))
        mock_ax = mock_plt.subplots.return_value.ax
        mock_ax.plot.assert_called_once()
        mock_ax.scatter.assert_called()
        mock_ax.set_xlabel.assert_called_once_with('X')
        mock_ax.set_ylabel.assert_called_once_with('Z')
        mock_ax.legend.assert_called_once()

        # Additional assertions based on your function's logic

    # Add more test cases for different scenarios (e.g., saving to file, different data)

# Mock required modules
humanize = MagicMock()

class TestVisualizeExtremePlayerMovement(unittest.TestCase):

    def setUp(self):
        # Sample DataFrames
        self.logs_df = pd.DataFrame({'session_uuid': [1, 1, 2], 'scene_id': [1, 2, 1]})
        self.df = pd.DataFrame({'session_uuid': [1, 1, 1, 2], 'scene_id': [1, 2, 1, 2], 'movement_time': [1000, 500, 2000, 1500]})
        self.mask_series = pd.Series([True, True, False, True])

    def test_empty_dataframe(self):
        # Patch visualize_player_movement
        with patch.object(self, 'visualize_extreme_player_movement.visualize_player_movement') as mock_visualize:
            # Empty DataFrame
            empty_df = pd.DataFrame(columns=['session_uuid', 'scene_id', 'movement_time'])
            fu.visualize_extreme_player_movement(empty_df, self.df, 'movement_time')
            # Assert visualize_player_movement not called
            mock_visualize.assert_not_called()

    def test_ascending_sort(self):
        # Mock humanize.precisedelta
        humanize.precisedelta.return_value = "1 second"
        with patch.object(self, 'visualize_extreme_player_movement.visualize_player_movement') as mock_visualize:
            fu.visualize_extreme_player_movement(self.logs_df, self.df, 'movement_time', self.mask_series)
            # Assert visualize_player_movement called with expected arguments
            mock_visualize.assert_called_once_with(
                self.logs_df,
                (self.logs_df['session_uuid'] == 1) & (self.logs_df['scene_id'] == 2),
                title="Location Map for UUID 1 (2nd Scene) showing trainee with the slowest action to control time (500 milliseconds)"
            )

    def test_descending_sort(self):
        # Mock humanize.precisedelta
        humanize.precisedelta.return_value = "2 seconds"
        with patch.object(self, 'visualize_extreme_player_movement') as mock_visualize:
            fu.visualize_extreme_player_movement(self.logs_df, self.df, 'movement_time', self.mask_series, is_ascending=False)
            # Assert visualize_player_movement called with expected arguments
            mock_visualize.assert_called_once_with(
                self.logs_df,
                (self.logs_df['session_uuid'] == 1) & (self.logs_df['scene_id'] == 1),
                title="Location Map for UUID 1 (1st Scene) showing trainee with the fastest action to control time (2 seconds)"
            )

    def test_humanize_percentage(self):
        # Mock humanize.intword
        humanize.intword.return_value = "200%"
        with patch.object(self, 'visualize_extreme_player_movement') as mock_visualize:
            fu.visualize_extreme_player_movement(self.logs_df, self.df, 'movement_time', self.mask_series, humanize_type='percentage')
            # Assert mock called with expected title
            self.assertIn("200%", mock_visualize.call_args[1]['title'])

    def test_missing_humanize_type(self):
        with self.assertRaises(ValueError):
            fu.visualize_extreme_player_movement(self.logs_df, self.df, 'movement_time', self.mask_series, humanize_type='invalid_type')


class TestShowTimelines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create mock data
        data = {
            "session_uuid": ["session_1", "session_1", "session_2", "session_2"],
            "scene_id": [1, 2, 1, 2],
            "patient_id": ["patient_a", "patient_b", "patient_a", "patient_b"],
            "action_tick": [100000, 200000, 300000, 400000],
            "action_type": ["PATIENT_ENGAGED", "INJURY_TREATED", "PATIENT_ENGAGED", "PULSE_TAKEN"]
        }
        cls.df = pd.DataFrame(data)

    def test_fu.show_timelines_all_params(self):
        # Test with all parameters provided
        random_session_uuid = "test_session_uuid"
        random_scene_index = 1
        color_cycler = lambda: iter(["red", "blue"])
        returned_session_uuid, returned_scene_index = fu.show_timelines(
            self.df, random_session_uuid, random_scene_index, color_cycler, verbose=True
        )
        self.assertEqual(returned_session_uuid, random_session_uuid)
        self.assertEqual(returned_scene_index, random_scene_index)
        # Additional assertions for specific timeline elements or plot properties

    def test_fu.show_timelines_no_random_params(self):
        # Test with no random parameters provided
        returned_session_uuid, returned_scene_index = fu.show_timelines(self.df, verbose=True)
        self.assertIn(returned_session_uuid, self.df.session_uuid.unique())
        self.assertIn(returned_scene_index, self.df.scene_id.unique())
        # Additional assertions for specific timeline elements or plot properties

    def test_fu.show_timelines_no_color_cycler(self):
        # Test with no color cycler provided
        random_session_uuid = "test_session_uuid"
        random_scene_index = 1
        returned_session_uuid, returned_scene_index = fu.show_timelines(
            self.df, random_session_uuid, random_scene_index, verbose=True
        )
        self.assertEqual(returned_session_uuid, random_session_uuid)
        self.assertEqual(returned_scene_index, random_scene_index)
        # Additional assertions for specific timeline elements or plot properties

    def tearDown(self):
        # Clear matplotlib plot (optional)
        plt.clf()


class TestPlotGroupedBoxAndWhiskers(unittest.TestCase):

    @patch('humanize.precisedelta')  # Mocking the humanize function
    def test_plot_grouped_box_and_whiskers(self, mock_precisedelta):
        # Sample data
        data = {
            'x_col': ['A', 'A', 'B', 'B', 'C'],
            'y_col': [1000, 1200, 800, 1100, 900]
        }
        df = pd.DataFrame(data)

        # Mock the function to avoid actual plotting during testing
        with patch.object(plt, 'show'):
            self.plot_grouped_box_and_whiskers(df, 'x_col', 'y_col', 'X-Label', 'Y-Label')

        # Assertions about the function behavior
        # You can add assertions here to check the behavior of the function
        # based on your specific requirements. For example,
        # - Check if the correct data is passed to seaborn.boxplot
        # - Check if the x-axis labels are rotated
        # - Check if the y-axis labels are humanized (if is_y_temporal is True)

        # Example assertion on mock function call
        mock_precisedelta.assert_not_called()  # Assert humanize is not called when is_y_temporal is False

    def test_plot_grouped_box_and_whiskers_with_transform(self):
        # Similar test structure as above, but with transformer_name set
        # You can adapt this test case to test different transformation functions


class TestShowGazeTimeline(unittest.TestCase):

    @patch('fu.show_gaze_timeline.random.choice')
    @patch('fu.show_gaze_timeline.display')
    def test_fu.show_gaze_timeline(self, mock_display, mock_random_choice):
        # Mock data
        logs_df = pd.DataFrame({'session_uuid': ['uuid1', 'uuid1', 'uuid2', 'uuid2'],
                                'scene_id': [1, 1, 2, 2],
                                'action_type': ['PLAYER_GAZE', 'SESSION_END', 'PLAYER_GAZE', 'SESSION_START'],
                                'player_gaze_patient_id': ['patient1', None, 'patient2', None],
                                'action_tick': [1000, 2000, 3000, 4000]})

        # Mock return values
        mock_random_choice.side_effect = ['uuid1', 1]

        # Call the function
        random_session_uuid, random_scene_index = fu.show_gaze_timeline(logs_df=logs_df, verbose=True)

        # Assertions
        self.assertEqual(random_session_uuid, 'uuid1')
        self.assertEqual(random_scene_index, 1)

        # Assert calls to mock functions
        mock_random_choice.assert_called_once_with(logs_df[logs_df['action_type'].isin(['PLAYER_GAZE'])].session_uuid.unique())
        mock_random_choice.assert_called_with(logs_df[(logs_df['session_uuid'] == 'uuid1') & (logs_df['action_type'].isin(['PLAYER_GAZE']))].scene_id.unique())
        mock_display.assert_called_once()

        # Assert plot elements (using mocks)
        fig_mock = MagicMock()
        ax_mock = MagicMock()
        fig_mock.add_subplot.return_value = ax_mock

        with patch('fu.show_gaze_timeline.plt.figure', return_value=fig_mock):
            fu.show_gaze_timeline(logs_df=logs_df, verbose=True)

        ax_mock.hlines.assert_called_once()
        ax_mock.vlines.assert_called_once()
        ax_mock.set_yticks.assert_called_once()
        ax_mock.set_yticklabels.assert_called_once()
        ax_mock.set_title.assert_called_once()
        ax_mock.set_xlabel.assert_called_once()
        for label in ['Elapsed Time since Scene Start']:
            self.assertIn(mock.call.set_label(label), ax_mock.mock_calls)

class TestPlotSequenceBySceneTuple(unittest.TestCase):

    def setUp(self):
        # Initialize any required data or objects before each test method
        pass

    def tearDown(self):
        # Clean up any resources or revert changes made during the test
        pass

    def test_plot_sequence_by_scene_tuple(self):
        # Define test data
        scene_tuple = ("session_uuid_test", "scene_id_test")
        sequence = ["action1", "action2", "action3"]
        logs_df = pd.DataFrame({
            'session_uuid': ["session_uuid_test"]*3,
            'scene_id': ["scene_id_test"]*3,
            'action_tick': [1, 2, 3],
            'voice_command_message': [np.nan]*3,
            'action_type': ["ACTION_TYPE_1", "ACTION_TYPE_2", "ACTION_TYPE_3"]
        })
        summary_statistics_df = pd.DataFrame({
            'session_uuid': ["session_uuid_test"],
            'scene_id': ["scene_id_test"],
            'sequence_entropy': [0.5],
            'sequence_turbulence': [0.3],
            'sequence_complexity': [0.7]
        })

        # Run the function
        fig, ax = fu.plot_sequence_by_scene_tuple(scene_tuple, sequence, logs_df, summary_statistics_df)

        # Asserts
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        # Add more assertions if needed

if __name__ == "__main__":
    unittest.main()