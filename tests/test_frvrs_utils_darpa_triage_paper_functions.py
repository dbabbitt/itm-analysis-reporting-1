
from datetime import timedelta
from numpy import nan, isnan
from pandas import DataFrame, isna
from unittest.mock import patch
import re
import unittest

# Import the class containing the functions
import sys
if ('../py' not in sys.path): sys.path.insert(1, '../py')
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
        actual_df = fu.get_statistics(self.describable_df, columns_list)
        self.assertTrue(self.expected_df.equals(actual_df))

    def test_get_statistics_subset_columns(self):
        # Test with a subset of columns
        columns_list = ['triage_time']
        actual_df = fu.get_statistics(self.describable_df, columns_list)
        self.assertTrue(self.expected_df.drop(columns=['last_controlled_time']).equals(actual_df))

class TestGetElevensDataFrame(unittest.TestCase):

    def setUp(self):
        # Create sample DataFrames for testing
        data_frames_list = nu.load_data_frames(
            verbose=False, first_responder_master_registry_df='', first_responder_master_registry_file_stats_df='', first_responder_master_registry_scene_stats_df=''
        )
        self.logs_df = data_frames_list['first_responder_master_registry_df']
        self.file_stats_df = data_frames_list['first_responder_master_registry_file_stats_df']
        self.scene_stats_df = data_frames_list['first_responder_master_registry_scene_stats_df']

    def test_all_columns_in_logs_df(self):
        """
        Test case where all needed columns are present in the logs DataFrame.
        """
        needed_columns = ['session_uuid', 'scene_id']
        elevens_df = fu.get_elevens_data_frame(self.logs_df, self.file_stats_df, self.scene_stats_df, needed_columns)
        self.assertEqual(elevens_df.shape[1], self.logs_df.shape[1]+2)  # All rows from logs_df should be present

    def test_missing_columns_from_all_dfs(self):
        """
        Test case where a needed column is missing from all DataFrames.
        """
        needed_columns = ['missing_column']
        with self.assertRaises(KeyError):
            fu.get_elevens_data_frame(self.logs_df, self.file_stats_df, self.scene_stats_df, needed_columns)

    def test_missing_column_from_logs_df_only(self):
        """
        Test case where a needed column is missing only from the logs DataFrame.
        """
        needed_columns = ['scene_type', 'is_scene_aborted', 'is_a_one_triage_file', 'responder_category']
        elevens_df = fu.get_elevens_data_frame(self.logs_df, self.file_stats_df, self.scene_stats_df, needed_columns)
        self.assertEqual(elevens_df.shape[1], self.logs_df.shape[1] + len(needed_columns))  # Merged DataFrames with additional columns

    def test_empty_dataframes(self):
        """
        Test case where all DataFrames are empty.
        """
        empty_df = DataFrame()
        with self.assertRaises(ValueError):
            fu.get_elevens_data_frame(empty_df, empty_df, empty_df)


### Patient Functions ###


class TestGetMaxSalt(unittest.TestCase):
    def setUp(self):
        self.patient_df = DataFrame({
            'action_tick': [497423, 357709, 336847, 305828, 874373],
            'patient_id': ['Gary_3 Root', 'Gary_3 Root', 'Gary_3 Root', 'Gary_3 Root', 'Gary_3 Root'],
            'patient_salt': ['IMMEDIATE', nan, 'IMMEDIATE', nan, 'IMMEDIATE']
        })
    def test_max_salt_with_dataframe(self):
        result = fu.get_max_salt(self.patient_df)
        self.assertEqual(result, 'IMMEDIATE')
    def test_max_salt_with_session_uuid(self):
        result = fu.get_max_salt(self.patient_df, session_uuid='some_uuid')
        self.assertEqual(result, ('Gary_3 Root', 'IMMEDIATE'))

class TestGetLastTag(unittest.TestCase):

    def setUp(self):
        # Set up any common data or configurations needed for the tests
        pass

    def tearDown(self):
        # Clean up any resources created during the tests
        pass

    def test_get_last_tag_with_tags(self):
        # Test when there are tags in the DataFrame
        patient_data = {
            'tag_applied_type': ['tag1', 'tag2', 'tag3'],
            # Add other relevant columns to create a sample DataFrame
        }
        patient_df = DataFrame(patient_data)
        result = fu.get_last_tag(patient_df)
        self.assertEqual(result, 'tag3')

    def test_get_last_tag_with_no_tags(self):
        # Test when there are no tags in the DataFrame
        patient_data = {
            'tag_applied_type': [nan, nan, nan],
            # Add other relevant columns to create a sample DataFrame
        }
        patient_df = DataFrame(patient_data)
        result = fu.get_last_tag(patient_df)
        self.assertTrue(isnan(result))

class TestIsPatientHemorrhagingFunction(unittest.TestCase):
    def test_patient_hemorrhaging_positive(self):
        patient_df = DataFrame({
            'action_tick': [305827, 305827, 305827, 693191, 357708],
            'patient_id': ['Bob_0 Root', 'Bob_0 Root', 'Bob_0 Root', 'Bob_0 Root', 'Bob_0 Root'],
            'injury_required_procedure': ['decompress', nan, 'tourniquet', nan, nan]
        })
        result = fu.get_is_patient_hemorrhaging(patient_df)
        self.assertTrue(result)
    def test_patient_hemorrhaging_negative(self):
        patient_df = DataFrame({
            'action_tick': [649936, 242804, 659793, 648599, 651913],
            'patient_id': ['Gloria_8 Root', 'Gloria_8 Root', 'Gloria_8 Root', 'Gloria_8 Root', 'Gloria_8 Root'],
            'injury_required_procedure': ['gauzePressure', 'gauzePressure', 'gauzePressure', 'gauzePressure', 'gauzePressure']
        })
        result = fu.get_is_patient_hemorrhaging(patient_df)
        self.assertFalse(result)

class TestGetTimeToHemorrhageControl(unittest.TestCase):

    def setUp(self):
        
        # Create mock data for the patient DataFrame
        self.patient_df = DataFrame({
            'action_type': ['INJURY_TREATED', 'INJURY_TREATED', 'INJURY_RECORD'],
            'action_tick': [455202, 448909, 305828],
            'injury_id': ['R Side Puncture', 'R Thigh Laceration', 'R Side Puncture'],
            'injury_record_required_procedure': [nan, nan, 'woundpack'],
            'injury_treated_required_procedure': ['woundpack', 'gauzePressure', nan]
        })
    
    def test_get_time_to_hemorrhage_control_no_scene_start(self):
        
        # Test the function when scene_start is not provided
        result = fu.get_time_to_hemorrhage_control(self.patient_df)
        expected_result = 6_293

        # Add assertions based on the expected result
        self.assertEqual(result, expected_result)

    def test_get_time_to_hemorrhage_control_with_scene_start(self):
        
        # Test the function when scene_start is provided
        scene_start = 300_000
        result = fu.get_time_to_hemorrhage_control(self.patient_df, scene_start=scene_start)
        expected_result = 155_202

        # Add assertions based on the expected result
        self.assertEqual(result, expected_result)


### Scene Functions ###


class TestGetTriageTime(unittest.TestCase):

    def setUp(self):
        # Create a mock DataFrame with scene start and end time columns
        self.scene_df = DataFrame({
            "action_tick": [575896, 693191, 699598]
        })

    def test_get_triage_time(self):
        # Test with first scene
        triage_time = fu.get_triage_time(self.scene_df)
        self.assertEqual(triage_time, 123702)

class TestGetPercentHemorrhageControlled(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.procedures_list = [
            'tourniquet', 'tourniquet', 'tourniquet', 'tourniquet', 'woundpack',
            'tourniquet', 'decompress', 'woundpack', 'tourniquet', 'gauzePressure'
        ]
        self.nons_list = [
            'gauzePressure', 'airway', 'airway', 'gauzePressure', 'gauzePressure',
            'gauzePressure', 'decompress', 'airway', 'decompress', 'decompress'
        ]
        self.all_trues = [True] * len(self.procedures_list)
        self.some_trues = [
            False, False, False, False, False,
            True, False, False, False, False
        ]
        self.all_falses = [False] * len(self.procedures_list)
    def test_all_controlled(self):
        """
        Tests the function with all hemorrhage cases controlled.
        """
        scene_df = DataFrame({
            'action_tick': [
                53712, 111438, 173005, 201999, 235440,
                249305, 296810, 329527, 363492, 442375
            ],
            'patient_id': [
                'Gloria_2 Root', 'Lily_1 Root', 'Lily_1 Root', 'Mike_3 Root', 'Helga_0 Root',
                'Helga_0 Root', 'Mike_6 Root', 'Mike_9 Root', 'Mike_9 Root', 'Mike_4 Root'
            ],
            'injury_id': [
                'L Shin Amputation', 'R Thigh Laceration', 'R Bicep Puncture', 'R Wrist Amputation', 'L Stomach Puncture',
                'L Thigh Puncture', 'R Chest Collapse', 'L Side Puncture', 'L Thigh Puncture', 'R Palm Laceration'
            ],
            'injury_required_procedure': self.procedures_list,
            'injury_record_required_procedure': self.procedures_list,
            'injury_treated_required_procedure': self.procedures_list,
            'injury_treated_injury_treated_with_wrong_treatment': self.all_falses
        })
        percent_controlled = fu.get_percent_hemorrhage_controlled(scene_df)
        self.assertEqual(percent_controlled, 100.0)

    def test_no_hemorrhage_cases(self):
        """
        Tests the function with no hemorrhage cases.
        """
        scene_df = DataFrame({
            'action_tick': [
                53712, 111438, 173005, 201999, 235440,
                249305, 296810, 329527, 363492, 442375
            ],
            'patient_id': [
                'Gloria_2 Root', 'Lily_1 Root', 'Lily_1 Root', 'Mike_3 Root', 'Helga_0 Root',
                'Helga_0 Root', 'Mike_6 Root', 'Mike_9 Root', 'Mike_9 Root', 'Mike_4 Root'
            ],
            'injury_id': [
                'L Shin Amputation', 'R Thigh Laceration', 'R Bicep Puncture', 'R Wrist Amputation', 'L Stomach Puncture',
                'L Thigh Puncture', 'R Chest Collapse', 'L Side Puncture', 'L Thigh Puncture', 'R Palm Laceration'
            ],
            'injury_required_procedure': self.nons_list,
            'injury_record_required_procedure': self.nons_list,
            'injury_treated_required_procedure': self.nons_list,
            'injury_treated_injury_treated_with_wrong_treatment': self.some_trues
        })
        percent_controlled = fu.get_percent_hemorrhage_controlled(scene_df)
        self.assertTrue(isna(percent_controlled))

    def test_wrong_treatment(self):
        """
        Tests the function with a case where a hemorrhage control treatment is marked as wrong.
        """
        scene_df = DataFrame({
            'action_tick': [
                53712, 111438, 173005, 201999, 235440,
                249305, 296810, 329527, 363492, 442375
            ],
            'patient_id': [
                'Gloria_2 Root', 'Lily_1 Root', 'Lily_1 Root', 'Mike_3 Root', 'Helga_0 Root',
                'Helga_0 Root', 'Mike_6 Root', 'Mike_9 Root', 'Mike_9 Root', 'Mike_4 Root'
            ],
            'injury_id': [
                'L Shin Amputation', 'R Thigh Laceration', 'R Bicep Puncture', 'R Wrist Amputation', 'L Stomach Puncture',
                'L Thigh Puncture', 'R Chest Collapse', 'L Side Puncture', 'L Thigh Puncture', 'R Palm Laceration'
            ],
            'injury_required_procedure': self.procedures_list,
            'injury_record_required_procedure': self.procedures_list,
            'injury_treated_required_procedure': self.procedures_list,
            'injury_treated_injury_treated_with_wrong_treatment': self.some_trues
        })
        percent_controlled = fu.get_percent_hemorrhage_controlled(scene_df)
        self.assertEqual(percent_controlled, 87.5)

    def test_duplicate_treatments(self):
        """
        Tests the function with duplicate treatment entries for the same injury.
        """
        scene_df = DataFrame({
            'action_tick': [
                53712, 111438, 173005, 201999, 235440,
                249305, 296810, 329527, 363492, 442375
            ],
            'patient_id': [
                'Gloria_2 Root', 'Lily_1 Root', 'Lily_1 Root', 'Mike_3 Root', 'Helga_0 Root',
                'Helga_0 Root', 'Mike_6 Root', 'Mike_9 Root', 'Mike_9 Root', 'Mike_4 Root'
            ],
            'injury_id': [
                'L Shin Amputation', 'R Thigh Laceration', 'R Bicep Puncture', 'R Wrist Amputation', 'L Stomach Puncture',
                'L Thigh Puncture', 'R Chest Collapse', 'L Side Puncture', 'L Thigh Puncture', 'R Palm Laceration'
            ],
            'injury_required_procedure': self.procedures_list,
            'injury_record_required_procedure': self.procedures_list,
            'injury_treated_required_procedure': self.procedures_list,
            'injury_treated_injury_treated_with_wrong_treatment': self.some_trues
        })
        
        # Create additional row with duplicate treatment
        duplicate_df = scene_df.append(scene_df.iloc[3])
        
        percent_controlled = fu.get_percent_hemorrhage_controlled(duplicate_df)
        self.assertEqual(percent_controlled, 87.5)

class TestGetTimeToLastHemorrhageControlled(unittest.TestCase):

    def setUp(self):
        
        # Create mock data for the scene DataFrame
        self.scene_df = DataFrame({
            'action_tick': [223819, 223819, 223819, 241120, 256019, 256924, 285654, 289814, 317108, 319906, 321245, 367706, 368149, 568501, 571875],
            'patient_id': [
                'Lily_4 Root', 'Lily_2 Root', 'Bob_0 Root', 'Gloria_6 Root', nan, nan, 'Lily_2 Root', nan,
                nan, nan, 'Mike_7 Root', nan, nan, 'Gloria_8 Root', nan
            ],
            'injury_required_procedure': ['woundpack', 'tourniquet', 'tourniquet', nan, nan, nan, 'tourniquet', nan, nan, nan, 'tourniquet', nan, nan, nan, nan],
            'action_type': [
                'INJURY_RECORD', 'INJURY_RECORD', 'INJURY_RECORD', 'S_A_L_T_WAVE_IF_CAN', 'TOOL_HOVER', 'TOOL_HOVER', 'INJURY_TREATED', 'VOICE_CAPTURE',
                'TOOL_HOVER', 'TOOL_HOVER', 'INJURY_TREATED', 'TOOL_HOVER', 'TOOL_HOVER', 'S_A_L_T_WALK_IF_CAN', 'TELEPORT'
            ],
            'injury_id': [
                'L Side Puncture', 'R Shin Amputation', 'L Thigh Laceration', nan, nan, nan, 'R Shin Amputation', nan,
                nan, nan, 'L Thigh Puncture', nan, nan, nan, nan
            ],
            'injury_record_required_procedure': ['woundpack', 'tourniquet', 'tourniquet', nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            'injury_treated_required_procedure': [nan, nan, nan, nan, nan, nan, 'tourniquet', nan, nan, nan, 'tourniquet', nan, nan, nan, nan]
        })

    def test_no_hemorrhage(self):
        # Mock the is_patient_hemorrhaging function to return False for all patients
        self.patch_is_patient_hemorrhaging(return_value=False)

        # Call the function
        last_controlled_time = fu.get_time_to_last_hemorrhage_controlled(self.scene_df)

        # Assert that the returned time is 0
        self.assertEqual(last_controlled_time, 0)

    def test_single_patient_hemorrhage(self):
        # Mock the get_time_to_hemorrhage_control function to return a specific value
        self.patch_get_time_to_hemorrhage_control(return_value=1000)

        # Mock the is_patient_hemorrhaging function to return True for the first patient only
        def get_is_patient_hemorrhaging(patient_df):
            return patient_df["patient_id"].iloc[0] == 'Lily_4 Root'

        self.patch_is_patient_hemorrhaging(side_effect=is_patient_hemorrhaging)

        # Call the function
        last_controlled_time = fu.get_time_to_last_hemorrhage_controlled(self.scene_df)

        # Assert that the returned time is the controlled time from the mocked function
        self.assertEqual(last_controlled_time, 1000)

    def test_multiple_patients_hemorrhage(self):
        # Mock the get_time_to_hemorrhage_control function to return different values for each patient
        def get_time_to_hemorrhage_control(patient_df, scene_start, verbose=False):
            mask_series = ~patient_df.patient_id.isnull()
            patient_id = patient_df[mask_series].patient_id.iloc[0]
            return int(re.sub(r'\D+', '', patient_id))

        self.patch_get_time_to_hemorrhage_control(side_effect=get_time_to_hemorrhage_control)

        # Mock the is_patient_hemorrhaging function to return True for all patients
        self.patch_is_patient_hemorrhaging(return_value=True)

        # Call the function
        last_controlled_time = fu.get_time_to_last_hemorrhage_controlled(self.scene_df, verbose=False)

        # Assert that the returned time is the maximum controlled time from the mocked function
        mask_series = ~self.scene_df.patient_id.isnull()
        controlled_times = [int(re.sub(r'\D+', '', patient_id)) for patient_id in self.scene_df[mask_series].patient_id.unique()]
        self.assertEqual(last_controlled_time, max(controlled_times))

    # Helper methods for patching functions
    def patch_is_patient_hemorrhaging(self, **kwargs):
        patcher = patch.object(fu, "is_patient_hemorrhaging", **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def patch_get_time_to_hemorrhage_control(self, **kwargs):
        patcher = patch.object(fu, "get_time_to_hemorrhage_control", **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

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
            formatted_string = fu.format_timedelta(delta)
            self.assertEqual(formatted_string, expected_string, msg=description)

    def test_format_minutes(self):
        """
        Tests formatting timedelta with minimum unit as minutes.
        """
        for description, delta, _, expected_string in self.test_cases:
            formatted_string = fu.format_timedelta(delta, minimum_unit="minutes")
            self.assertEqual(formatted_string, expected_string, msg=description)

    def test_invalid_minimum_unit(self):
        """
        Tests handling of invalid minimum_unit argument.
        """
        with self.assertRaises(ValueError):
            fu.format_timedelta(timedelta(minutes=1), minimum_unit="invalid_unit")

if __name__ == "__main__":
    unittest.main()