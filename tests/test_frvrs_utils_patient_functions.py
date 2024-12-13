
from numpy import nan
from pandas import DataFrame
import numpy as np
import pandas as pd
import unittest

# Import the class containing the functions
import sys
if (osp.join(os.pardir, 'py') not in sys.path): sys.path.insert(1, osp.join(os.pardir, 'py'))
from FRVRS import fu, nu


### Patient Functions ###


class TestBleedingToolApplication(unittest.TestCase):
    
    def setUp(self):
        # Create a sample DataFrame for testing
        self.sample_data = {
            'tool_applied_sender': ['AppliedTourniquet', 'SomeOtherAction', 'AppliedPackingGauze'],
            # Add other relevant columns as needed
        }
        self.patient_df = pd.DataFrame(self.sample_data)
    
    def test_correct_tool_applied(self):
        # Test if the function correctly identifies the correct tool applied
        result = fu.get_is_correct_bleeding_tool_applied(self.patient_df)
        self.assertTrue(result, "Correct tool should be identified as applied")
    
    def test_incorrect_tool_applied(self):
        # Test if the function correctly identifies when the incorrect tool is applied
        # Modify the sample_data DataFrame to include an incorrect tool application
        self.sample_data['tool_applied_sender'] = ['SomeOtherAction', 'AnotherIncorrectAction']
        self.patient_df = pd.DataFrame(self.sample_data)
        
        result = fu.get_is_correct_bleeding_tool_applied(self.patient_df)
        self.assertFalse(result, "Incorrect tool should not be identified as applied")

class TestIsPatientDead(unittest.TestCase):

    def test_patient_dead(self):
        # Test case where patient is considered dead
        data = {'patient_record_salt': ['DEAD', np.nan],
                'patient_engaged_salt': [np.nan, np.nan]}
        df = pd.DataFrame(data)
        result = fu.get_is_patient_dead(df)
        self.assertTrue(result)

    def test_patient_alive(self):
        # Test case where patient is considered alive
        data = {'patient_record_salt': ['ALIVE', np.nan],
                'patient_engaged_salt': ['ALIVE', np.nan]}
        df = pd.DataFrame(data)
        result = fu.get_is_patient_dead(df)
        self.assertFalse(result)

    def test_unknown_status(self):
        # Test case where patient status is unknown (both columns are empty)
        data = {'patient_record_salt': [np.nan, np.nan],
                'patient_engaged_salt': [np.nan, np.nan]}
        df = pd.DataFrame(data)
        result = fu.get_is_patient_dead(df)
        self.assertTrue(np.isnan(result))

class TestIsPatientStill(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'patient_record_sort': ['still', None, 'not_still', None],
            'patient_engaged_sort': [None, 'still', 'not_still', None]
        }
        self.patient_df = pd.DataFrame(data)

    def test_is_patient_still_true(self):
        result = fu.get_is_patient_still(self.patient_df)
        self.assertTrue(result)

    def test_is_patient_still_false(self):
        # Modify the DataFrame to test for a case where the patient is not still
        self.patient_df['patient_record_sort'] = ['not_still', None, 'not_still', None]
        result = fu.get_is_patient_still(self.patient_df)
        self.assertFalse(result)

    def test_is_patient_still_unknown(self):
        # Modify the DataFrame to test for a case where the result is unknown
        self.patient_df['patient_record_sort'] = [None, None, None, None]
        self.patient_df['patient_engaged_sort'] = [None, None, None, None]
        result = fu.get_is_patient_still(self.patient_df)
        self.assertTrue(np.isnan(result))

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
        patient_df = pd.DataFrame(patient_data)
        result = fu.get_last_tag(patient_df)
        self.assertEqual(result, 'tag3')

    def test_get_last_tag_with_no_tags(self):
        # Test when there are no tags in the DataFrame
        patient_data = {
            'tag_applied_type': [np.nan, np.nan, np.nan],
            # Add other relevant columns to create a sample DataFrame
        }
        patient_df = pd.DataFrame(patient_data)
        result = fu.get_last_tag(patient_df)
        self.assertTrue(np.isnan(result))

class TestGetPatientLocation(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {'location_id': ['(1, 2)', '(3, 4)', '(5, 6)'],
                'action_tick': [100, 200, 300]}
        self.patient_df = pd.DataFrame(data)

    def test_get_location_of_patient(self):
        # Test the function with known input and expected output
        result = fu.get_location_of_patient(self.patient_df, 250)
        self.assertEqual(result, (3, 4))

    def test_get_location_of_patient_invalid_input(self):
        # Test the function with invalid input
        with self.assertRaises(TypeError):
            fu.get_location_of_patient(self.patient_df, 'invalid_input')

    def test_null_location_ids(self):
        # Test the function with null location_id input and expected output
        data = {'location_id': [nan, nan, nan],
                'action_tick': [100, 200, 300]}
        patient_df = pd.DataFrame(data)
        result = fu.get_location_of_patient(patient_df, 250)
        self.assertEqual(result, (0.0, 0.0, 0.0))

class TestIsTagCorrect(unittest.TestCase):

    def setUp(self):
        # Initialize objects needed for testing
        self.your_object = fu()
        self.sample_data = {
            'tag_applied_type': ['A', 'B', 'C'],
            'patient_record_salt': [1, 2, 3]
        }
        self.sample_df = pd.DataFrame(self.sample_data)

    def test_is_tag_correct_with_valid_data(self):
        # Test when both 'tag_applied_type' and 'patient_record_salt' have non-null values
        result = self.your_object.is_tag_correct(self.sample_df)
        self.assertIsInstance(result, bool)

    def test_is_tag_correct_with_insufficient_data(self):
        # Test when either 'tag_applied_type' or 'patient_record_salt' has null values
        insufficient_data_df = pd.DataFrame({'tag_applied_type': [np.nan], 'patient_record_salt': [1]})
        result = self.your_object.is_tag_correct(insufficient_data_df)
        self.assertTrue(np.isnan(result))

    # You can add more test cases for edge cases, exceptions, and different scenarios

class TestPatientInjuryDetection(unittest.TestCase):
    
    def setUp(self):
        # Set up any necessary data or objects for your tests
        pass

    def tearDown(self):
        # Clean up after your tests, if necessary
        pass

    def test_is_patient_severely_hemorrhaging_positive(self):
        # Test the function with a case where the patient has severe injuries
        # You may need to customize the test data based on your specific use case
        patient_df = pd.DataFrame({
            'injury_id': [1, 1, 2, 2],
            'severity': ['Severe', 'Moderate', 'Mild', 'Severe']
        })
        result = fu.get_is_patient_severely_injured(patient_df)
        self.assertTrue(result)

    def test_is_patient_severely_hemorrhaging_negative(self):
        # Test the function with a case where the patient has no severe injuries
        # You may need to customize the test data based on your specific use case
        patient_df = pd.DataFrame({
            'injury_id': [1, 1, 2, 2],
            'severity': ['Mild', 'Moderate', 'Mild', 'Moderate']
        })
        result = fu.get_is_patient_severely_injured(patient_df)
        self.assertFalse(result)

    def test_is_patient_severely_hemorrhaging_empty_dataframe(self):
        # Test the function with an empty DataFrame
        patient_df = pd.DataFrame(columns=['injury_id', 'severity'])
        result = fu.get_is_patient_severely_injured(patient_df)
        self.assertFalse(result)

class TestGetFirstPatientInteraction(unittest.TestCase):
    
    def setUp(self):
        # You may initialize any test-specific data or configurations here
        pass
    
    def tearDown(self):
        # Clean up any resources or data created during the test
        pass
    
    def test_get_first_patient_interaction_with_data(self):
        # Test the function with a DataFrame containing responder negotiations
        data = {
            'action_type': ['negotiation', 'other_action', 'negotiation'],
            'action_tick': [10, 15, 20]
        }
        df = pd.DataFrame(data)
        result = fu.get_first_patient_interaction(df)
        self.assertEqual(result, 10)  # Update the expected result based on your test case
    
    def test_get_first_patient_interaction_without_data(self):
        # Test the function with a DataFrame without responder negotiations
        data = {
            'action_type': ['other_action', 'another_action'],
            'action_tick': [5, 8]
        }
        df = pd.DataFrame(data)
        result = fu.get_first_patient_interaction(df)
        self.assertIsNone(result)  # Update the expected result based on your test case
    
    # Add more test cases as needed

class TestGetLastPatientInteraction(unittest.TestCase):

    def setUp(self):
        # Sample data
        self.data = {
            "action_type": ["LOGIN", "VOICE_COMMAND", "CHAT", "VOICE_COMMAND"],
            "action_tick": [10, 20, 30, 40],
            "voice_command_message": ["Hello", "Goodbye", None, "Help"]
        }
        self.patient_df = pd.DataFrame(self.data)
        self.action_types_list = ["CHAT", "LOGIN"]
        self.command_messages_list = ["Help"]

    def test_no_interaction(self):
        """Test case for no patient interaction"""
        self.patient_df.loc[1, "action_type"] = "PAYMENT"  # Change one interaction type
        result = get_last_patient_interaction(self, self.patient_df)
        self.assertIsNone(result)

    def test_single_interaction(self):
        """Test case for single patient interaction"""
        result = get_last_patient_interaction(self, self.patient_df)
        self.assertEqual(result, 40)  # Last action tick

    def test_multiple_interactions(self):
        """Test case for multiple patient interactions"""
        self.patient_df.loc[3, "action_type"] = "CHAT"  # Add another interaction
        result = get_last_patient_interaction(self, self.patient_df)
        self.assertEqual(result, 40)  # Still last action tick

    def test_voice_command_filter(self):
        """Test case for filtering voice commands"""
        result = get_last_patient_interaction(self, self.patient_df)
        self.assertEqual(result, 40)  # "Help" command should be included


class Testfu(unittest.TestCase):

    def setUp(self):
        # Create sample data with varying gaze behavior
        self.data = {
            "action_type": ["PLAYER_MOVE", "PLAYER_GAZE", "PLAYER_MOVE", "OTHER_ACTION"]
        }
        self.df = pd.DataFrame(self.data)

    def test_gazed_at_patient(self):
        """Tests if the function correctly identifies gazing at the patient."""
        result = fu.get_is_patient_gazed_at(self.df)
        self.assertTrue(result)

    def test_not_gazed_at_patient(self):
        """Tests if the function correctly identifies no gazing."""
        # Modify data to exclude gazing action
        self.df.loc[1, "action_type"] = "PLAYER_MOVE"
        result = fu.get_is_patient_gazed_at(self.df)
        self.assertFalse(result)


class TestWanderings(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'action_tick': [1, 2, 3, 4],
            'patient_demoted_position': [(1, 2, 3), None, (4, 5, 6), None],
            'patient_engaged_position': [None, (7, 8, 9), None, (10, 11, 12)],
            'patient_record_position': [None, None, (13, 14, 15), None],
            'player_gaze_location': [(16, 17, 18), None, None, (19, 20, 21)]
        }
        self.patient_df = pd.DataFrame(data)

    def test_get_wanderings_empty_df(self):
        # Test with an empty DataFrame
        empty_df = pd.DataFrame(columns=self.patient_df.columns)
        x_dim, z_dim = self.patient_df.get_wanderings(empty_df)
        self.assertEqual(x_dim, [])
        self.assertEqual(z_dim, [])

    def test_get_wanderings_single_location(self):
        # Test with a single valid location
        x_dim, z_dim = self.patient_df.get_wanderings(self.patient_df)
        self.assertEqual(x_dim, [1, 7, 13, 16])
        self.assertEqual(z_dim, [3, 9, 15, 18])

    def test_get_wanderings_multiple_locations(self):
        # Test with multiple valid locations
        data = {
            'action_tick': [1, 2, 3],
            'patient_demoted_position': [(1, 2, 3), None, (10, 11, 12)],
            'patient_engaged_position': [None, (4, 5, 6), None],
            'patient_record_position': [None, None, (13, 14, 15)],
            'player_gaze_location': [(7, 8, 9), None, None]
        }
        df = pd.DataFrame(data)
        x_dim, z_dim = df.get_wanderings(df)
        self.assertEqual(x_dim, [1, 4, 10, 7])
        self.assertEqual(z_dim, [3, 6, 12, 9])


class TestGetWrappedLabel(unittest.TestCase):

    def setUp(self):
        self.patient_data = {
            'patient_id': ['123 Root', '456', '789 Special'],
            'patient_demoted_sort': [None, 'High', 'Low'],
            'patient_engaged_sort': ['Low', None, None],
            'patient_record_sort': [None, None, 'Oldest']
        }
        self.df = pd.DataFrame(self.patient_data)

    def test_single_sort_column(self):
        label = get_wrapped_label(self.df)
        self.assertEqual(label, "(High)\n123")

    def test_multiple_sort_columns(self):
        label = get_wrapped_label(self.df.iloc[2])
        self.assertEqual(label, "(Oldest)\n789 Special")

    def test_no_sort_info(self):
        df = self.df.iloc[1]
        df['patient_demoted_sort'] = pd.NA
        df['patient_engaged_sort'] = pd.NA
        df['patient_record_sort'] = pd.NA
        label = get_wrapped_label(df)
        self.assertEqual(label, "no SORT info\n456")

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

    # Add more test cases as needed

class TestPatientEngagementCount(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {'action_type': ['PATIENT_ENGAGED', 'OTHER_ACTION', 'PATIENT_ENGAGED', 'PATIENT_ENGAGED']}
        self.patient_df = pd.DataFrame(data)

    def test_get_patient_engagement_count(self):
        result = fu.get_patient_engagement_count(self.patient_df)
        self.assertEqual(result, 3)  # Adjust the expected count based on your sample data

class TestGetMaximumInjurySeverity(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.data = {
            "patient_id": [1, 2, 3],
            "injury_severity": ["High", None, "Low"]
        }
        self.patient_df = pd.DataFrame(self.data)

    def test_empty_dataframe(self):
        # Test with an empty dataframe
        empty_df = pd.DataFrame(columns=["patient_id", "injury_severity"])
        
        # Assert that it returns None
        self.assertIsNone(fu.get_maximum_injury_severity(empty_df))

    def test_single_value(self):
        # Test with a dataframe containing only one non-null value
        single_value_df = self.patient_df[["patient_id", "injury_severity"]].iloc[1]  # Select second row
        
        # Assert that it returns the single value
        self.assertEqual(fu.get_maximum_injury_severity(single_value_df), "Low")

    def test_multiple_values(self):
        # Test with the original dataframe containing multiple values
        
        # Assert that it returns the minimum non-null value (High)
        self.assertEqual(fu.get_maximum_injury_severity(self.patient_df), "High")

class TestIsLifeThreatened(unittest.TestCase):

    def setUp(self):
        # Create mock data for testing
        self.patient_df_high_severity_hemorrhaging = pd.DataFrame({
            # ... Add columns and data representing high severity and hemorrhaging
        })
        self.patient_df_high_severity_no_hemorrhaging = pd.DataFrame({
            # ... Add columns and data representing high severity and no hemorrhaging
        })
        self.patient_df_low_severity_hemorrhaging = pd.DataFrame({
            # ... Add columns and data representing low severity and hemorrhaging
        })
        self.patient_df_low_severity_no_hemorrhaging = pd.DataFrame({
            # ... Add columns and data representing low severity and no hemorrhaging
        })

    def test_life_threatened_high_severity_hemorrhaging(self):
        # Call the function with data representing high severity and hemorrhaging
        result = fu.get_is_life_threatened(self.patient_df_high_severity_hemorrhaging)
        self.assertTrue(result)

    def test_life_threatened_high_severity_no_hemorrhaging(self):
        # Call the function with data representing high severity and no hemorrhaging
        result = fu.get_is_life_threatened(self.patient_df_high_severity_no_hemorrhaging)
        self.assertFalse(result)

    def test_life_threatened_low_severity_hemorrhaging(self):
        # Call the function with data representing low severity and hemorrhaging
        result = fu.get_is_life_threatened(self.patient_df_low_severity_hemorrhaging)
        self.assertFalse(result)

    def test_life_threatened_low_severity_no_hemorrhaging(self):
        # Call the function with data representing low severity and no hemorrhaging
        result = fu.get_is_life_threatened(self.patient_df_low_severity_no_hemorrhaging)
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()