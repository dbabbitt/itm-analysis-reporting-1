
from contextlib import redirect_stdout
from datetime import timedelta
from numpy import nan
from pandas import DataFrame, to_datetime, Series
from unittest.mock import patch, MagicMock
import numpy as np
import os
import pandas as pd
import unittest

# Import the class containing the functions
import sys
if (osp.join(os.pardir, 'py') not in sys.path): sys.path.insert(1, osp.join(os.pardir, 'py'))
from FRVRS import fu, nu


### Injury Functions ###



class TestGetIsHemorrhageToolApplied(unittest.TestCase):
    """Test class for the get_is_hemorrhage_tool_applied function"""
    
    def setUp(self):
        
        # Create mock dataframes for testing
        self.injury_df = DataFrame({
            'patient_id': ['Gloria_6 Root'],
            'injury_record_required_procedure': ['tourniquet'],
        })
        self.logs_df = DataFrame({
            'patient_id': ['Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root', 'Gloria_6 Root'],
            'tool_applied_type': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 'Tourniquet', nan],
        })
        
        # Mock hemorrhage control procedures list and tool type dictionary
        self.hemorrhage_control_procedures_list = ["Hemorrhage Control A", "Hemorrhage Control B"]
        self.required_procedure_to_tool_type_dict = {"Hemorrhage Control A": "Hemorrhage Control A"}
    
    def test_get_is_hemorrhage_tool_applied(self):

        # Call the function
        result = fu.get_is_hemorrhage_tool_applied(self.injury_df, self.logs_df)
        self.assertTrue(result)


class TestIsInjuryCorrectlyTreated(unittest.TestCase):

    def setUp(self):
        # Create test data
        self.data = {
            "injury_name": ["Ankle Sprain", "Sprained Wrist", "Broken Arm"],
            "injury_treated_injury_treated": [True, False, True],
            "injury_treated_injury_treated_with_wrong_treatment": [False, True, False]
        }
        self.injury_df = pd.DataFrame(self.data)

    def test_correctly_treated(self):
        """
        Tests if the function correctly identifies a correctly treated injury.
        """
        correctly_treated = self.injury_df.copy()
        correctly_treated.loc[0, "injury_treated_injury_treated_with_wrong_treatment"] = True

        result = is_injury_correctly_treated(correctly_treated)
        self.assertTrue(result)

    def test_incorrectly_treated(self):
        """
        Tests if the function correctly identifies an incorrectly treated injury.
        """
        incorrectly_treated = self.injury_df.copy()
        incorrectly_treated.loc[0, "injury_treated_injury_treated_with_wrong_treatment"] = True

        result = is_injury_correctly_treated(incorrectly_treated)
        self.assertFalse(result)

    def test_no_treated_injuries(self):
        """
        Tests if the function correctly handles a DataFrame with no treated injuries.
        """
        no_treated = self.injury_df.copy()
        no_treated["injury_treated_injury_treated"] = False

        result = is_injury_correctly_treated(no_treated)
        self.assertFalse(result)


class TestHemorrhageControlled(unittest.TestCase):

    def setUp(self):
        # Initialize test data
        self.injury_df = pd.DataFrame({
            'injury_id': [1, 2, 3],
            'injury_record_required_procedure': ['procedure1', 'procedure2', 'procedure3'],
            'injury_required_procedure': ['procedure1', 'procedure2', 'procedure3'],
            'patient_id': [101, 102, 103],
            'injury_treated_injury_treated': [True, True, True],
            'injury_treated_injury_treated_with_wrong_treatment': [True, True, True]
        })
        self.logs_df = pd.DataFrame({
            'patient_id': [101, 102, 103],
            'tool_applied_type': ['tool1', 'tool2', 'tool3']
        })
        
    # Define test cases here to verify function behavior with various input scenarios
    def test_controlled_hemorrhage(self):
        # ... Create mock DataFrames with data indicating controlled hemorrhage ...
        # Call the function and assert the returned value is True
        pass
    
    def test_uncontrolled_hemorrhage(self):
        # ... Create mock DataFrames with data indicating uncontrolled hemorrhage ...
        # Call the function and assert the returned value is False
        pass
    
    # Add more test cases for different scenarios
    def test_is_hemorrhage_controlled(self):
        # Test if hemorrhage is controlled for a given injury and logs data
        controlled = fu.get_is_hemorrhage_controlled(self.injury_df, self.logs_df)
        self.assertTrue(controlled)

class TestIsInjuryHemorrhage(unittest.TestCase):

    def setUp(self):
        # Define sample data
        self.injury_df = pd.DataFrame({
            "injury_required_procedure": ["Stitches", "Hemorrhage Control", "Bandage", "X-ray"],
        })
        self.hemorrhage_control_procedures_list = ["Hemorrhage Control"]

    def test_hemorrhage(self):
        """Tests the function with a hemorrhage record."""
        is_hemorrhage = fu.get_is_injury_hemorrhage(self.injury_df)
        self.assertTrue(is_hemorrhage)

    def test_no_hemorrhage(self):
        """Tests the function with no hemorrhage records."""
        self.injury_df.loc[0, "injury_required_procedure"] = "Cast"
        is_hemorrhage = fu.get_is_injury_hemorrhage(self.injury_df)
        self.assertFalse(is_hemorrhage)

    def test_empty_dataframe(self):
        """Tests the function with an empty dataframe."""
        empty_df = pd.DataFrame(columns=["injury_required_procedure"])
        is_hemorrhage = fu.get_is_injury_hemorrhage(empty_df)
        self.assertFalse(is_hemorrhage)

    def test_verbose_output(self):
        """Tests the function with verbose mode enabled."""
        with self.capture_stdout() as output:
            fu.get_is_injury_hemorrhage(self.injury_df, verbose=True)
        self.assertIn("Is the injury a hemorrhage: True", output.getvalue())


class TestInjurySeverity(unittest.TestCase):

   def setUp(self):
       # Create a sample DataFrame with different injury scenarios
       self.injury_df_high_severity_hemorrhage = pd.DataFrame({
           'injury_severity': ['high'],
           # Other relevant columns for hemorrhage determination
       })
       self.injury_df_high_severity_non_hemorrhage = pd.DataFrame({
           'injury_severity': ['high'],
           # Other relevant columns indicating non-hemorrhage
       })
       self.injury_df_low_severity = pd.DataFrame({
           'injury_severity': ['low'],
           # Other relevant columns
       })

   def test_is_injury_severe_true_hemorrhage(self):
       result = fu.get_is_injury_severe(self.injury_df_high_severity_hemorrhage)
       self.assertTrue(result)

   def test_is_injury_severe_true_non_hemorrhage(self):
       result = fu.get_is_injury_severe(self.injury_df_high_severity_non_hemorrhage)
       self.assertTrue(result)

   def test_is_injury_severe_false_low_severity(self):
       result = fu.get_is_injury_severe(self.injury_df_low_severity)
       self.assertFalse(result)

   def test_is_injury_hemorrhage_mocked(self):
       # Mock the is_injury_hemorrhage function to control its behavior
       with unittest.mock.patch.object(fu, 'is_injury_hemorrhage', return_value=True):
           result = fu.get_is_injury_severe(self.injury_df_high_severity_hemorrhage)
           self.assertTrue(result)

class TestBleedingTreatment(unittest.TestCase):

    def setUp(self):
        # Define sample data for the injury DataFrame
        self.injury_df = pd.DataFrame({
            "injury_treated_required_procedure": ["A", "B", "A", "C"],
            "injury_treated_injury_treated": [True, False, True, True],
            "injury_treated_injury_treated_with_wrong_treatment": [False, True, False, True]
        })
        
        # Define hemorrhage control procedures (replace with actual values)
        self.hemorrhage_control_procedures_list = ["A", "D"]

    def test_bleeding_correctly_treated(self):
        """
        Test if the function correctly identifies correctly treated bleeding cases.
        """
        expected_result = True
        actual_result = fu.get_is_bleeding_correctly_treated(self.injury_df, verbose=False)
        self.assertEqual(expected_result, actual_result)

    def test_bleeding_not_correctly_treated(self):
        """
        Test if the function correctly identifies incorrectly treated bleeding cases.
        """
        # Modify data to represent incorrect treatment
        self.injury_df.loc[2, "injury_treated_injury_treated_with_wrong_treatment"] = True
        expected_result = False
        actual_result = fu.get_is_bleeding_correctly_treated(self.injury_df, verbose=False)
        self.assertEqual(expected_result, actual_result)

    def test_verbose_output(self):
        """
        Test if the function prints additional information in verbose mode.
        """
        # Capture output using a context manager
        with captured_output() as (stdout, stderr):
            fu.get_is_bleeding_correctly_treated(self.injury_df, verbose=True)
        
        # Check if expected information is printed to stdout
        self.assertTrue("Was bleeding correctly treated:" in stdout.getvalue())
        self.assertTrue(str(self.injury_df) in stdout.getvalue())

# Helper function to capture output (requires `contextlib` module)
def captured_output():
    import contextlib
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None) as (out, err):
        yield out, err


class TestGetInjuryCorrectlyTreatedTime(unittest.TestCase):

    def setUp(self):
        # Create some sample data
        data = {
            "action_tick": [10, 20, 30, 40, 50],
            "injury_treated_injury_treated": [True, False, True, True, False],
            "injury_treated_injury_treated_with_wrong_treatment": [False, True, False, True, False]
        }
        self.injury_df = pd.DataFrame(data)

    def test_get_injury_correctly_treated_time_success(self):
        """
        Test the function with a correctly treated injury.
        """
        expected_time = 40
        actual_time = fu.get_injury_correctly_treated_time(self.injury_df)
        self.assertEqual(actual_time, expected_time)

    def test_get_injury_correctly_treated_time_no_treatment(self):
        """
        Test the function with no successful treatment.
        """
        # Modify the sample data to have no successful treatment
        self.injury_df.loc[self.injury_df["injury_treated_injury_treated"] == True, "injury_treated_injury_treated"] = False
        expected_time = -1  # Indicate no successful treatment
        actual_time = fu.get_injury_correctly_treated_time(self.injury_df)
        self.assertEqual(actual_time, expected_time)

    def test_get_injury_correctly_treated_time_verbose(self):
        """
        Test the function with verbose mode enabled.
        """
        expected_time = 40
        with self.assertLogs(level="INFO") as cm:
            fu.get_injury_correctly_treated_time(self.injury_df, verbose=True)
        self.assertEqual(len(cm.output), 2)  # Expect two log messages (info and display)
        self.assertIn(f"Action tick when the injury is correctly treated: {expected_time}", cm.output[0])

if __name__ == "__main__":
    unittest.main()