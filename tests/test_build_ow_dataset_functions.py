
from datetime import timedelta
from unittest.mock import patch
import os
import pandas as pd
import unittest

# Import the class containing the functions
import sys
if (osp.join(os.pardir, 'py') not in sys.path): sys.path.insert(1, osp.join(os.pardir, 'py'))
from FRVRS import fu, nu

class TestBuildOWDatasetFunctions(unittest.TestCase):
    
    def setUp(self):
        self.df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3], 'C': ['a', 'b', 'c']})
        self.df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, np.nan, 3.3], 'C': ['a', 'b', 'c']})
        self.df3 = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'], 'C': ['g', 'h', 'i']})

    # Test scaffolding
    def test_get_numeric_columns():
        
        # Test case 1: Numeric columns with NaN dropping enabled
        assert nu.get_numeric_columns(self.df1) == ['A', 'B']

        # Test case 2: Numeric columns without NaN dropping
        assert nu.get_numeric_columns(self.df2, is_na_dropped=False) == ['A', 'B']

        # Test case 3: No numeric columns
        assert nu.get_numeric_columns(self.df3) == []

    # Test scaffolding
    def test_get_maximum_injury_severity():
        
        # Test case 1: DataFrame with valid values
        data = {'injury_severity': [1, 2, 3, np.nan, 5]}
        df = pd.DataFrame(data)
        assert YourClass.get_maximum_injury_severity(df) == 1
        
        # Test case 2: DataFrame with missing values
        data = {'injury_severity': [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(data)
        assert YourClass.get_maximum_injury_severity(df) is None
        
        # Test case 3: DataFrame with non-numeric values
        data = {'injury_severity': ['high', 'medium', 'low']}
        df = pd.DataFrame(data)
        assert YourClass.get_maximum_injury_severity(df) is None
        
        print("All tests passed successfully!")