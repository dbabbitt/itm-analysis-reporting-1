
from datetime import timedelta
from unittest.mock import patch
import os
import pandas as pd
import unittest

# Import the class containing the functions
import sys
if (osp.join('..', 'py') not in sys.path): sys.path.insert(1, osp.join('..', 'py'))
from FRVRS import fu, nu

class TestCountSwapsToPerfectOrder(unittest.TestCase):
    
    def setUp(self):
        self.ordered_ints_list = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        self.disordered_ints_list = [1, 3, 2, 2, 3, 3, 3, 2, 3, 1, 2, 2, 3, 3, 3]
    
    def test_equal_lists(self):
        ideal_list = [1, 2, 3, 4, 5]
        compared_list = [1, 2, 3, 4, 5]
        self.assertEqual(nu.count_swaps_to_perfect_order(ideal_list, compared_list), 0)
    
    def test_different_lists(self):
        ideal_list = [1, 2, 3, 4, 5]
        compared_list = [5, 4, 3, 2, 1]
        self.assertEqual(nu.count_swaps_to_perfect_order(ideal_list, compared_list), 2)
    
    def test_lists_with_duplicates(self):
        ideal_list = [1, 2, 3, 4, 5]
        compared_list = [1, 3, 5, 2, 4]
        self.assertEqual(nu.count_swaps_to_perfect_order(ideal_list, compared_list), 3)
    
    def test_different_length_lists(self):
        ideal_list = [1, 2, 3, 4, 5]
        compared_list = [1, 2, 3, 4]
        with self.assertRaises(ValueError): nu.count_swaps_to_perfect_order(ideal_list, compared_list)
    
    def test_ints_lists(self):
        self.assertEqual(nu.count_swaps_to_perfect_order(self.ordered_ints_list, self.disordered_ints_list), 8)