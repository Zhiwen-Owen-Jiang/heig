import unittest
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from heig.utils import find_loc


def find_window(positions, start, end):
    """
    Get index of a window [start, end)
    
    """
    start_idx = find_loc(positions, start)
    end_idx = find_loc(positions, end-1)
    if positions[start_idx] < start:
        start_idx += 1
    if end_idx > start_idx + 1:
        return start_idx, end_idx
    else:
        return None, None
    

class Test_find_window(unittest.TestCase):
    def test_case1(self):
        positions = [1,3,4,6,8,9]
        start = 3
        end = 8
        start_idx, end_idx = find_window(positions, start, end)
        self.assertEqual(start_idx, 1)
        self.assertEqual(end_idx, 3)

    def test_case2(self):
        positions = [1,3,4,6,8,9]
        start = 3
        end = 7
        start_idx, end_idx = find_window(positions, start, end)
        self.assertEqual(start_idx, 1)
        self.assertEqual(end_idx, 3)

    def test_case3(self):
        positions = [1,3,4,6,8,9]
        start = 2
        end = 8
        start_idx, end_idx = find_window(positions, start, end)
        self.assertEqual(start_idx, 1)
        self.assertEqual(end_idx, 3)

    def test_case4(self):
        positions = [1,3,4,6,8,9]
        start = 2
        end = 10
        start_idx, end_idx = find_window(positions, start, end)
        self.assertEqual(start_idx, 1)
        self.assertEqual(end_idx, 5)