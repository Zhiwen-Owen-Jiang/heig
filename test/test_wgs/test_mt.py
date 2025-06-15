import unittest
import os, sys
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from heig.wgs.mt import remove_variants_inplace


class Test_remove_variants(unittest.TestCase):
    def test_good_cases(self):
        target_array = np.array(
            [[0, 0, 0, 0],
             [0, 0, 2, 1],
             [0, 0, 0, 1]],
             dtype=np.int8
        )
        to_flip_array = np.array(
            [[1, 2, 2, 2],
             [0, 0, 2, 1],
             [0, 0, 0, 1]],
             dtype=np.int8
        )
        to_flip_matrix = csr_matrix(to_flip_array)
        remove_variants_inplace(to_flip_matrix)
        assert_array_equal(target_array, to_flip_matrix.toarray())

        target_array = np.array(
            [[1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
             dtype=np.int8
        )
        to_flip_array = np.array(
            [[1, 0, 0, 0],
             [2, 2, 0, 1],
             [2, 2, 2, 1]],
             dtype=np.int8
        )
        to_flip_matrix = csr_matrix(to_flip_array)
        remove_variants_inplace(to_flip_matrix)
        assert_array_equal(target_array, to_flip_matrix.toarray())


if __name__ == '__main__':
    to_flip_array = np.array(
            [[1, 2, 2, 2],
             [0, 0, 2, 1],
             [0, 0, 0, 1]],
             dtype=np.int8
        )
    to_flip_matrix = csr_matrix(to_flip_array)
    remove_variants_inplace(to_flip_matrix)
    to_flip_matrix