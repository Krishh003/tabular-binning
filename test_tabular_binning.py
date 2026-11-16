import unittest

import numpy as np


class BinningTests(unittest.TestCase):
    def test_quantiles_are_fit_on_training_rows(self):
        from tabular_binning import quantile_bins

        training = np.array([[0.], [1.], [2.], [3.]])
        test = np.array([[-100.], [1.5], [100.]])
        np.testing.assert_array_equal(quantile_bins(training, test, 2).ravel(), [0, 1, 1])
