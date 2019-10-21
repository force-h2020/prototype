from unittest import TestCase
import numpy as np

from itwm_example.mco.scaling_tools.kpi_scaling import sen_scaling_method


class MockOptimizer:
    def __init__(self, dimension):
        self.dimension = dimension
        self.margins = np.array([10.0 + i for i in range(self.dimension)])
        self.min_values = np.array([i for i in range(self.dimension)])

        self.scaling_values = np.reciprocal(self.margins)

    def optimize(self, weights):
        return None, self.min_values + weights * self.margins


class TestSenScaling(TestCase):
    def setUp(self):
        self.optimizer = MockOptimizer(5)
        self.scaling_values = self.optimizer.scaling_values.tolist()

    def test_sen_scaling(self):
        scaling = sen_scaling_method(
            self.optimizer.dimension, self.optimizer.optimize
        )
        self.assertListEqual(scaling.tolist(), self.scaling_values)
