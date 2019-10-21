from unittest import TestCase

from itwm_example.mco.scaling_tools.kpi_scaling import sen_scaling_method
from itwm_example.mco.optimizers.optimizers import MockOptimizer


class TestSenScaling(TestCase):
    def setUp(self):
        self.optimizer = MockOptimizer(None, [0] * 5, None)
        self.scaling_values = self.optimizer.scaling_values.tolist()

    def test_sen_scaling(self):
        scaling = sen_scaling_method(
            self.optimizer.dimension, self.optimizer.weighted_optimize
        )
        self.assertListEqual(scaling.tolist(), self.scaling_values)
