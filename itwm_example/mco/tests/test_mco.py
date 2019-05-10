import unittest
from unittest import mock

from itwm_example.mco.mco import (
    get_weight_combinations, MCO, InternalSinglePointEvaluator,
    get_scaling_factors
    )
from force_bdss.tests.probe_classes.mco import ProbeMCOFactory
from force_bdss.core.kpi_specification import KPISpecification
from force_bdss.tests.probe_classes.probe_extension_plugin import (
    ProbeExtensionPlugin
    )


class MockEval():

    def __init__(self, eval, weights, param, **kwargs):
        self.value = 10
        self.weights = weights

    def optimize(self):
        result = [0 if weight != 0 else self.value for weight in self.weights]
        return 0, result


class TestMCO(unittest.TestCase):
    def setUp(self):
        self.plugin = ProbeExtensionPlugin()
        self.factory = ProbeMCOFactory(self.plugin)
        self.mco = MCO(self.factory)
        self.kpis = [KPISpecification(), KPISpecification()]
        self.parameters = [1, 1, 1, 1]

        self.mock_evaluator = mock.Mock(spec=InternalSinglePointEvaluator)

    def test_scaling_factors(self):

        with mock.patch('itwm_example.mco.mco.WeightedEvaluator') as mock_eval:
            mock_eval.side_effect = MockEval
            scaling_factors = get_scaling_factors(self.mock_evaluator,
                                                  self.kpis,
                                                  self.parameters)

            self.assertEqual(scaling_factors, [0.1, 0.1])

    def test_auto_scale(self):

        temp_kpis = [KPISpecification(), KPISpecification(auto_scale=False)]

        with mock.patch('itwm_example.mco.mco.WeightedEvaluator') as mock_eval:
            mock_eval.side_effect = MockEval
            scaling_factors = get_scaling_factors(self.mock_evaluator,
                                                  temp_kpis,
                                                  self.parameters)

            self.assertEqual(scaling_factors, [0.1, 1.])

    def test_get_weight_combinations(self):
        self.assertEqual(list(get_weight_combinations(1, 5)), [[1.0]])
        self.assertEqual(list(get_weight_combinations(2, 5)), [
            [1.0, 0.0],
            [0.75, 0.25],
            [0.50, 0.50],
            [0.25, 0.75],
            [0.0, 1.0],
        ])

        self.assertEqual(list(get_weight_combinations(3, 5)), [
            [1.0, 0.0, 0.0],
            [0.75, 0.25, 0.0],
            [0.75, 0.0, 0.25],
            [0.50, 0.50, 0.0],
            [0.50, 0.25, 0.25],
            [0.50, 0.0, 0.50],
            [0.25, 0.75, 0.0],
            [0.25, 0.50, 0.25],
            [0.25, 0.25, 0.50],
            [0.25, 0.0, 0.75],
            [0.0, 1.0, 0.0],
            [0.0, 0.75, 0.25],
            [0.0, 0.50, 0.50],
            [0.0, 0.25, 0.75],
            [0.0, 0.0, 1.0],
        ])

        self.assertEqual(
            list(get_weight_combinations(3, 9)),
            [[1.0, 0.0, 0.0], [0.875, 0.125, 0.0], [0.875, 0.0, 0.125],
             [0.75, 0.25, 0.0], [0.75, 0.125, 0.125], [0.75, 0.0, 0.25],
             [0.625, 0.375, 0.0], [0.625, 0.25, 0.125], [0.625, 0.125, 0.25],
             [0.625, 0.0, 0.375], [0.5, 0.5, 0.0], [0.5, 0.375, 0.125],
             [0.5, 0.25, 0.25], [0.5, 0.125, 0.375], [0.5, 0.0, 0.5],
             [0.375, 0.625, 0.0], [0.375, 0.5, 0.125], [0.375, 0.375, 0.25],
             [0.375, 0.25, 0.375], [0.375, 0.125, 0.5], [0.375, 0.0, 0.625],
             [0.25, 0.75, 0.0], [0.25, 0.625, 0.125], [0.25, 0.5, 0.25],
             [0.25, 0.375, 0.375], [0.25, 0.25, 0.5], [0.25, 0.125, 0.625],
             [0.25, 0.0, 0.75], [0.125, 0.875, 0.0], [0.125, 0.75, 0.125],
             [0.125, 0.625, 0.25], [0.125, 0.5, 0.375], [0.125, 0.375, 0.5],
             [0.125, 0.25, 0.625], [0.125, 0.125, 0.75], [0.125, 0.0, 0.875],
             [0.0, 1.0, 0.0], [0.0, 0.875, 0.125], [0.0, 0.75, 0.25],
             [0.0, 0.625, 0.375], [0.0, 0.5, 0.5], [0.0, 0.375, 0.625],
             [0.0, 0.25, 0.75], [0.0, 0.125, 0.875], [0.0, 0.0, 1.0]]
        )

        self.assertEqual(list(get_weight_combinations(2, 5, False)), [
            [0.75, 0.25],
            [0.50, 0.50],
            [0.25, 0.75],
        ])

        self.assertEqual(
            list(get_weight_combinations(3, 9, False)),
            [[0.75, 0.125, 0.125],
             [0.625, 0.25, 0.125],
             [0.625, 0.125, 0.25],
             [0.5, 0.375, 0.125],
             [0.5, 0.25, 0.25],
             [0.5, 0.125, 0.375],
             [0.375, 0.5, 0.125],
             [0.375, 0.375, 0.25],
             [0.375, 0.25, 0.375],
             [0.375, 0.125, 0.5],
             [0.25, 0.625, 0.125],
             [0.25, 0.5, 0.25],
             [0.25, 0.375, 0.375],
             [0.25, 0.25, 0.5],
             [0.25, 0.125, 0.625],
             [0.125, 0.75, 0.125],
             [0.125, 0.625, 0.25],
             [0.125, 0.5, 0.375],
             [0.125, 0.375, 0.5],
             [0.125, 0.25, 0.625],
             [0.125, 0.125, 0.75]]
        )
