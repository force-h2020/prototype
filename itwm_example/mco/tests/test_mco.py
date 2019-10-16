from unittest import mock, TestCase

from force_bdss.api import (
    KPISpecification, Workflow, DataValue, WorkflowEvaluator
)

from itwm_example.mco.mco import (
    get_weight_combinations, WeightedEvaluator,
    get_scaling_factors
    )
from itwm_example.mco.mco_factory import MCOFactory


class MockEval():

    def __init__(self, eval, weights, param, **kwargs):
        self.value = 10
        self.weights = weights

    def optimize(self):
        result = [0 if weight != 0 else self.value for weight in self.weights]
        return 0, result


class TestMCO(TestCase):

    def setUp(self):
        self.plugin = {'id': 'pid', 'name': 'Plugin'}
        self.factory = MCOFactory(self.plugin)
        self.mco = self.factory.create_optimizer()
        self.mco_model = self.factory.create_model()

        self.kpis = [KPISpecification(), KPISpecification()]
        self.parameters = [1, 1, 1, 1]

        self.mco_model.kpis = self.kpis
        self.mco_model.parameters = [
            self.factory.parameter_factories[0].create_model()
            for _ in self.parameters
        ]
        self.evaluator = WorkflowEvaluator(
            workflow=Workflow()
        )
        self.evaluator.workflow.mco = self.mco_model

    def test_basic_eval(self):
        mock_kpi_return = [
            DataValue(value=2), DataValue(value=3)
        ]

        with mock.patch('force_bdss.api.Workflow.execute',
                        return_value=mock_kpi_return):
            self.mco.run(self.evaluator)

    def test_internal_weighted_evaluator(self):
        parameters = self.mco_model.parameters

        evaluator = WeightedEvaluator(
            single_point_evaluator=self.evaluator,
            weights=[0.5, 0.5],
            parameters=parameters
        )
        mock_kpi_return = [
            DataValue(value=2), DataValue(value=3)
        ]

        with mock.patch('force_bdss.api.Workflow.execute',
                        return_value=mock_kpi_return) as mock_exec:
            evaluator.optimize()
            self.assertEqual(7, mock_exec.call_count)

    def test_scaling_factors(self):

        with mock.patch('itwm_example.mco.mco.WeightedEvaluator') as mock_eval:
            mock_eval.side_effect = MockEval
            scaling_factors = get_scaling_factors(self.evaluator,
                                                  self.kpis,
                                                  self.parameters)

            self.assertEqual(scaling_factors, [0.1, 0.1])

    def test_auto_scale(self):

        temp_kpis = [KPISpecification(), KPISpecification(auto_scale=False)]

        with mock.patch('itwm_example.mco.mco.WeightedEvaluator') as mock_eval:
            mock_eval.side_effect = MockEval
            scaling_factors = get_scaling_factors(self.evaluator,
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

        self.assertEqual(list(get_weight_combinations(2, 5, zero_values=False)), [
            [0.75, 0.25],
            [0.50, 0.50],
            [0.25, 0.75],
        ])

        self.assertEqual(
            list(get_weight_combinations(3, 9, zero_values=False)),
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
