from unittest import mock, TestCase

from force_bdss.api import (
    KPISpecification,
    Workflow,
    DataValue,
    WorkflowEvaluator,
)

from itwm_example.mco.tests.mock_classes import MockOptimizer
from itwm_example.mco.mco_factory import MCOFactory
from itwm_example.mco.space_sampling.space_samplers import (
    UniformSpaceSampler,
    DirichletSpaceSampler,
)


class TestMCO(TestCase):
    def setUp(self):
        self.plugin = {"id": "pid", "name": "Plugin"}
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
        self.evaluator = WorkflowEvaluator(workflow=Workflow())
        self.evaluator.workflow.mco = self.mco_model

    def test_basic_eval(self):
        mock_kpi_return = [DataValue(value=2), DataValue(value=3)]

        with mock.patch(
            "force_bdss.api.Workflow.execute", return_value=mock_kpi_return
        ):
            self.mco.run(self.evaluator)

    def test_internal_weighted_evaluator(self):
        parameters = self.mco_model.parameters

        evaluator = self.mco.optimizer(
            single_point_evaluator=self.evaluator, parameters=parameters
        )
        mock_kpi_return = [DataValue(value=2), DataValue(value=3)]

        with mock.patch(
            "force_bdss.api.Workflow.execute", return_value=mock_kpi_return
        ) as mock_exec:
            evaluator.optimize([0.5, 0.5])
            self.assertEqual(7, mock_exec.call_count)

    def test_scaling_factors(self):
        optimizer = MockOptimizer(self.evaluator, self.parameters)

        scaling_factors = self.mco.get_scaling_factors(optimizer, self.kpis)

        self.assertEqual(scaling_factors, [0.1, 0.1])

    def test_auto_scale(self):

        temp_kpis = [KPISpecification(), KPISpecification(auto_scale=False)]

        optimizer = MockOptimizer(self.evaluator, self.parameters)

        scaling_factors = self.mco.get_scaling_factors(optimizer, temp_kpis)

        self.assertEqual(scaling_factors, [0.1, 1.0])

    def test__space_search_distribution(self):
        for strategy, klass in (
            ("Uniform", UniformSpaceSampler),
            ("Dirichlet", DirichletSpaceSampler),
            ("Uniform", UniformSpaceSampler),
        ):
            self.mco_model.space_search_mode = strategy
            distribution = self.mco._space_search_distribution(self.mco_model)
            self.assertIsInstance(distribution, klass)
            self.assertEqual(len(self.kpis), distribution.dimension)
            self.assertEqual(distribution.resolution, 7)
