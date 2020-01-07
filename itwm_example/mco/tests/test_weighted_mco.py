from unittest import TestCase, mock

from traits.testing.unittest_tools import UnittestTools

from force_bdss.api import (
    WorkflowEvaluator,
    KPISpecification,
    Workflow,
    DataValue,
    RangedMCOParameterFactory,
    FixedMCOParameterFactory,
)

from itwm_example.mco.weighted_mco_factory import WeightedMCOFactory
from itwm_example.mco.weighted_mco_model import WeightedMCOModel
from itwm_example.mco.weighted_mco import WeightedMCO
from itwm_example.example_plugin import ExamplePlugin


class TestWeightedMCO(TestCase, UnittestTools):
    def setUp(self):
        self.plugin = ExamplePlugin()
        self.factory = self.plugin.mco_factories[0]
        self.mco = self.factory.create_optimizer()
        self.model = self.factory.create_model()

        self.kpis = [KPISpecification(), KPISpecification()]
        self.parameters = [1, 1, 1, 1]

        self.parameters = [
            RangedMCOParameterFactory(self.factory).create_model(
                {"lower_bound": 0.0, "upper_bound": 1.0}
            )
            for _ in self.parameters
        ]
        self.model.parameters = self.parameters

    def test_mco_factory(self):
        self.assertIsInstance(self.factory, WeightedMCOFactory)
        self.assertEqual("weighted_mco", self.factory.get_identifier())
        self.assertIs(self.factory.model_class, WeightedMCOModel)
        self.assertIs(self.factory.optimizer_class, WeightedMCO)
        self.assertEqual(
            [FixedMCOParameterFactory, RangedMCOParameterFactory],
            self.factory.parameter_factory_classes,
        )
        self.assertEqual(
            "Weighted Multi Criteria optimizer", self.factory.name
        )

    def test_mco_model(self):
        self.assertEqual("SLSQP", self.model.algorithms)
        self.assertEqual(7, self.model.num_points)
        self.assertEqual(True, self.model.verbose_run)
        self.assertEqual("Uniform", self.model.space_search_mode)
        self.assertEqual("Internal", self.model.evaluation_mode)

    def test_simple_run(self):
        mco = self.factory.create_optimizer()
        model = self.factory.create_model()
        model.parameters = self.parameters
        model.kpis = [
            KPISpecification(auto_scale=False),
            KPISpecification(auto_scale=False),
        ]

        evaluator = WorkflowEvaluator(
            workflow=Workflow(), workflow_filepath="whatever"
        )
        evaluator.workflow.mco = model
        kpis = [DataValue(value=1), DataValue(value=2)]
        with self.assertTraitChanges(mco, "event", count=model.num_points - 2):
            with mock.patch(
                "force_bdss.api.Workflow.execute", return_value=kpis
            ) as mock_exec:
                mco.run(evaluator)
                self.assertEqual(49, mock_exec.call_count)
