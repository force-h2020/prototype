#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from unittest import TestCase, mock

from traits.testing.unittest_tools import UnittestTools

from force_bdss.api import (
    KPISpecification,
    FixedMCOParameterFactory,
    WeightedMCOStartEvent,
    WeightedMCOProgressEvent,
    Workflow,
    DataValue,
)

from itwm_example.mco.weighted_mco_factory import WeightedMCOFactory
from itwm_example.mco.weighted_mco_model import WeightedMCOModel
from itwm_example.mco.weighted_mco import WeightedMCO
from itwm_example.itwm_example_plugin import ITWMExamplePlugin
from itwm_example.mco.parameters import (
    ITWMRangedMCOParameterFactory,
    ITWMRangedMCOParameter,
)


class TestWeightedMCO(TestCase, UnittestTools):
    def setUp(self):
        self.plugin = ITWMExamplePlugin()
        self.factory = self.plugin.mco_factories[0]
        self.mco = self.factory.create_optimizer()
        self.model = self.factory.create_model()

        self.kpis = [KPISpecification(), KPISpecification()]
        self.parameters = [1, 1, 1, 1]
        self.itwm_parameters_factory = ITWMRangedMCOParameterFactory(
            self.factory
        )

        self.parameters = [
            self.itwm_parameters_factory.create_model(
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
            [FixedMCOParameterFactory, ITWMRangedMCOParameterFactory],
            self.factory.parameter_factory_classes,
        )
        self.assertEqual(
            "Weighted Multi Criteria optimizer", self.factory.name
        )

    def test_mco_parameters(self):
        self.assertIs(
            self.factory.parameter_factory_classes[1],
            ITWMRangedMCOParameterFactory,
        )
        self.assertIn(
            " Initial value is assigned by `Parameter.initial_value`.",
            self.itwm_parameters_factory.description,
        )
        self.assertIs(
            self.itwm_parameters_factory.get_model_class(),
            ITWMRangedMCOParameter,
        )

        for parameter in self.parameters:
            self.assertIsInstance(parameter, ITWMRangedMCOParameter)
            self.assertEqual(0.0, parameter.initial_value)

    def test_mco_model(self):
        self.assertEqual("SLSQP", self.model.algorithms)
        self.assertEqual(7, self.model.num_points)
        self.assertEqual(True, self.model.verbose_run)
        self.assertEqual("Uniform", self.model.space_search_mode)
        self.assertEqual("Internal", self.model.evaluation_mode)
        self.assertIsInstance(
            self.model._start_event_type(), WeightedMCOStartEvent)
        self.assertIsInstance(
            self.model._progress_event_type(), WeightedMCOProgressEvent)

    def test_simple_run(self):
        mco = self.factory.create_optimizer()
        model = self.factory.create_model()
        model.parameters = self.parameters
        model.kpis = [
            KPISpecification(auto_scale=False),
            KPISpecification(auto_scale=False),
        ]

        evaluator = Workflow()
        evaluator.mco_model = model
        kpis = [DataValue(value=1), DataValue(value=2)]
        with self.assertTraitChanges(evaluator.mco_model,
                                     "event",
                                     count=model.num_points):
            with mock.patch(
                "force_bdss.api.Workflow.execute", return_value=kpis
            ) as mock_exec:
                mco.run(evaluator)
                self.assertEqual(54, mock_exec.call_count)
