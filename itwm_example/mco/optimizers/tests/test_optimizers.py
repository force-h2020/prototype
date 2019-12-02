from unittest import TestCase

from force_bdss.api import KPISpecification

from itwm_example.mco.mco_factory import MCOFactory
from itwm_example.mco.optimizers.optimizers import WeightedOptimizer
from itwm_example.mco.space_sampling.space_samplers import (
    UniformSpaceSampler,
    DirichletSpaceSampler,
)
from itwm_example.mco.tests.mock_classes import MockOptimizer


class TestWeightedOptimizer(TestCase):
    def setUp(self):
        self.plugin = {"id": "pid", "name": "Plugin"}
        self.factory = MCOFactory(self.plugin)
        self.mco_model = self.factory.create_model()

        self.kpis = [KPISpecification(), KPISpecification()]
        self.parameters = [1, 1, 1, 1]

        self.mco_model.kpis = self.kpis
        self.mco_model.parameters = [
            self.factory.parameter_factories[0].create_model()
            for _ in self.parameters
        ]

        self.optimizer = WeightedOptimizer(None, self.mco_model)

    def test__space_search_distribution(self):
        for strategy, klass in (
            ("Uniform", UniformSpaceSampler),
            ("Dirichlet", DirichletSpaceSampler),
            ("Uniform", UniformSpaceSampler),
        ):
            self.mco_model.space_search_mode = strategy
            distribution = self.optimizer._space_search_distribution()
            self.assertIsInstance(distribution, klass)
            self.assertEqual(len(self.kpis), distribution.dimension)
            self.assertEqual(7, distribution.resolution)

    def test_auto_scale(self):
        temp_kpis = [KPISpecification(), KPISpecification(auto_scale=False)]
        self.mco_model.kpis = temp_kpis

        mock_optimizer = MockOptimizer(None, None)
        self.optimizer._weighted_optimize = mock_optimizer.optimize

        scaling_factors = self.optimizer.get_scaling_factors()
        self.assertEqual([0.1, 1.0], scaling_factors)

    def test_scaling_factors(self):
        mock_optimizer = MockOptimizer(None, None)
        self.optimizer._weighted_optimize = mock_optimizer.optimize
        scaling_factors = self.optimizer.get_scaling_factors()
        self.assertEqual([0.1, 0.1], scaling_factors)
