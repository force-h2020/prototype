import unittest

from itwm_example.mco.mco_factory import MCOFactory
from itwm_example.mco.space_sampling.space_samplers import (
    UniformSpaceSampler,
    DirichletSpaceSampler,
)


class TestMCOModel(unittest.TestCase):
    def setUp(self):
        self.plugin = {"id": "pid", "name": "Plugin"}
        self.factory = MCOFactory(self.plugin)

        self.num_points = 7
        self.kpis = [None, None]

        self.mco_model = self.factory.create_model()
        self.mco_model.kpis = self.kpis
        self.mco_model.num_points = self.num_points

    def test__space_search_distribution(self):
        for strategy, klass in (
            ("Uniform", UniformSpaceSampler),
            ("Dirichlet", DirichletSpaceSampler),
            ("Uniform", UniformSpaceSampler),
        ):
            self.mco_model.space_search_strategy = strategy
            distribution = self.mco_model._space_search_distribution()
            self.assertIsInstance(distribution, klass)
            self.assertEqual(len(self.kpis), distribution.dimension)
            self.assertEqual(distribution.resolution, self.num_points)
