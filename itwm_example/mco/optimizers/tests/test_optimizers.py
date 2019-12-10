from unittest import TestCase, mock

import nevergrad as ng
from nevergrad.instrumentation.transforms import ArctanBound

from force_bdss.api import KPISpecification
from force_bdss.mco.parameters.mco_parameters import (
    FixedMCOParameterFactory,
    RangedMCOParameterFactory,
    ListedMCOParameterFactory,
    CategoricalMCOParameterFactory,
)

from itwm_example.mco.mco_factory import MCOFactory
from itwm_example.mco.space_sampling.space_samplers import (
    UniformSpaceSampler,
    DirichletSpaceSampler,
)
from itwm_example.mco.tests.mock_classes import MockOptimizer
from itwm_example.mco.optimizers.optimizers import (
    WeightedOptimizer,
    NevergradOptimizer,
    NevergradTypeError,
)


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

        self.optimizer = self.mco_model.optimizer

    def test_init(self):
        self.assertIsInstance(self.optimizer, WeightedOptimizer)
        self.assertEqual("Weighted_Optimizer", self.optimizer.name)
        self.assertIs(self.optimizer.single_point_evaluator, None)
        self.assertEqual("SLSQP", self.optimizer.algorithms)
        self.assertEqual(7, self.optimizer.num_points)
        self.assertEqual("Uniform", self.optimizer.space_search_mode)

    def test__space_search_distribution(self):
        for strategy, klass in (
            ("Uniform", UniformSpaceSampler),
            ("Dirichlet", DirichletSpaceSampler),
            ("Uniform", UniformSpaceSampler),
        ):
            self.optimizer.space_search_mode = strategy
            distribution = self.optimizer._space_search_distribution()
            self.assertIsInstance(distribution, klass)
            self.assertEqual(len(self.kpis), distribution.dimension)
            self.assertEqual(7, distribution.resolution)

    def test_scaling_factors(self):
        mock_optimizer = MockOptimizer(None, None)
        self.optimizer._weighted_optimize = mock_optimizer.optimize
        scaling_factors = self.optimizer.get_scaling_factors()
        self.assertEqual([0.1, 0.1], scaling_factors)

    def test_auto_scale(self):
        temp_kpis = [KPISpecification(), KPISpecification(auto_scale=False)]
        self.mco_model.kpis = temp_kpis

        mock_optimizer = MockOptimizer(None, None)
        self.optimizer._weighted_optimize = mock_optimizer.optimize

        scaling_factors = self.optimizer.get_scaling_factors()
        self.assertEqual([0.1, 1.0], scaling_factors)

    def test___getstate__(self):
        state = self.optimizer.__getstate__()
        self.assertDictEqual(
            {
                "name": "Weighted_Optimizer",
                "single_point_evaluator": None,
                "algorithms": "SLSQP",
                "num_points": 7,
                "space_search_mode": "Uniform",
            },
            state,
        )


class TestNevergradOptimizer(TestCase):
    def setUp(self):
        self.plugin = {"id": "pid", "name": "Plugin"}
        self.factory = MCOFactory(self.plugin)
        self.mco_model = self.factory.create_model()
        self.mco_model.optimizer_mode = "NeverGrad"

        self.kpis = [KPISpecification(), KPISpecification()]
        self.parameters = [1, 1, 1, 1]

        self.mco_model.kpis = self.kpis
        self.mco_model.parameters = [
            self.factory.parameter_factories[0].create_model()
            for _ in self.parameters
        ]

        self.optimizer = self.mco_model.optimizer

    def test_init(self):
        self.assertIsInstance(self.optimizer, NevergradOptimizer)
        self.assertEqual("Nevergrad", self.optimizer.name)
        self.assertIs(self.optimizer.single_point_evaluator, None)
        self.assertEqual("TwoPointsDE", self.optimizer.algorithms)
        self.assertEqual(500, self.optimizer.n_objective_calls)

    def test__create_instrumentation_variable(self):
        mock_factory = mock.Mock(
            spec=self.factory,
            plugin_id="pid",
            plugin_name="Plugin",
            id="mcoid",
        )
        fixed_variable = FixedMCOParameterFactory(mock_factory).create_model(
            data_values={"value": 42}
        )
        fixed_variable = self.optimizer._create_instrumentation_variable(
            fixed_variable
        )
        self.assertIsInstance(fixed_variable, ng.var._Constant)
        self.assertEqual(42, fixed_variable.value)

        ranged_variable = RangedMCOParameterFactory(mock_factory).create_model(
            data_values={"lower_bound": -1.0, "upper_bound": 3.14}
        )
        ranged_variable = self.optimizer._create_instrumentation_variable(
            ranged_variable
        )
        self.assertIsInstance(ranged_variable, ng.var.Scalar)
        self.assertEqual(1, len(ranged_variable.transforms))
        self.assertEqual(3.14, ranged_variable.transforms[0].a_max[0])
        self.assertEqual(-1.0, ranged_variable.transforms[0].a_min[0])
        self.assertIsInstance(ranged_variable.transforms[0], ArctanBound)

        listed_variable = ListedMCOParameterFactory(mock_factory).create_model(
            data_values={"levels": [2.0, 1.0, 0.0]}
        )
        listed_variable = self.optimizer._create_instrumentation_variable(
            listed_variable
        )
        self.assertIsInstance(listed_variable, ng.var.OrderedDiscrete)
        self.assertListEqual([0.0, 1.0, 2.0], listed_variable.possibilities)

        categorical_variable = CategoricalMCOParameterFactory(
            mock_factory
        ).create_model(data_values={"categories": ["2.0", "1.0", "0.0"]})
        categorical_variable = self.optimizer._create_instrumentation_variable(
            categorical_variable
        )
        self.assertIsInstance(categorical_variable, ng.var.SoftmaxCategorical)
        self.assertListEqual(
            ["2.0", "1.0", "0.0"], categorical_variable.possibilities
        )

        with self.assertRaises(NevergradTypeError):
            self.optimizer._create_instrumentation_variable(1)

    def test__create_instrumentation(self):
        instrumentation = self.optimizer._assemble_instrumentation()
        self.assertIsInstance(instrumentation, ng.Instrumentation)
        self.assertEqual(
            len(self.mco_model.parameters), len(instrumentation.args)
        )
        for i, parameter in enumerate(self.mco_model.parameters):
            self.assertListEqual(
                [parameter.upper_bound],
                list(instrumentation.args[i].transforms[0].a_max),
            )
            self.assertListEqual(
                [parameter.lower_bound],
                list(instrumentation.args[i].transforms[0].a_min),
            )

    def test__create_kpi_bounds(self):
        self.optimizer.kpis[0].scale_factor = 10
        bounds = self.optimizer._create_kpi_bounds()
        self.assertEqual(len(self.optimizer.kpis), len(bounds))
        for kpi, kpi_bound in zip(self.optimizer.kpis, bounds):
            self.assertEqual(kpi.scale_factor, kpi_bound)

    def test__swap_minmax_kpivalues(self):
        temp_kpis = [
            KPISpecification(),
            KPISpecification(objective="MAXIMISE"),
        ]
        self.mco_model.kpis = temp_kpis
        values = [10.0, 20.0]
        inv_values = self.mco_model.optimizer._swap_minmax_kpivalues(
            values
        )
        self.assertListEqual(list(inv_values), [10.0, -20.0])
