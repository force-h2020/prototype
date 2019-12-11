from unittest import mock, TestCase

from force_bdss.api import (
    KPISpecification,
    Workflow,
    DataValue,
    WorkflowEvaluator,
)

from itwm_example.mco.mco_factory import MCOFactory
from itwm_example.mco.optimizers.optimizers import (
    WeightedOptimizer,
    NevergradOptimizer,
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

    def test_init_model(self):
        model = self.factory.create_model()
        self.assertEqual([], model.parameters)
        self.assertEqual([], model.kpis)
        self.assertEqual("Weighted", model.optimizer_mode)
        self.assertIsInstance(model.optimizer, WeightedOptimizer)
        self.assertListEqual(model.parameters, model.optimizer.parameters)
        self.assertListEqual(model.kpis, model.optimizer.kpis)
        self.assertEqual("SLSQP", model.optimizer.algorithms)

        kpis = [KPISpecification(), KPISpecification()]
        parameters = [
            self.factory.parameter_factories[0].create_model()
            for _ in [1, 1, 1, 1]
        ]
        optimizer_data = {"algorithms": "TNC", "num_points": 10}
        model = self.factory.create_model(
            {
                "kpis": kpis,
                "parameters": parameters,
                "optimizer_data": optimizer_data,
            }
        )
        self.assertListEqual(parameters, model.parameters)
        self.assertListEqual(kpis, model.kpis)
        self.assertEqual("Weighted", model.optimizer_mode)
        self.assertIsInstance(model.optimizer, WeightedOptimizer)
        self.assertListEqual(model.parameters, model.optimizer.parameters)
        self.assertListEqual(model.kpis, model.optimizer.kpis)
        self.assertEqual("TNC", model.optimizer.algorithms)
        self.assertEqual(10, model.optimizer.num_points)

        optimizer_mode = "NeverGrad"
        optimizer_data = {"algorithms": "RandomSearch", "budget": 1000}
        model = self.factory.create_model(
            {
                "kpis": kpis,
                "parameters": parameters,
                "optimizer_mode": optimizer_mode,
                "optimizer_data": optimizer_data,
            }
        )
        self.assertListEqual(parameters, model.parameters)
        self.assertListEqual(kpis, model.kpis)
        self.assertEqual("NeverGrad", model.optimizer_mode)
        self.assertIsInstance(model.optimizer, NevergradOptimizer)
        self.assertListEqual(model.parameters, model.optimizer.parameters)
        self.assertListEqual(model.kpis, model.optimizer.kpis)
        self.assertEqual("RandomSearch", model.optimizer.algorithms)
        self.assertEqual(1000, model.optimizer.budget)

    def test_basic_eval(self):
        mock_kpi_return = [DataValue(value=2), DataValue(value=3)]

        with mock.patch(
            "force_bdss.api.Workflow.execute", return_value=mock_kpi_return
        ):
            self.mco.run(self.evaluator)

    def test_internal_weighted_evaluator(self):
        self.mco_model.optimizer.single_point_evaluator = self.evaluator
        mock_kpi_return = [DataValue(value=2), DataValue(value=3)]

        with mock.patch(
            "force_bdss.api.Workflow.execute", return_value=mock_kpi_return
        ) as mock_exec:
            self.mco_model.optimizer._weighted_optimize([0.5, 0.5])
            self.assertEqual(7, mock_exec.call_count)

    def test_nevergrad_optimizer(self):
        self.mco_model.optimizer_mode = "NeverGrad"
        self.mco_model.optimizer.single_point_evaluator = self.evaluator
        mock_kpi_return = [DataValue(value=0.1), DataValue(value=0.2)]

        with mock.patch(
            "force_bdss.api.Workflow.execute", return_value=mock_kpi_return
        ) as mock_exec:
            for point, value, _ in self.mco_model.optimizer.optimize():
                self.assertListEqual([0.1, 0.2], list(value))
            self.assertEqual(
                self.mco_model.optimizer.budget,
                mock_exec.call_count,
            )

        mock_kpi_return = [DataValue(value=2), DataValue(value=3)]

        with mock.patch(
            "force_bdss.api.Workflow.execute", return_value=mock_kpi_return
        ) as mock_exec:
            self.assertEqual(0, len(list(self.mco_model.optimizer.optimize())))
            self.assertEqual(
                self.mco_model.optimizer.budget,
                mock_exec.call_count,
            )

    def test_default_optimizer(self):
        self.assertIsInstance(self.mco_model.optimizer, WeightedOptimizer)

    def test_update_optimizer(self):
        self.mco_model.optimizer_mode = "NeverGrad"
        self.assertIs(
            self.mco_model._optimizer_from_mode(), NevergradOptimizer
        )
        self.assertIsInstance(self.mco_model.optimizer, NevergradOptimizer)
        self.assertEqual(self.mco_model.optimizer.algorithms, "TwoPointsDE")
        self.mco_model.optimizer_mode = "Weighted"
        self.assertIs(self.mco_model._optimizer_from_mode(), WeightedOptimizer)
        self.assertIsInstance(self.mco_model.optimizer, WeightedOptimizer)
        self.assertEqual(self.mco_model.optimizer.algorithms, "SLSQP")

    def test_update_kpis(self):
        new_kpis = [
            KPISpecification(name="new"),
            KPISpecification(name="another_new"),
        ]
        self.mco_model.kpis = new_kpis
        self.assertEqual(self.mco_model.kpis, new_kpis)
        self.assertEqual(self.mco_model.optimizer.kpis, new_kpis)

    def test_update_parameters(self):
        new_parameters = [2, 2, 2]
        new_parameters = [
            self.factory.parameter_factories[0].create_model()
            for _ in new_parameters
        ]
        self.mco_model.parameters = new_parameters
        self.assertEqual(self.mco_model.parameters, new_parameters)
        self.assertEqual(self.mco_model.optimizer.parameters, new_parameters)

    def test___getstate__(self):
        state = self.mco_model.__getstate__()
        self.assertEqual("Weighted", state["optimizer_mode"])
        self.assertEqual("Weighted_Optimizer", state["optimizer_data"]["name"])
        self.assertEqual("SLSQP", state["optimizer_data"]["algorithms"])

        self.mco_model.optimizer_mode = "NeverGrad"
        state = self.mco_model.__getstate__()
        self.assertEqual("NeverGrad", state["optimizer_mode"])
        self.assertEqual("Nevergrad", state["optimizer_data"]["name"])
        self.assertEqual("TwoPointsDE", state["optimizer_data"]["algorithms"])
