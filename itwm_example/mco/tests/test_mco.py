from unittest import mock, TestCase

from force_bdss.api import (
    KPISpecification,
    Workflow,
    DataValue,
    WorkflowEvaluator,
)

from itwm_example.mco.mco_factory import MCOFactory


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
        evaluator = self.mco.optimizer(
            single_point_evaluator=self.evaluator, model=self.mco_model
        )
        mock_kpi_return = [DataValue(value=2), DataValue(value=3)]

        with mock.patch(
            "force_bdss.api.Workflow.execute", return_value=mock_kpi_return
        ) as mock_exec:
            evaluator._weighted_optimize([0.5, 0.5])
            self.assertEqual(7, mock_exec.call_count)
