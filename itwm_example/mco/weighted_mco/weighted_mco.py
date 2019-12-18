import logging
import sys

from force_bdss.api import BaseMCO, DataValue, WeightedOptimizerEngine

from itwm_example.mco.subprocess_workflow_evaluator import (
    SubprocessWorkflowEvaluator,
)


log = logging.getLogger(__name__)


class WeightedMCO(BaseMCO):
    def run(self, evaluator):

        model = evaluator.mco_model

        if model.evaluation_mode == "Subprocess":
            # Here we create an instance of our WorkflowEvaluator subclass
            # that allows for evaluation of a state in the workflow via calling
            # `force_bdss` on a new subprocess running in 'evaluate' mode.
            # Note: a BaseMCOCommunicator must be present to pass in parameter
            # values and returning the KPI for a `force_bdss` run in 'evaluate'
            # mode
            evaluator = SubprocessWorkflowEvaluator(
                workflow=evaluator.workflow,
                workflow_filepath=evaluator.workflow_filepath,
                executable_path=sys.argv[0],
            )

        optimizer = WeightedOptimizerEngine(
            kpis=model.kpis,
            parameters=model.parameters,
            num_points=model.num_points,
            algorithms=model.algorithms,
            space_search_mode=model.space_search_mode,
            single_point_evaluator=evaluator,
            verbose_run=model.verbose_run,
        )

        for (
            optimal_point,
            optimal_kpis,
            scaled_weights,
        ) in optimizer.optimize():

            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            self.notify_new_point(
                [DataValue(value=v) for v in optimal_point],
                [DataValue(value=v) for v in optimal_kpis],
                scaled_weights,
            )
