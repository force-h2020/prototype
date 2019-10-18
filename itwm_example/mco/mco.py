import logging
import sys

import numpy as np
from scipy import optimize

from traits.api import (
    HasStrictTraits, List, Float, Instance, Interface, provides, Type
)

from force_bdss.api import (
    BaseMCO, BaseMCOParameter, DataValue
)
from force_bdss.mco.i_evaluator import IEvaluator

from .subprocess_workflow_evaluator import SubprocessWorkflowEvaluator
from .scaling_tools.kpi_scaling import get_scaling_factors

log = logging.getLogger(__name__)


class IOptimizer(Interface):

    def _score(self, *args, **kwargs):
        """ Objective function score with given parameters"""

    def optimize(self):
        """ Perform an optimization procedure"""


@provides(IOptimizer)
class WeightedEvaluator(HasStrictTraits):
    """Performs an optimization given a set of weights for the individual
    KPIs.
    """
    single_point_evaluator = Instance(IEvaluator)

    weights = List(Float)

    parameters = List(BaseMCOParameter)

    def __init__(self, single_point_evaluator, weights,
                 parameters):
        super(WeightedEvaluator, self).__init__(
            single_point_evaluator=single_point_evaluator,
            weights=weights,
            parameters=parameters,
        )

    def _score(self, point):

        score = np.dot(
            self.weights,
            self.single_point_evaluator.evaluate(point))

        log.info("Weighted score: {}".format(score))

        return score

    def optimize(self):
        initial_point = [p.initial_value for p in self.parameters]
        constraints = [(p.lower_bound, p.upper_bound) for p in self.parameters]

        weighted_score_func = self._score

        log.info("Running optimisation.")
        log.info("Initial point: {}".format(initial_point))
        log.info("Constraints: {}".format(constraints))
        optimal_point = opt(weighted_score_func, initial_point, constraints)
        optimal_kpis = self.single_point_evaluator.evaluate(optimal_point)
        log.info("Optimal point : {}".format(optimal_point))
        log.info("KPIs at optimal point : {}".format(optimal_kpis))

        return optimal_point, optimal_kpis


class MCO(BaseMCO):

    optimizer = Type(IOptimizer)

    def _optimizer_default(self):
        return WeightedEvaluator

    def get_scaling_factors(self, evaluator, kpis, parameters):
        """ Generate Sen's scaling factors for optimization."""
        scaling_factors_evaluator = self.optimizer(
            evaluator,
            [1. for _ in kpis],
            parameters,
        )
        scaling_factors = get_scaling_factors(
            scaling_factors_evaluator, kpis
        )
        return scaling_factors

    def run(self, evaluator):

        model = evaluator.mco_model
        parameters = model.parameters
        kpis = model.kpis

        if model.evaluation_mode == "Subprocess":
            # Here we create an instance of our WorkflowEvaluator subclass
            # that allows for evaluation of a state in the workflow via calling
            # force_bdss on a new subprocess running in 'evaluate' mode.
            # Note: a BaseMCOCommunicator must be present to pass in parameter
            # values and returning the KPI for a force_bdss run in 'evaluate'
            # mode
            single_point_evaluator = SubprocessWorkflowEvaluator(
                workflow=evaluator.workflow,
                workflow_filepath=evaluator.workflow_filepath,
                executable_path=sys.argv[0]
            )
        else:
            single_point_evaluator = evaluator

        #: Get scaling factors and non-zero weight combinations for each KPI
        scaling_factors = self.get_scaling_factors(
            single_point_evaluator, kpis, parameters
        )

        for weights in model.weights_samples(with_zero_values=False):

            log.info("Doing MCO run with weights: {}".format(weights))

            generator = zip(weights, scaling_factors)
            scaled_weights = [weight * scale for weight, scale in generator]

            evaluator = self.optimizer(
                single_point_evaluator,
                scaled_weights,
                parameters,
            )

            optimal_point, optimal_kpis = evaluator.optimize()
            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.

            self.notify_new_point(
                [DataValue(value=v) for v in optimal_point],
                [DataValue(value=v) for v in optimal_kpis],
                scaled_weights
            )


def opt(weighted_score_func, initial_point, constraints):
    """Partial func. Performs a scipy optimise with SLSQP method given the
    scoring function, the initial point, and a set of constraints."""

    return optimize.minimize(
        weighted_score_func,
        initial_point,
        method="SLSQP",
        bounds=constraints).x
