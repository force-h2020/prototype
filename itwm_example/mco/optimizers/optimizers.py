import logging

import numpy as np
from scipy import optimize as scipy_optimize

from traits.api import (
    Interface,
    HasStrictTraits,
    provides,
    List,
    Float,
    Instance,
)

from force_bdss.api import BaseMCOParameter
from force_bdss.mco.i_evaluator import IEvaluator

log = logging.getLogger(__name__)


def opt(objective, initial_point, constraints):
    """Partial func. Performs a scipy optimise with SLSQP method given the
    objective function, the initial point, and a set of constraints."""

    return scipy_optimize.minimize(
        objective, initial_point, method="SLSQP", bounds=constraints
    ).x


class IOptimizer(Interface):
    def _score(self, *args, **kwargs):
        """ Objective function score with given parameters"""

    def optimize(self):
        """ Perform an optimization procedure"""


@provides(IOptimizer)
class WeightedOptimizer(HasStrictTraits):
    """Performs an optimization given a set of weights for the individual
    KPIs.
    """

    single_point_evaluator = Instance(IEvaluator)

    weights = List(Float)

    parameters = List(BaseMCOParameter)

    def __init__(self, single_point_evaluator, weights, parameters):
        super().__init__(
            single_point_evaluator=single_point_evaluator,
            weights=weights,
            parameters=parameters,
        )

    def _score(self, point):

        score = np.dot(
            self.weights, self.single_point_evaluator.evaluate(point)
        )

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
