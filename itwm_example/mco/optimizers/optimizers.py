import logging
from functools import partial

import numpy as np
from scipy import optimize as scipy_optimize
import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction

from traits.api import Interface, HasStrictTraits, provides, List, Instance

from force_bdss.api import BaseMCOParameter
from force_bdss.mco.i_evaluator import IEvaluator

from itwm_example.mco.mco_model import MCOModel

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

    parameters = List(BaseMCOParameter)

    def __init__(self, single_point_evaluator, parameters):
        super().__init__(
            single_point_evaluator=single_point_evaluator,
            parameters=parameters,
        )

    def _score(self, point, weights):

        score = np.dot(weights, self.single_point_evaluator.evaluate(point))

        log.info("Weighted score: {}".format(score))

        return score

    def optimize(self, weights):
        initial_point = [p.initial_value for p in self.parameters]
        constraints = [(p.lower_bound, p.upper_bound) for p in self.parameters]

        weighted_score_func = partial(self._score, weights=weights)

        log.info(
            "Running optimisation."
            + "Initial point: {}".format(initial_point)
            + "Constraints: {}".format(constraints)
        )
        optimal_point = opt(weighted_score_func, initial_point, constraints)
        optimal_kpis = self.single_point_evaluator.evaluate(optimal_point)

        log.info(
            "Optimal point : {}".format(optimal_point)
            + "KPIs at optimal point : {}".format(optimal_kpis)
        )

        return optimal_point, optimal_kpis


@provides(IOptimizer)
class NevergradOptimizer(HasStrictTraits):
    single_point_evaluator = Instance(IEvaluator)

    parameters = List(BaseMCOParameter)

    model = Instance(MCOModel)

    def __init__(self, single_point_evaluator, model):
        super().__init__(
            single_point_evaluator=single_point_evaluator, model=model
        )

    def _score(self, point):
        return self.single_point_evaluator.evaluate(point)

    def optimize(self):
        instrumentation = [
            ng.var.Scalar().bounded(p.lower_bound, p.upper_bound)
            for p in self.model.parameters
        ]
        instrumentation = ng.Instrumentation(*instrumentation)
        f = MultiobjectiveFunction(
            multiobjective_function=self._score,
            upper_bounds=[100] * len(self.model.kpis),  # [0.4, 50, 5000]
        )
        budget = 200
        ng_optimizer = ng.optimizers.registry["TwoPointsDE"](
            instrumentation=instrumentation, budget=budget
        )
        for _ in range(ng_optimizer.budget):
            x = ng_optimizer.ask()
            value = f.multiobjective_function(x.args)
            volume = f.compute_aggregate_loss(value, *x.args, **x.kwargs)
            ng_optimizer.tell(x, volume)
            yield x.args, value
