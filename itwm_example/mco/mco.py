import sys
import subprocess
import logging
import numpy as np
from scipy import optimize

from traits.api import HasStrictTraits, List, Float, Str, Instance

from force_bdss.api import BaseMCO
from force_bdss.mco.parameters.base_mco_parameter import BaseMCOParameter


log = logging.getLogger(__name__)


class MCO(BaseMCO):

    def run(self, model):
        parameters = model.parameters
        kpis = model.kpis

        weight_combinations = get_weight_combinations(len(kpis),
                                                      model.num_points)

        application = self.factory.plugin.application
        single_point_evaluator = SinglePointEvaluator(
            sys.argv[0], application.workflow_filepath
        )

        self.started = True

        for weights in weight_combinations:
            log.info("Doing MCO run with weights: {}".format(weights))

            evaluator = WeightedEvaluator(
                single_point_evaluator,
                weights,
                parameters,
            )
            optimal_point, optimal_kpis = evaluator.optimize()
            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            self.new_data = {
                'input': tuple(optimal_point),
                'output': tuple(optimal_kpis)
            }

        # To inform the rest of the system that the evaluation has completed.
        # we set this event to True
        self.finished = True


class SinglePointEvaluator(HasStrictTraits):
    """Evaluates a single point."""
    evaluation_executable_path = Str()
    workflow_filepath = Str()

    def __init__(self, evaluation_executable_path, workflow_filepath):
        super(SinglePointEvaluator, self).__init__(
            evaluation_executable_path=evaluation_executable_path,
            workflow_filepath=workflow_filepath
        )

    def evaluate(self, in_values):
        cmd = [self.evaluation_executable_path,
               "--logfile",
               "bdss.log",
               "--evaluate",
               self.workflow_filepath]

        log.info("Spawning subprocess: {}".format(cmd))
        ps = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        log.info("Sending values: {}".format([str(v) for v in in_values]))

        out = ps.communicate(
            " ".join([str(v) for v in in_values]).encode("utf-8"))

        log.info(
            "Received values: {}".format(
                [x for x in out[0].decode("utf-8").split()]))

        return [float(x) for x in out[0].decode("utf-8").split()]


class WeightedEvaluator(HasStrictTraits):
    """Performs an optimization given a set of weights for the individual
    KPIs.
    """
    single_point_evaluator = Instance(SinglePointEvaluator)
    weights = List(Float)
    parameters = List(BaseMCOParameter)

    def __init__(self, single_point_evaluator, weights, parameters):
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


def opt(weighted_score_func, initial_point, constraints):
    """Partial func. Performs a scipy optimise with SLSQP method given the
    scoring function, the initial point, and a set of constraints."""

    return optimize.minimize(
        weighted_score_func,
        initial_point,
        method="SLSQP",
        bounds=constraints).x


def get_weight_combinations(dimension, num_points):
    """Given the number of dimensions, this function provides all possible
    combinations of weights adding to 1.0. For example, a dimension 3
    will give all combinations (x, y, z) where x+y+z = 1.0.

    The num_points parameter indicates how many divisions along a single
    dimension will be performed. For example num_points == 3 will evaluate
    for x being 0.0, 0.5 and 1.0. The returned (x, y, z) combinations will
    of course be much higher than 3.

    Parameters
    ----------
    dimension: int
        The dimension of the vector

    num_points: int
        The number of divisions along each dimension

    Returns
    -------
    generator
        A generator returning all the possible combinations satisfying the
        requirement that the sum of all the weights must always be 1.0
    """
    scaling = 1.0 / (num_points - 1)
    for int_w in _int_weights(dimension, num_points):
        yield [scaling * val for val in int_w]


def _int_weights(dimension, num_points):
    """Helper routine for the previous one. The meaning is the same, but
    works with integers instead of floats, adding up to num_points"""
    if dimension == 1:
        yield [num_points - 1]
    else:
        for i in list(range(num_points-1, -1, -1)):
            for entry in _int_weights(dimension - 1, num_points - i):
                yield [i] + entry
