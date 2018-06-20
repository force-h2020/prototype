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
    NUM_POINTS = 7

    def run(self, model):
        parameters = model.parameters
        kpis = model.kpis

        weight_combinations = get_weight_combinations(len(kpis),
                                                      self.NUM_POINTS)

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
        ps = subprocess.Popen(cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        out = ps.communicate(
            " ".join([str(v) for v in in_values]).encode("utf-8"))

        return [float(x) for x in out[0].decode("utf-8").split()]


class WeightedEvaluator(HasStrictTraits):
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
        return np.dot(
            self.weights,
            self.single_point_evaluator.evaluate(point))

    def optimize(self):
        initial_point = [p.initial_value for p in self.parameters]
        constraints = [(p.lower_bound, p.upper_bound) for p in self.parameters]

        weighted_score_func = self._score

        log.info("Running optimisation.")
        log.info("Initial point: {}".format(initial_point))
        log.info("Constraints: {}".format(constraints))
        optimal_point = opt(weighted_score_func, initial_point, constraints)
        optimal_kpis = self.single_point_evaluator.evaluate(optimal_point)

        return optimal_point, optimal_kpis


def opt(weighted_score_func,
        initial_point,
        constraints):

    return optimize.minimize(
        weighted_score_func,
        initial_point,
        method="SLSQP",
        bounds=constraints).x


def get_weight_combinations(dimension, num_points):
    scaling = 1.0 / (num_points - 1)
    for int_w in _int_weights(dimension, num_points):
        yield [scaling * val for val in int_w]


def _int_weights(dimension, num_points):
    if dimension == 1:
        yield [num_points - 1]
    else:
        for i in list(range(num_points-1, -1, -1)):
            for entry in _int_weights(dimension - 1, num_points - i):
                yield [i] + entry
