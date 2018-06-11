import subprocess
import sys
import collections
from traits.api import HasStrictTraits, List, Float
import numpy as np
from scipy import optimize

from force_bdss.api import BaseMCO
from force_bdss.mco.parameters.base_mco_parameter import BaseMCOParameter


def rotated_range(start, stop, starting_value):
    """Produces a range of integers, then rotates so that the starting value
    is starting_value"""
    r = list(range(start, stop))
    start_idx = r.index(starting_value)
    d = collections.deque(r)
    d.rotate(-start_idx)
    return list(d)


class MCO(BaseMCO):
    NUM_POINTS = 7

    def run(self, model):
        parameters = model.parameters
        kpis = model.kpis

        weight_combinations = get_weight_combinations(len(kpis),
                                                      self.NUM_POINTS)


        self.started = True

        for weights in weight_combinations:
            evaluator = WeightedEvaluator(weights, parameters)
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

    def calculate_singlepoint(self, in_values):
        application = self.factory.plugin.application

        ps = subprocess.Popen(
            [sys.argv[0],
             "--evaluate",
             application.workflow_filepath],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE)

        out = ps.communicate(
            " ".join([str(v) for v in in_values]).encode("utf-8"))

        return out[0].decode("utf-8").split()


class WeightedEvaluator(HasStrictTraits):
    weights = List(Float)
    parameters = List(BaseMCOParameter)

    def __init__(self, weights, parameters):
        super(WeightedEvaluator, self).__init__(
            weights=weights, parameters=parameters)

    def score(self, point):
        return np.dot(self.weights, self.calculate_singlepoint(point))

    def optimize(self):
        initial_point = [p.initial_value for p in self.parameters]
        constraints = [(p.lower_bound, p.upper_bound) for p in self.parameters]

        weighted_score_func = self.score

        optimal_point = opt(weighted_score_func, initial_point, constraints)
        optimal_kpis = self.calculate_singlepoint(optimal_point)

        return (optimal_point, optimal_kpis)


def opt(weighted_score_func,
        initial_point,
        constraints):

    return optimize.minimize(
        weighted_score_func,
        initial_point,
        method="SLSQP",
        bounds=constraints).x


def get_weight_combinations(dimension, num_points):
    scaling = 1.0 / (num_points)
    for int_w in _int_weights(dimension, num_points):
        yield [scaling * val for val in int_w]


def _int_weights(dimension, num_points):
    if dimension == 1:
        yield [num_points]
    else:
        for i in list(range(num_points, -1, -1)):
            for entry in _int_weights(dimension - 1, num_points - i):
                yield [i] + entry

