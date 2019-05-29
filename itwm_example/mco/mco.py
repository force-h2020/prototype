import sys
import subprocess
import logging
import numpy as np
from scipy import optimize

from traits.api import (
    HasStrictTraits, List, Float, Str, Instance, Interface, provides
)

from force_bdss.api import (
    BaseMCO, BaseMCOParameter, execute_workflow, Workflow, DataValue
)


log = logging.getLogger(__name__)


class MCO(BaseMCO):

    def run(self, model):
        parameters = model.parameters
        kpis = model.kpis

        application = self.factory.plugin.application
        if model.evaluation_mode == "Subprocess":
            single_point_evaluator = SubprocessSinglePointEvaluator(
                sys.argv[0], application.workflow_filepath
            )
        else:
            single_point_evaluator = InternalSinglePointEvaluator(
                application.workflow,
                model.parameters
            )

        #: Get scaling factors and non-zero weight combinations for each KPI
        scaling_factors = get_scaling_factors(single_point_evaluator,
                                              kpis,
                                              parameters)
        weight_combinations = get_weight_combinations(len(kpis),
                                                      model.num_points,
                                                      False)

        for weights in weight_combinations:

            log.info("Doing MCO run with weights: {}".format(weights))

            generator = zip(weights, scaling_factors)
            scaled_weights = [weight * scale for weight, scale in generator]

            evaluator = WeightedEvaluator(
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


class ISinglePointEvaluator(Interface):
    def evaluate(self, in_values):
        """"""


@provides(ISinglePointEvaluator)
class SubprocessSinglePointEvaluator(HasStrictTraits):
    """Evaluates a single point."""
    evaluation_executable_path = Str()
    workflow_filepath = Str()

    def __init__(self, evaluation_executable_path, workflow_filepath):
        super(SubprocessSinglePointEvaluator, self).__init__(
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


@provides(ISinglePointEvaluator)
class InternalSinglePointEvaluator(HasStrictTraits):
    workflow = Instance(Workflow)
    parameters = List()

    def __init__(self, workflow, parameters):
        super(InternalSinglePointEvaluator, self).__init__(
            workflow=workflow,
            parameters=parameters
        )

    def evaluate(self, in_values):
        value_names = [p.name for p in self.parameters]
        value_types = [p.type for p in self.parameters]

        # The values must be given a type. The MCO may pass raw numbers
        # with no type information. You are free to use metadata your MCO may
        # provide, but it is not mandatory that this data is available. You
        # can also use the model specification itself.
        # In any case, you must return a list of DataValue objects.
        data_values = [
            DataValue(type=type_, name=name, value=value)
            for type_, name, value in zip(
                value_types, value_names, in_values)]

        kpis = execute_workflow(self.workflow, data_values)

        return [kpi.value for kpi in kpis]


class WeightedEvaluator(HasStrictTraits):
    """Performs an optimization given a set of weights for the individual
    KPIs.
    """
    single_point_evaluator = Instance(ISinglePointEvaluator)
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


def opt(weighted_score_func, initial_point, constraints):
    """Partial func. Performs a scipy optimise with SLSQP method given the
    scoring function, the initial point, and a set of constraints."""

    return optimize.minimize(
        weighted_score_func,
        initial_point,
        method="SLSQP",
        bounds=constraints).x


def opt_acadopy(weighted_score_func, initial_point, constraints):
    """Partial func. Performs a acadopy optimise given the
    scoring function, the initial point, and a set of constraints."""

    pass


def get_weight_combinations(dimension, num_points, zero_values=True):
    """Given the number of dimensions, this function provides all possible
    combinations of weights adding to 1.0. For example, a dimension 3
    will give all combinations (x, y, z) where x+y+z = 1.0.

    The num_points parameter indicates how many divisions along a single
    dimension will be performed. For example num_points == 3 will evaluate
    for x being 0.0, 0.5 and 1.0. The returned (x, y, z) combinations will
    of course be much higher than 3.

    Note that if the zero_values parameter is set to false, then ensure
    num_points > dimension in order for the generator to return any values.

    Parameters
    ----------
    dimension: int
        The dimension of the vector

    num_points: int
        The number of divisions along each dimension

    zero_values: bool (default=True)
        Whether to include zero valued weights

    Returns
    -------
    generator
        A generator returning all the possible combinations satisfying the
        requirement that the sum of all the weights must always be 1.0
    """

    scaling = 1.0 / (num_points - 1)
    for int_w in _int_weights(dimension, num_points, zero_values):
        yield [scaling * val for val in int_w]


def _int_weights(dimension, num_points, zero_values):
    """Helper routine for the previous one. The meaning is the same, but
    works with integers instead of floats, adding up to num_points"""

    if dimension == 1:
        yield [num_points - 1]
    else:
        if zero_values:
            integers = np.arange(num_points-1, -1, -1)
        else:
            integers = np.arange(num_points-2, 0, -1)
        for i in integers:
            for entry in _int_weights(dimension - 1,
                                      num_points - i,
                                      zero_values):
                yield [i] + entry


def get_scaling_factors(single_point_evaluator, kpis, parameters):
    """KPI Scaling factors for MCO are calculated (as required) by
    normalising by the possible range of each optimal KPI value.
    Also known as Sen's Multi-Objective Programming Method[1]_.

    References
    ----------
    .. [1] Chandra Sen, "Sen's Multi-Objective Programming Method and Its
       Comparison with Other Techniques", American Journal of Operational
       Research, vol. 8, pp. 10-13, 2018
    """

    #: Get initial weights referring to extrema of each variable range
    auto_scales = [kpi.auto_scale for kpi in kpis]
    scaling_factors = [kpi.scale_factor for kpi in kpis]
    extrema = np.zeros((len(kpis), len(kpis)))
    initial_weights = get_weight_combinations(len(kpis), 2)

    #: Calculate extrema for each KPI optimisation
    for i, weights in enumerate(initial_weights):

        log.info("Doing extrema MCO run with weights: {}".format(weights))

        evaluator = WeightedEvaluator(
            single_point_evaluator,
            weights,
            parameters,
        )

        optimal_point, optimal_kpis = evaluator.optimize()
        extrema[i] += np.asarray(optimal_kpis)

    #: Calculate required scaling factors by normalising KPI range
    for i in np.argwhere(auto_scales).flatten():
        minimum = extrema[i][i]
        maximum = np.max(extrema[:, i])
        scaling_factors[i] = 1 / (maximum - minimum)

    log.info("Using KPI scaling factors: {}".format(scaling_factors))

    return scaling_factors
