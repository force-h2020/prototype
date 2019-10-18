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
from .space_sampling.space_samplers import UniformSpaceSampler

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
        scaling_factors = get_scaling_factors(single_point_evaluator,
                                              kpis,
                                              parameters)
        weight_combinations = get_weight_combinations(len(kpis),
                                                      model.num_points,
                                                      with_zero_values=False)

        weights_generator = UniformSpaceSampler(
            len(kpis),
            model.num_points,
            with_zero_values=False
        ).generate_space_sample()

        for weights in weights_generator:

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


def get_weight_combinations(dimension, num_points, **kwargs):
    """ Given the problem dimension and _effective_ number of points,
    generates samples for the unit simplex.
    Parameters
    ----------
    dimension: int
        The dimension of the vector

    num_points: int
        The number of divisions along each dimension

    Returns
    -------
    generator
        A generator returning random samples of vector satisfying the
        requirement that the sum of all the elements always equal 1.0
    """

    generator = UniformSpaceSampler(dimension, num_points, **kwargs)
    yield from generator.generate_space_sample()


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

    #: Initialize a `WeightedEvaluator` for scaling calculations
    evaluator = WeightedEvaluator(
        single_point_evaluator,
        [1. for _ in kpis],
        parameters,
    )

    #: Get initial weights referring to extrema of each variable range
    auto_scales = [kpi.auto_scale for kpi in kpis]
    scaling_factors = np.array([kpi.scale_factor for kpi in kpis])

    #: Calculate default Sen's scaling factors
    sen_scaling_factors = generate_sen_scaling_factors(
        evaluator,
        len(auto_scales)
    )

    #: Apply the Sen's scaling factors where necessary
    scales_mask = np.argwhere(auto_scales).flatten()
    scaling_factors[scales_mask] = sen_scaling_factors[scales_mask]

    log.info("Using KPI scaling factors: {}".format(scaling_factors))

    return scaling_factors.tolist()


def generate_sen_scaling_factors(weighted_evaluator, dimension):
    """ Calculate the default Sen's scaling factors for the
    "Multi-Objective Programming Method".

    Parameters
    ----------
    weighted_evaluator: WeightedEvaluator
        Instance that provides optimization functionality
    dimension: int
        The dimension of the KPIs vector
    Returns
    -------
    scaling_factors: np.array
        Sen's scaling factors
    """
    extrema = np.zeros((dimension, dimension))

    initial_weights = np.eye(dimension)

    for i, weights in enumerate(initial_weights):

        weighted_evaluator.weights = weights.tolist()

        log.info(
            f"Doing extrema MCO run with weights: {weighted_evaluator.weights}"
        )

        _, optimal_kpis = weighted_evaluator.optimize()
        extrema[i] += np.asarray(optimal_kpis)

    scaling_factors = np.reciprocal(extrema.max(0) - extrema.min(0))
    return scaling_factors
