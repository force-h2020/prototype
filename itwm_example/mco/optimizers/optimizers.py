import logging
from functools import partial

import numpy as np
from scipy import optimize as scipy_optimize
import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction

from traits.api import (
    Interface,
    HasTraits,
    HasStrictTraits,
    provides,
    Instance,
)

from force_bdss.mco.i_evaluator import IEvaluator

from itwm_example.mco.mco_model import MCOModel
from itwm_example.mco.scaling_tools.kpi_scaling import sen_scaling_method
from itwm_example.mco.space_sampling.space_samplers import (
    UniformSpaceSampler,
    DirichletSpaceSampler,
)

log = logging.getLogger(__name__)


class IOptimizer(Interface):
    def _score(self, *args, **kwargs):
        """ Objective function score with given parameters"""

    def optimize(self):
        """ Perform an optimization procedure"""


@provides(IOptimizer)
class WeightedOptimizer(HasTraits):
    """Performs a scipy optimise with SLSQP method given a set of weights
    for the individual KPIs.
    """

    single_point_evaluator = Instance(IEvaluator)

    model = Instance(MCOModel)

    scaling_method = staticmethod(sen_scaling_method)

    def __init__(self, single_point_evaluator, model):
        super().__init__(
            single_point_evaluator=single_point_evaluator, model=model
        )

    def _score(self, point, weights):

        score = np.dot(weights, self.single_point_evaluator.evaluate(point))

        log.info("Weighted score: {}".format(score))

        return score

    def get_scaling_factors(self, scaling_method=None):
        """ Calculates scaling factors for KPIs, defined in MCO.
        Scaling factors are calculated (as required) by the provided scaling
        method. In general, this provides normalization values for the possible
        range of each KPI.
        Performs scaling for all KPIs that have `auto_scale == True`.
        Otherwise, keeps the default scale factor.

        Parameters
        ----------
        scaling_method: callable
            A method to scale KPI weights. Default set to the Sen's
            "Multi-Objective Programming Method"
        """
        if scaling_method is None:
            scaling_method = self.scaling_method

        kpis = self.model.kpis

        #: Get default scaling weights for each KPI variable
        default_scaling_factors = np.array([kpi.scale_factor for kpi in kpis])

        #: Apply a wrapper for the evaluator weights assignment and
        #: call of the .optimize method.
        #: Then, calculate scaling factors defined by the `scaling_method`
        scaling_factors = scaling_method(len(kpis), self._weighted_optimize)

        #: Apply the scaling factors where necessary
        auto_scales = [kpi.auto_scale for kpi in kpis]
        default_scaling_factors[auto_scales] = scaling_factors[auto_scales]

        log.info(
            "Using KPI scaling factors: {}".format(default_scaling_factors)
        )

        return default_scaling_factors.tolist()

    def _space_search_distribution(self, **kwargs):
        """ Generates space search distribution object, based on
        the user settings of the `space_search_strategy` trait."""

        if self.model.space_search_mode == "Uniform":
            distribution = UniformSpaceSampler
        elif self.model.space_search_mode == "Dirichlet":
            distribution = DirichletSpaceSampler
        else:
            raise NotImplementedError
        return distribution(
            len(self.model.kpis), self.model.num_points, **kwargs
        )

    def weights_samples(self, **kwargs):
        """ Generates necessary number of search space sample points
        from the internal search strategy."""
        return self._space_search_distribution(
            **kwargs
        ).generate_space_sample()

    def optimize(self):
        #: Get scaling factors and non-zero weight combinations for each KPI
        scaling_factors = self.get_scaling_factors()
        for weights in self.weights_samples():

            log.info("Doing MCO run with weights: {}".format(weights))

            scaled_weights = [
                weight * scale
                for weight, scale in zip(weights, scaling_factors)
            ]

            optimal_point, optimal_kpis = self._weighted_optimize(
                scaled_weights
            )
            yield optimal_point, optimal_kpis, scaled_weights

    def _weighted_optimize(self, weights):
        initial_point = [p.initial_value for p in self.model.parameters]
        bounds = [
            (p.lower_bound, p.upper_bound) for p in self.model.parameters
        ]

        log.info(
            "Running optimisation."
            + "Initial point: {}".format(initial_point)
            + "Bounds: {}".format(bounds)
        )

        weighted_score_func = partial(self._score, weights=weights)

        optimization_result = scipy_optimize.minimize(
            weighted_score_func, initial_point, method="SLSQP", bounds=bounds
        )
        optimal_point = optimization_result.x
        optimal_kpis = self.single_point_evaluator.evaluate(optimal_point)

        log.info(
            "Optimal point : {}".format(optimal_point)
            + "KPIs at optimal point : {}".format(optimal_kpis)
        )

        return optimal_point, optimal_kpis


@provides(IOptimizer)
class NevergradOptimizer(HasStrictTraits):
    single_point_evaluator = Instance(IEvaluator)

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
