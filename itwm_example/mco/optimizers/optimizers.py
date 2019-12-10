import logging
from functools import partial

import numpy as np
from scipy import optimize as scipy_optimize
import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction

from traits.api import Interface, HasTraits, provides, Instance, Unicode, Enum
from traitsui.api import View, Item, Group

from force_bdss.io.workflow_writer import pop_dunder_recursive
from force_bdss.mco.i_evaluator import IEvaluator
from force_bdss.api import PositiveInt

from itwm_example.mco.scaling_tools.kpi_scaling import sen_scaling_method
from itwm_example.mco.space_sampling.space_samplers import (
    UniformSpaceSampler,
    DirichletSpaceSampler,
)

log = logging.getLogger(__name__)


class NevergradTypeError(Exception):
    pass


class IOptimizer(Interface):
    """" Generic optimizer interface."""

    def _score(self, *args, **kwargs):
        """ Objective function score with given parameters"""

    def optimize(self):
        """ Perform an optimization procedure"""

    def __getstate__(self):
        """ Serialization of the optimizer state"""


@provides(IOptimizer)
class WeightedOptimizer(HasTraits):
    """Performs a scipy optimise with SLSQP method given a set of weights
    for the individual KPIs.
    """

    #: Optimizer name
    name = Unicode("Weighted_Optimizer")

    single_point_evaluator = Instance(IEvaluator)

    #: Algorithms available to work with
    algorithms = Enum("SLSQP", "TNC")

    scaling_method = staticmethod(sen_scaling_method)

    #: Search grid resolution per KPI
    num_points = PositiveInt(7)

    #: Space search distribution for weight points sampling
    space_search_mode = Enum("Uniform", "Dirichlet")

    def default_traits_view(self):
        return View(
            Group(
                Item("name", style="readonly"),
                Item("algorithms"),
                Item("num_points"),
                Item("space_search_mode"),
            )
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

        #: Get default scaling weights for each KPI variable
        default_scaling_factors = np.array(
            [kpi.scale_factor for kpi in self.kpis]
        )

        #: Apply a wrapper for the evaluator weights assignment and
        #: call of the .optimize method.
        #: Then, calculate scaling factors defined by the `scaling_method`
        scaling_factors = scaling_method(
            len(self.kpis), self._weighted_optimize
        )

        #: Apply the scaling factors where necessary
        auto_scales = [kpi.auto_scale for kpi in self.kpis]
        default_scaling_factors[auto_scales] = scaling_factors[auto_scales]

        log.info(
            "Using KPI scaling factors: {}".format(default_scaling_factors)
        )

        return default_scaling_factors.tolist()

    def _space_search_distribution(self, **kwargs):
        """ Generates space search distribution object, based on
        the user settings of the `space_search_strategy` trait."""

        if self.space_search_mode == "Uniform":
            distribution = UniformSpaceSampler
        elif self.space_search_mode == "Dirichlet":
            distribution = DirichletSpaceSampler
        else:
            raise NotImplementedError
        return distribution(len(self.kpis), self.num_points, **kwargs)

    def weights_samples(self, **kwargs):
        """ Generates necessary number of search space sample points
        from the internal search strategy."""
        return self._space_search_distribution(
            **kwargs
        ).generate_space_sample()

    def optimize(self):
        """ Generates optimization results with weighted optimization.

        Yields
        ----------
        optimization result: tuple(np.array, np.array, list)
            Point of evaluation, objective value, dummy list of weights
        """
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
        """ Performs single scipy.minimize operation on the convolution of
        the multiobjective function with `weights`.

        Parameters
        ----------
        weights: List[Float]
            Weights for each KPI objective

        Returns
        ----------
        optimization result: tuple(np.array, np.array)
            Point of evaluation, and objective values
        """
        initial_point = [p.initial_value for p in self.parameters]
        bounds = [(p.lower_bound, p.upper_bound) for p in self.parameters]

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

    def __getstate__(self):
        state_data = pop_dunder_recursive(super().__getstate__())
        state_data.pop("kpis")
        state_data.pop("parameters")
        return state_data


@provides(IOptimizer)
class NevergradOptimizer(HasTraits):
    single_point_evaluator = Instance(IEvaluator)

    #: Optimizer name
    name = Unicode("Nevergrad")

    #: Algorithms available to work with
    algorithms = Enum(*ng.optimizers.registry.keys())

    #: Optimization budget defines the allowed number of objective calls
    budget = PositiveInt(100)

    def _algorithms_default(self):
        return "TwoPointsDE"

    def default_traits_view(self):
        return View(
            Item("name", style="readonly"), Item("algorithms"), Item("budget")
        )

    def _create_instrumentation_variable(self, parameter):
        """ Create nevergrad.variable from `MCOParameter`. Different
        MCOParameter subclasses have different signature attributes.
        The mapping between MCOParameters and nevergrad types is bijective.

        Parameters
        ----------
        parameter: BaseMCOParameter
            object to convert to nevergrad type

        Returns
        ----------
        nevergrad_parameter: nevergrad.Variable
            nevergrad variable of corresponding type
        """
        if hasattr(parameter, "lower_bound") and hasattr(
            parameter, "upper_bound"
        ):
            return ng.var.Scalar().bounded(
                parameter.lower_bound, parameter.upper_bound
            )
        elif hasattr(parameter, "value"):
            return ng.var._Constant(value=parameter.value)
        elif hasattr(parameter, "levels"):
            return ng.var.OrderedDiscrete(parameter.sample_values)
        elif hasattr(parameter, "categories"):
            return ng.var.SoftmaxCategorical(
                possibilities=parameter.sample_values, deterministic=True
            )
        else:
            raise NevergradTypeError(
                f"Can not convert {parameter} to any of"
                " supported Nevergrad types"
            )

    def _assemble_instrumentation(self, parameters=None):
        """ Assemble nevergrad.Instrumentation object from `parameters` list.

        Parameters
        ----------
        parameters: List(BaseMCOParameter)
            parameter objects containing lower and upper numerical bounds

        Returns
        ----------
        instrumentation: ng.Instrumentation
        """
        if parameters is None:
            parameters = self.parameters

        instrumentation = [
            self._create_instrumentation_variable(p) for p in parameters
        ]
        return ng.Instrumentation(*instrumentation)

    def _create_kpi_bounds(self, kpis=None):
        """ Assemble optimization bounds on KPIs, provided by
        `scaled_factor` attributes.
        Note: Ideally, a different kpi attribute should be
        responsible for the bounds.

        Parameters
        ----------
        kpis: List(KPISpecification)
            kpi objects containing upper numerical bounds

        Returns
        ----------
        upper_bounds: np.array
            kpis upper bounds
        """
        if kpis is None:
            kpis = self.kpis
        upper_bounds = np.zeros(len(kpis))
        for i, kpi in enumerate(kpis):
            try:
                upper_bounds[i] = kpi.scale_factor
            except AttributeError:
                upper_bounds[i] = 100
        return upper_bounds

    def _score(self, point):
        score = self.single_point_evaluator.evaluate(point)
        log.info("Objective score: {}".format(score))
        return score

    def optimize(self):
        """ Constructs objects required by the nevergrad engine to
        perform optimization.

        Yields
        ----------
        optimization result: tuple(np.array, np.array, list)
            Point of evaluation, objective value, dummy list of weights
        """
        upper_bounds = self._create_kpi_bounds()
        f = MultiobjectiveFunction(
            multiobjective_function=self._score, upper_bounds=upper_bounds
        )
        instrumentation = self._assemble_instrumentation()
        instrumentation.random_state.seed(12)
        ng_optimizer = ng.optimizers.registry[self.algorithms](
            instrumentation=instrumentation, budget=self.budget
        )
        for _ in range(ng_optimizer.budget):
            x = ng_optimizer.ask()
            value = f.multiobjective_function(x.args)
            volume = f.compute_aggregate_loss(value, *x.args, **x.kwargs)
            ng_optimizer.tell(x, volume)

        for point, value in f._points:
            yield point[0], value, [1] * len(self.kpis)

    def __getstate__(self):
        state_data = pop_dunder_recursive(super().__getstate__())
        state_data.pop("kpis")
        state_data.pop("parameters")
        return state_data
