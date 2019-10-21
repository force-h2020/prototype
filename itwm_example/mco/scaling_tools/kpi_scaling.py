import logging
import numpy as np


log = logging.getLogger(__name__)


def sen_scaling_method(dimension, optimize):
    """ Calculate the default Sen's scaling factors for the
    "Multi-Objective Programming Method" [1].

    References
    ----------
    .. [1] Chandra Sen, "Sen's Multi-Objective Programming Method and Its
       Comparison with Other Techniques", American Journal of Operational
       Research, vol. 8, pp. 10-13, 2018

    Parameters
    ----------
    dimension: int
        Number of KPIs, used in the optimization process

    optimize: unbound method
        Callable function with `weights` as the argument, that wraps all
        internal processes of the optimizer / evaluator

    Returns
    -------
    scaling_factors: np.array
        Sen's scaling factors
    """
    extrema = np.zeros((dimension, dimension))

    initial_weights = np.eye(dimension)

    for i, weights in enumerate(initial_weights):

        log.info(f"Doing extrema MCO run with weights: {weights}")

        _, optimal_kpis = optimize(weights)
        extrema[i] += np.asarray(optimal_kpis)

    scaling_factors = np.reciprocal(extrema.max(0) - extrema.min(0))
    return scaling_factors


def kpi_scaling_factors(evaluator, kpis, scaling_method=sen_scaling_method):
    """Calculates scaling factors for KPIs, defined in MCO.
    Scaling factors are calculated (as required) by the provided scaling
    method. In general, this provides normalization values for the possible
    range of each KPI..
    Performs scaling for all KPIs that have `auto_scale == True`.
    Otherwise, keeps the default scale factor.

    Parameters
    ----------
    evaluator: IOptimizer
        Instance that provides optimization functionality
    kpis: List[KPISpecification]
        List of KPI objects to scale
    scaling_method: callable
        A method to scale KPI weights. Default set to the Sen's
        "Multi-Objective Programming Method".
    """

    #: Get default scaling weights for each KPI variable
    default_scaling_factors = np.array([kpi.scale_factor for kpi in kpis])

    #: Define a wrapper for the evaluator weights assignment and
    #: call of the .optimize method.
    def optimization_wrapper(weights):
        evaluator.weights = weights.tolist()
        return evaluator.optimize()

    #: Calculate scaling factors defined by the `scaling_method`
    scaling_factors = scaling_method(
        len(evaluator.weights), optimization_wrapper
    )

    #: Apply the scaling factors where necessary
    auto_scales = [kpi.auto_scale for kpi in kpis]
    default_scaling_factors[auto_scales] = scaling_factors[auto_scales]

    log.info("Using KPI scaling factors: {}".format(default_scaling_factors))

    return default_scaling_factors.tolist()
