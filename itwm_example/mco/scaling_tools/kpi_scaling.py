import logging
import numpy as np


log = logging.getLogger(__name__)


def sen_scaling_method(evaluator):
    """ Calculate the default Sen's scaling factors for the
    "Multi-Objective Programming Method" [1].

    References
    ----------
    .. [1] Chandra Sen, "Sen's Multi-Objective Programming Method and Its
       Comparison with Other Techniques", American Journal of Operational
       Research, vol. 8, pp. 10-13, 2018

    Parameters
    ----------
    evaluator: IOptimizer
        Instance that provides optimization functionality

    Returns
    -------
    scaling_factors: np.array
        Sen's scaling factors
    """
    dimension = len(evaluator.weights)
    extrema = np.zeros((dimension, dimension))

    initial_weights = np.eye(dimension)

    for i, weights in enumerate(initial_weights):

        evaluator.weights = weights.tolist()

        log.info(f"Doing extrema MCO run with weights: {evaluator.weights}")

        _, optimal_kpis = evaluator.optimize()
        extrema[i] += np.asarray(optimal_kpis)

    scaling_factors = np.reciprocal(extrema.max(0) - extrema.min(0))
    return scaling_factors


def get_scaling_factors(evaluator, kpis, scaling_method=sen_scaling_method):
    """KPI Scaling factors for MCO are calculated (as required) by
    normalising by the possible range of each optimal KPI value.
    Performs scaling for all KPIs that have `auto_scale == True`.
    Otherwise, keeps default scale factor.

     Parameters
    ----------
    evaluator: IOptimizer
        Instance that provides optimization functionality
    kpis: List[KPISpecification]
        List of KPI objects to scale
    scaling_method: callable
        A method to scale KPI weights
    """

    #: Get initial weights referring to extrema of each variable range
    auto_scales = [kpi.auto_scale for kpi in kpis]
    default_scaling_factors = np.array([kpi.scale_factor for kpi in kpis])

    #: Calculate default Sen's scaling factors
    scaling_factors = scaling_method(evaluator)

    #: Apply the Sen's scaling factors where necessary
    default_scaling_factors[auto_scales] = scaling_factors[auto_scales]

    log.info("Using KPI scaling factors: {}".format(default_scaling_factors))

    return default_scaling_factors.tolist()
