import logging
import numpy as np

log = logging.getLogger(__name__)


def get_scaling_factors(evaluator, kpis):
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
    scaling_factors = np.array([kpi.scale_factor for kpi in kpis])

    #: Calculate default Sen's scaling factors
    sen_scaling_factors = generate_sen_scaling_factors(evaluator)

    #: Apply the Sen's scaling factors where necessary
    scaling_factors[auto_scales] = sen_scaling_factors[auto_scales]

    log.info("Using KPI scaling factors: {}".format(scaling_factors))

    return scaling_factors.tolist()


def generate_sen_scaling_factors(evaluator):
    """ Calculate the default Sen's scaling factors for the
    "Multi-Objective Programming Method".

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
