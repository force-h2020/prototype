from force_bdss.mco.optimizer_engines.weighted_optimizer_engine import (
    WeightedOptimizerEngine
)
from force_bdss.mco.optimizers.scipy_optimizer import ScipyOptimizer


class WeightedScipyEngine(ScipyOptimizer, WeightedOptimizerEngine):
    """ A priori (weighted) multi-objective optimization
    using the scipy optimizer.
    """
    pass
