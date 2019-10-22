import numpy as np

from traits.api import provides

from itwm_example.mco.optimizers.optimizers import IOptimizer


@provides(IOptimizer)
class MockOptimizer:
    def __init__(self, eval, param, **kwargs):
        self.dimension = 2
        self.margins = np.array([10.0 for _ in range(self.dimension)])
        self.min_values = np.array([i for i in range(self.dimension)])

        self.scaling_values = np.array([0.1] * self.dimension)

    def optimize(self, weights):
        return 0, self.min_values + weights * self.margins
