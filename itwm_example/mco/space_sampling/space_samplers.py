import abc
import numpy as np

from traits.api import ABCHasStrictTraits, Bool, ListFloat, Float
from force_bdss.api import PositiveInt


def convert_samples_pp_to_samples_total(space_dimension, nof_points):
    samples_total = (
            np.math.factorial(space_dimension + nof_points - 2)
            / np.math.factorial(space_dimension - 1)
            / np.math.factorial(nof_points - 1)
    )
    return int(samples_total)


class SpaceSampler(ABCHasStrictTraits):
    dimension = PositiveInt()

    resolution = PositiveInt()

    def __init__(self, dimension, resolution, **kwargs):
        self.dimension = dimension
        self.resolution = resolution

    @abc.abstractmethod
    def _get_sample_point(self):
        pass

    @abc.abstractmethod
    def generate_space_sample(self, *args, **kwargs):
        pass


class DirichletSpaceSampler(SpaceSampler):
    _centering_coef = Float()
    alpha = ListFloat()

    distribution_function = np.random.dirichlet

    def __init__(self, dimension, resolution, alpha=None, **kwargs):
        super().__init__(dimension, resolution, **kwargs)

        if alpha is None:
            self._centering_coef = 1.
        else:
            self._centering_coef = alpha

        self.alpha = [self._centering_coef for _ in range(self.dimension)]

    def _get_sample_point(self):
        return self.distribution_function(self.alpha).tolist()

    def generate_space_sample(self):
        total_nof_points = convert_samples_pp_to_samples_total(
            self.dimension,
            self.resolution
        )
        for _ in range(total_nof_points):
            yield self._get_sample_point()


class UniformSpaceSampler(SpaceSampler):
    with_zero_values = Bool()

    def __init__(self, dimension, resolution, with_zero_values=True, **kwargs):
        super().__init__(dimension, resolution, **kwargs)

        self.with_zero_values = with_zero_values

    def generate_space_sample(self, **kwargs):
        yield from self._get_sample_point()

    def _get_sample_point(self):
        scaling = 1.0 / (self.resolution - 1)
        for int_w in self._int_weights():
            yield [scaling * val for val in int_w]

    def _int_weights(self, resolution=None, dimension=None):
        """Helper routine for the previous one. The meaning is the same, but
        works with integers instead of floats, adding up to num_points"""

        if dimension is None:
            dimension = self.dimension

        if resolution is None:
            resolution = self.resolution

        if dimension == 1:
            yield [resolution - 1]
        else:
            if self.with_zero_values:
                integers = np.arange(resolution-1, -1, -1)
            else:
                integers = np.arange(resolution-2, 0, -1)
            for i in integers:
                for entry in self._int_weights(
                        resolution - i,
                        dimension - 1
                        ):
                    yield [i] + entry
