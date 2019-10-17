import abc
import numpy as np


def convert_samples_pp_to_samples_total(space_dimension, nof_points):
    samples_total = (
            np.math.factorial(space_dimension + nof_points - 2)
            / np.math.factorial(space_dimension - 1)
            / np.math.factorial(nof_points - 1)
    )
    return int(samples_total)


class SpaceSampler(abc.ABC):
    def __init__(self, dimension, **kwargs):
        self.dimension = dimension

    @abc.abstractmethod
    def _get_sample_point(self):
        pass

    def generate_space_sample(self, nof_points):
        for _ in range(
                convert_samples_pp_to_samples_total(
                    self.dimension,
                    nof_points
                )
        ):
            yield self._get_sample_point()


class DirichletSpaceSampler(SpaceSampler):
    distribution_function = np.random.dirichlet

    def __init__(self, dimension, alpha=None, **kwargs):
        super().__init__(dimension, **kwargs)
        if alpha is None:
            self._centering_coef = 1.
        else:
            self._centering_coef = alpha

        self.alpha = np.ones(dimension) * self._centering_coef

    def _get_sample_point(self):
        return self.distribution_function(self.alpha).tolist()
