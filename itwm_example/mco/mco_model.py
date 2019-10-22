from traits.api import Enum
from traitsui.api import View, Item
from force_bdss.api import BaseMCOModel, PositiveInt

from .space_sampling.space_samplers import (
    UniformSpaceSampler,
    DirichletSpaceSampler
)


class MCOModel(BaseMCOModel):
    num_points = PositiveInt(7)

    evaluation_mode = Enum("Internal", "Subprocess")

    space_search_mode = Enum("Uniform", "Dirichlet")

    def default_traits_view(self):
        return View(
            Item("num_points"),
            Item("evaluation_mode"),
            Item("space_search_mode"),
        )

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
