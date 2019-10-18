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

    space_search_strategy = Enum("Uniform", "Dirichlet")

    def default_traits_view(self):
        return View(
            Item('num_points'),
            Item('evaluation_mode'),
            Item('space_search_strategy')
        )

    def weights_samples(self, **kwargs):
        if self.space_search_strategy == "Uniform":
            distribution = UniformSpaceSampler
        elif self.space_search_strategy == "Dirichlet":
            distribution = DirichletSpaceSampler
        else:
            raise NotImplementedError
        return distribution(
            len(self.kpis),
            self.num_points,
            **kwargs
        ).generate_space_sample()
