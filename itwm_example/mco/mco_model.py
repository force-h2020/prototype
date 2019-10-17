from traits.api import Enum
from traitsui.api import View, Item
from force_bdss.api import BaseMCOModel, PositiveInt


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
