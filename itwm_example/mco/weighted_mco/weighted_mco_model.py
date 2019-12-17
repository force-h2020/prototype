from traits.api import Enum, Bool
from traitsui.api import View, Item

from force_bdss.api import BaseMCOModel, PositiveInt, WeightedOptimizerEngine


class WeightedMCOModel(BaseMCOModel):

    #: Algorithms available to work with
    algorithms = Enum(
        *WeightedOptimizerEngine.class_traits()["algorithms"].handler.values
    )

    #: Search grid resolution per KPI
    num_points = PositiveInt(7)

    #: Display the generated points at runtime
    verbose_run = Bool(True)

    #: Space search distribution for weight points sampling
    space_search_mode = Enum("Uniform", "Dirichlet")

    #:
    evaluation_mode = Enum("Direct", "Subprocess")

    def default_traits_view(self):
        return View(
            Item("algorithms"),
            Item("num_points", label="Weights grid resolution per KPI"),
            Item("space_search_mode"),
            Item("verbose_run"),
        )
