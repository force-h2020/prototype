from traits.api import Enum, Bool
from traitsui.api import View, Item

from force_bdss.api import (
    BaseMCOModel,
    PositiveInt,
    WeightedMCOStartEvent,
    WeightedMCOProgressEvent
)

from .weighted_scipy_engine import WeightedScipyEngine


class WeightedMCOModel(BaseMCOModel):

    #: Algorithms available to work with
    algorithms = Enum(
        *WeightedScipyEngine.class_traits()["algorithms"].handler.values
    )

    #: Search grid resolution per KPI
    num_points = PositiveInt(7)

    #: Display the generated points at runtime
    verbose_run = Bool(True)

    #: Space search distribution for weight points sampling
    space_search_mode = Enum("Uniform", "Dirichlet")

    #: 'Subprocess' mode performs evaluation of a state in the workflow via
    #: calling force_bdss on a new subprocess with SubprocessWorkflowEvaluator
    evaluation_mode = Enum("Internal", "Subprocess")

    def default_traits_view(self):
        return View(
            Item("evaluation_mode"),
            Item("algorithms"),
            Item("num_points", label="Weights grid resolution per KPI"),
            Item("space_search_mode"),
            Item("verbose_run"),
        )

    def __start_event_type_default(self):
        return WeightedMCOStartEvent

    def __progress_event_type_default(self):
        return WeightedMCOProgressEvent
