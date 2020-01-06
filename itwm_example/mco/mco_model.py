from traits.api import Enum, Instance, on_trait_change, Dict
from traitsui.api import View, Item, InstanceEditor
from force_bdss.api import BaseMCOModel

from .optimizers.optimizers import (
    IOptimizer,
    WeightedOptimizer,
    NevergradOptimizer,
)
from .driver_enents import ITWMMCOStartEvent


class MCOModel(BaseMCOModel):

    evaluation_mode = Enum("Internal", "Subprocess")

    optimizer_mode = Enum("Weighted", "NeverGrad")

    optimizer = Instance(IOptimizer, transient=True)

    optimizer_data = Dict()

    def __init__(self, *args, **kwargs):
        # We pop out the optimizer data to avoid inconsistent data
        # assignment to the default optimizer instance, which is created
        # during super().__init__
        optimizer_data = kwargs.pop("optimizer_data", None)

        super().__init__(*args, **kwargs)
        # Optimizer is created with proper "optimizer_mode" (not default).
        # It should only be created _after_ the parameters and KPIs
        # have been specified. We instantiate the optimizer after all
        # other attributes have been assigned.
        if optimizer_data is not None:
            self.optimizer_data = optimizer_data
        self.optimizer = self._optimizer_default()

    def default_traits_view(self):
        return View(
            Item("evaluation_mode"),
            Item("optimizer_mode"),
            Item("optimizer", style="custom", editor=InstanceEditor()),
        )

    def _optimizer_from_mode(self):
        klass = None
        if self.optimizer_mode == "Weighted":
            klass = WeightedOptimizer
        elif self.optimizer_mode == "NeverGrad":
            klass = NevergradOptimizer
        return klass

    def _optimizer_default(self):
        klass = self._optimizer_from_mode()
        return klass(
            kpis=self.kpis, parameters=self.parameters, **self.optimizer_data
        )

    @on_trait_change("optimizer_mode")
    def update_optimizer(self):
        klass = self._optimizer_from_mode()
        # In order to prevent collisions in "algorithms" Enum object, we pop
        # it out and therefore always have the default value of "algorithms"
        # and "name" on switch
        self.optimizer_data.pop("algorithms", None)
        self.optimizer_data.pop("name", None)
        self.optimizer = klass(
            kpis=self.kpis, parameters=self.parameters, **self.optimizer_data
        )

    @on_trait_change("kpis")
    def update_kpis(self):
        self.optimizer.kpis = self.kpis

    @on_trait_change("parameters")
    def update_parameters(self):
        self.optimizer.parameters = self.parameters

    def __getstate__(self):
        state = super().__getstate__()
        state["optimizer_data"] = self.optimizer.__getstate__()
        return state

    def create_start_event(self):
        event = ITWMMCOStartEvent(
            parameter_names=list(p.name for p in self.parameters),
            kpi_names=list(kpi.name for kpi in self.kpis),
        )
        return event
