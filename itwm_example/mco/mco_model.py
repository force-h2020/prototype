from traits.api import Enum, Instance, on_trait_change, Dict
from traitsui.api import View, Item, InstanceEditor
from force_bdss.api import BaseMCOModel

from .optimizers.optimizers import (
    IOptimizer,
    WeightedOptimizer,
    NevergradOptimizer,
)


class MCOModel(BaseMCOModel):

    evaluation_mode = Enum("Internal", "Subprocess")

    optimizer_mode = Enum("Weighted", "NeverGrad")

    optimizer = Instance(IOptimizer, transient=True)

    optimizer_data = Dict()

    def __init__(self, *args, **kwargs):
        optimizer_mode = kwargs.pop("optimizer_mode", None)
        super().__init__(*args, **kwargs)
        if optimizer_mode is not None:
            self.optimizer_mode = optimizer_mode

    def default_traits_view(self):
        return View(
            Item("evaluation_mode"),
            Item("optimizer_mode"),
            Item("optimizer", style="custom", editor=InstanceEditor()),
        )

    def _optimizer_from_mode(self):
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
        # on switch
        self.optimizer_data.pop("algorithms", None)
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
