from force_bdss.api import (
    BaseMCOFactory,
    FixedMCOParameterFactory,
    RangedMCOParameterFactory,
)

from itwm_example.mco.mco_communicator import MCOCommunicator
from .weighted_mco_model import WeightedMCOModel
from .weighted_mco import WeightedMCO


class WeightedMCOFactory(BaseMCOFactory):
    def get_identifier(self):
        return "weighted_mco"

    def get_name(self):
        return "Weighted Multi Criteria optimizer"

    #: Returns the model class
    def get_model_class(self):
        return WeightedMCOModel

    #: Returns the optimizer class
    def get_optimizer_class(self):
        return WeightedMCO

    #: Returns the communicator class
    def get_communicator_class(self):
        return MCOCommunicator

    #: Factory classes of the parameters the MCO supports.
    def get_parameter_factory_classes(self):
        return [FixedMCOParameterFactory, RangedMCOParameterFactory]
