from force_bdss.api import BaseMCOFactory

from .mco_communicator import MCOCommunicator
from .mco_model import MCOModel
from .mco import MCO
from .parameters import ITWMRangedMCOParameterFactory


class MCOFactory(BaseMCOFactory):
    def get_identifier(self):
        return "itwm_mco"

    def get_name(self):
        return "ITWM optimizer"

    #: Returns the model class
    def get_model_class(self):
        return MCOModel

    #: Returns the optimizer class
    def get_optimizer_class(self):
        return MCO

    #: Returns the communicator class
    def get_communicator_class(self):
        return MCOCommunicator

    #: Factory classes of the parameters the MCO supports.
    def get_parameter_factory_classes(self):
        return [
            ITWMRangedMCOParameterFactory
        ]
