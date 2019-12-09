from traits.api import Float

from force_bdss.mco.parameters.mco_parameters import (
    RangedMCOParameterFactory,
    RangedMCOParameter,
)


class ITWMRangedMCOParameter(RangedMCOParameter):
    """ Ranged MCO parameter with an initial value."""

    initial_value = Float(0.0)


class ITWMRangedMCOParameterFactory(RangedMCOParameterFactory):
    """The factory of the above model"""

    #: Definition of the associated model class.
    def get_model_class(self):
        return ITWMRangedMCOParameter
