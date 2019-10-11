from traits.api import Float

from force_bdss.api import BaseDataSourceModel


class ProductionCostDataSourceModel(BaseDataSourceModel):
    #: Cost per square kelvin per second.
    W = Float(1.0)

    #: Model specific temperature shift
    temperature_shift = Float(20.0)
