from traits.api import Float

from force_bdss.api import BaseDataSourceModel


class ProductionCostDataSourceModel(BaseDataSourceModel):
    #: Cost per square kelvin per second.
    W = Float(1.0)
