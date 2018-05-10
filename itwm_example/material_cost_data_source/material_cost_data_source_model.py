from traits.api import Float

from force_bdss.api import BaseDataSourceModel


class MaterialCostDataSourceModel(BaseDataSourceModel):
    const_A = Float(1.0)
    const_C = Float(1.0)
    cost_B = Float(1.0)
