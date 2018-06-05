from traits.api import Float

from force_bdss.api import BaseDataSourceModel


class PureDensitiesModel(BaseDataSourceModel):
    a_pure_density = Float(1.0)
    b_pure_density = Float(1.0)
    c_pure_density = Float(1.0)
