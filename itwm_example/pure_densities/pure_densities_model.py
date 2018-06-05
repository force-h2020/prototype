from traits.api import Float
from traitsui.api import View, Item

from force_bdss.api import BaseDataSourceModel


class PureDensitiesModel(BaseDataSourceModel):
    a_pure_density = Float(1.0)
    b_pure_density = Float(1.0)
    c_pure_density = Float(1.0)

    default_traits_view = View(
        Item("a_pure_density"),
        Item("b_pure_density"),
        Item("c_pure_density"),
    )
