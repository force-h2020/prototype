#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from traits.api import Float
from traitsui.api import View, Item

from force_bdss.api import BaseDataSourceModel


class ArrheniusParametersModel(BaseDataSourceModel):
    nu_main_reaction = Float(0.02)
    delta_H_main_reaction = Float(1.5)

    nu_secondary_reaction = Float(0.02)
    delta_H_secondary_reaction = Float(12.0)

    traits_view = View(
        Item("nu_main_reaction"),
        Item("delta_H_main_reaction"),
        Item("nu_secondary_reaction"),
        Item("delta_H_secondary_reaction"),
    )
