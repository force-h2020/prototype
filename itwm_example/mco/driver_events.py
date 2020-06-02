#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from force_bdss.api import MCOStartEvent


class ITWMMCOStartEvent(MCOStartEvent):
    """ UnileverMCOStartEvent class overloads the `serialize` method
    and introduces the KPI pass marks to the StartEvent representation.
    """

    def serialize(self):
        header = super().serialize()
        header += [f"{name} weight" for name in self.kpi_names]
        return header
