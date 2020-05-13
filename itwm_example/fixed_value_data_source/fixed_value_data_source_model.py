#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from traits.api import Float, Unicode, on_trait_change

from force_bdss.api import BaseDataSourceModel


class FixedValueDataSourceModel(BaseDataSourceModel):
    value = Float()
    cuba_type_out = Unicode()

    @on_trait_change("cuba_type_out")
    def _notify_changes_slots(self):
        self.changes_slots = True
