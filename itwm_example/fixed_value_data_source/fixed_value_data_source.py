#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from force_bdss.api import BaseDataSource, DataValue, Slot


class FixedValueDataSource(BaseDataSource):
    """Defines a data source that returns a fixed value
    specified in the model.
    """

    def run(self, model, parameters):
        return [
            DataValue(
                type=model.cuba_type_out,
                value=model.value
            )]

    def slots(self, model):
        return (
            (
            ),
            (
                Slot(type=model.cuba_type_out),
            )
        )
