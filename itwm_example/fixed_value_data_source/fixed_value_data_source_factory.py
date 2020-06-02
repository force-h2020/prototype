#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from force_bdss.api import BaseDataSourceFactory

from .fixed_value_data_source_model import FixedValueDataSourceModel
from .fixed_value_data_source import FixedValueDataSource


class FixedValueDataSourceFactory(BaseDataSourceFactory):
    def get_identifier(self):
        return "fixed_value_data_source"

    def get_name(self):
        return "Fixed Value"

    def get_model_class(self):
        return FixedValueDataSourceModel

    def get_data_source_class(self):
        return FixedValueDataSource
