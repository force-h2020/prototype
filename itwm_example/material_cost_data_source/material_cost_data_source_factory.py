#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from force_bdss.api import BaseDataSourceFactory
from itwm_example.material_cost_data_source.material_cost_data_source import \
    MaterialCostDataSource
from itwm_example.material_cost_data_source.material_cost_data_source_model \
    import \
    MaterialCostDataSourceModel


class MaterialCostDataSourceFactory(BaseDataSourceFactory):
    def get_identifier(self):
        return "material_cost_data_source"

    def get_name(self):
        return "Material cost"

    def get_model_class(self):
        return MaterialCostDataSourceModel

    def get_data_source_class(self):
        return MaterialCostDataSource
