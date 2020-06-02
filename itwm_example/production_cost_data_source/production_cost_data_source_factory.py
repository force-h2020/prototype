#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from force_bdss.api import BaseDataSourceFactory

from .production_cost_data_source_model import ProductionCostDataSourceModel
from .production_cost_data_source import ProductionCostDataSource


class ProductionCostDataSourceFactory(BaseDataSourceFactory):
    def get_identifier(self):
        return "production_cost_data_source"

    def get_name(self):
        return "Production cost (heat)"

    def get_model_class(self):
        return ProductionCostDataSourceModel

    def get_data_source_class(self):
        return ProductionCostDataSource
