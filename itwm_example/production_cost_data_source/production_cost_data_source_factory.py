from traits.api import String

from force_bdss.api import factory_id, BaseDataSourceFactory

from .production_cost_data_source_model import ProductionCostDataSourceModel
from .production_cost_data_source import ProductionCostDataSource


class ProductionCostDataSourceFactory(BaseDataSourceFactory):
    id = String(factory_id("itwm", "production_cost_data_source"))

    name = String("Production cost (heat)")

    model_class = ProductionCostDataSourceModel

    data_source_class = ProductionCostDataSource
