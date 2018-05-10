from traits.api import String

from force_bdss.api import factory_id, BaseDataSourceFactory
from itwm_example.material_cost_data_source.material_cost_data_source import \
    MaterialCostDataSource
from itwm_example.material_cost_data_source.material_cost_data_source_model \
    import \
    MaterialCostDataSourceModel


class MaterialCostDataSourceFactory(BaseDataSourceFactory):
    id = String(factory_id("itwm", "material_cost_data_source"))

    name = String("Material cost")

    model_class = MaterialCostDataSourceModel

    data_source_class = MaterialCostDataSource
