from traits.api import String

from force_bdss.api import factory_id, BaseDataSourceFactory

from .fixed_value_data_source_model import FixedValueDataSourceModel
from .fixed_value_data_source import FixedValueDataSource


class FixedValueDataSourceFactory(BaseDataSourceFactory):
    id = String(factory_id("itwm", "fixed_value_data_source"))

    name = String("Fixed Value")

    model_class = FixedValueDataSourceModel

    data_source_class = FixedValueDataSource
