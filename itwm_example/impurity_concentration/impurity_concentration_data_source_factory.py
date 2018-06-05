from force_bdss.data_sources.base_data_source_factory import \
    BaseDataSourceFactory
from itwm_example.impurity_concentration.impurity_concentration_data_source \
    import \
    ImpurityConcentrationDataSource
from itwm_example.impurity_concentration\
    .impurity_concentration_data_source_model import \
    ImpurityConcentrationDataSourceModel


class ImpurityConcentrationDataSourceFactory(BaseDataSourceFactory):
    def get_identifier(self):
        return "impurity_concentration"

    def get_name(self):
        return "Impurity Concentration"

    def get_data_source_class(self):
        return ImpurityConcentrationDataSource

    def get_model_class(self):
        return ImpurityConcentrationDataSourceModel
