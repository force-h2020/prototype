from itwm_example.pure_densities.pure_densities import PureDensities
from itwm_example.pure_densities.pure_densities_model import PureDensitiesModel

from force_bdss.data_sources.base_data_source_factory import \
    BaseDataSourceFactory


class PureDensitiesFactory(BaseDataSourceFactory):
    def get_identifier(self):
        return "pure_densities"

    def get_name(self):
        return "Pure densities"

    def get_data_source_class(self):
        return PureDensities

    def get_model_class(self):
        return PureDensitiesModel
