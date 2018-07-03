from itwm_example.mco.mco_factory import MCOFactory
from itwm_example.pure_densities.pure_densities_factory import \
    PureDensitiesFactory

from force_bdss.api import BaseExtensionPlugin, plugin_id
from itwm_example.csv_writer.csv_writer_factory import CSVWriterFactory
from itwm_example.fixed_value_data_source.fixed_value_data_source_factory \
    import \
    FixedValueDataSourceFactory
from itwm_example.impurity_concentration\
    .impurity_concentration_data_source_factory import \
    ImpurityConcentrationDataSourceFactory
from itwm_example.material_cost_data_source\
    .material_cost_data_source_factory import \
    MaterialCostDataSourceFactory
from itwm_example.production_cost_data_source\
    .production_cost_data_source_factory import \
    ProductionCostDataSourceFactory
from itwm_example.arrhenius_parameters.arrhenius_parameters_factory import \
    ArrheniusParametersFactory


class ExamplePlugin(BaseExtensionPlugin):
    id = plugin_id("itwm", "example", 0)

    def get_factory_classes(self):
        return [
            MCOFactory,
            FixedValueDataSourceFactory,
            ProductionCostDataSourceFactory,
            ArrheniusParametersFactory,
            MaterialCostDataSourceFactory,
            ImpurityConcentrationDataSourceFactory,
            PureDensitiesFactory,
            CSVWriterFactory
        ]
