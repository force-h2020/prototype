from force_bdss.api import BaseDataSourceFactory

from .arrhenius_parameters_model import ArrheniusParametersModel
from .arrhenius_parameters import ArrheniusParameters


class ArrheniusParametersFactory(BaseDataSourceFactory):
    def get_identifier(self):
        return "arrhenius_parameters"

    def get_name(self):
        return "Arrhenius Parameters"

    def get_model_class(self):
        return ArrheniusParametersModel

    def get_data_source_class(self):
        return ArrheniusParameters
