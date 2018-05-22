from force_bdss.api import (
    BaseNotificationListenerFactory)

from .csv_writer import CSVWriter
from .csv_writer_model import CSVWriterModel


class CSVWriterFactory(BaseNotificationListenerFactory):
    def get_identifier(self):
        return "csv_writer"

    def get_name(self):
        return "CSV Writer"

    def get_model_class(self):
        return CSVWriterModel

    def get_listener_class(self):
        return CSVWriter
