from traits.api import String

from force_bdss.api import (
    factory_id,
    BaseNotificationListenerFactory)

from .csv_writer import CSVWriter
from .csv_writer_model import CSVWriterModel


class CSVWriterFactory(BaseNotificationListenerFactory):
    """This is the factory of the notification listener.
    A notification listener listens to events provided by the MCO,
    and performs operations accordingly.
    """
    id = String(factory_id("itwm", "csv_writer"))

    name = String("CSV Writer")

    model_class = CSVWriterModel

    listener_class = CSVWriter
