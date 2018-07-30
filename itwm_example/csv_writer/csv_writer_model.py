from traits.api import Unicode
from force_bdss.api import BaseNotificationListenerModel


class CSVWriterModel(BaseNotificationListenerModel):
    path = Unicode("output.csv")
