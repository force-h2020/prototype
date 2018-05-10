from traits.api import String
from force_bdss.api import BaseNotificationListenerModel


class CSVWriterModel(BaseNotificationListenerModel):
    path = String("output.csv")
