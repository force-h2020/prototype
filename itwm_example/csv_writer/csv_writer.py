from traits.api import Instance

import csv

from force_bdss.api import (
    BaseNotificationListener,
    MCOProgressEvent
)
from itwm_example.csv_writer.csv_writer_model import CSVWriterModel


class CSVWriter(BaseNotificationListener):
    model = Instance(CSVWriterModel)

    def deliver(self, event):
        if isinstance(event, MCOProgressEvent):
            with open(self.model.path, 'a') as f:
                writer = csv.writer(f)
                row = ["%.10f" % dv.value for dv in event.optimal_point]
                for dv, weight in zip(event.optimal_kpis, event.weights):
                    row.extend([dv.value, weight])
                writer.writerow(row)

    def initialize(self, model):
        self.model = model

        # truncate the file
        with open(self.model.path, 'wb'):
            pass
