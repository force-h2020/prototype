from __future__ import print_function
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
            with open(self.model.path, 'ab') as f:
                writer = csv.writer(f)
                writer.writerow(event.input + event.output)

    def initialize(self, model):
        self.model = model
