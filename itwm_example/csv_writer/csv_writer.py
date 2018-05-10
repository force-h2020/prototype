from __future__ import print_function
import csv

from force_bdss.api import (
    BaseNotificationListener,
    MCOProgressEvent
)


class CSVWriter(BaseNotificationListener):
    def deliver(self, event):
        if isinstance(event, MCOProgressEvent):
            with open(self.model.path, 'ab') as f:
                writer = csv.writer(f)
                writer.writerow(event.input + event.output)

    def initialize(self, model):
        self.model = model
