from force_bdss.api import BaseCSVWriter, BaseCSVWriterFactory


class CSVWriter(BaseCSVWriter):
    def parse_progress_event(self, event):
        event_datavalues = super().parse_progress_event(event)
        event_datavalues.extend(event.weights)
        return event_datavalues

    def parse_start_event(self, event):
        header = super().parse_start_event(event)
        header.extend([f"{name} weight" for name in event.kpi_names])
        return header


class CSVWriterFactory(BaseCSVWriterFactory):
    def get_identifier(self):
        return "csv_writer"

    def get_name(self):
        return "CSV Writer"

    def get_listener_class(self):
        return CSVWriter
