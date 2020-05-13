#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from force_bdss.api import BaseCSVWriterFactory


class CSVWriterFactory(BaseCSVWriterFactory):
    """This class does not contribute any custom
    BaseCSVWriter or BaseCSVWriterModel subclasses, but is
    implemented for backwards compatibility of the
    identifier"""

    def get_identifier(self):
        return "csv_writer"

    def get_name(self):
        return "CSV Writer"
