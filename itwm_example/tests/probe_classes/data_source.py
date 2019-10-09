import unittest

from force_bdss.api import DataValue
from itwm_example.example_plugin import ExamplePlugin


class ProbeDataSource(unittest.TestCase):
    _data_source_index = None

    def setUp(self):
        self.plugin = ExamplePlugin()
        self.factory = self.plugin.data_source_factories[self._data_source_index]
        self.data_source = self.factory.create_data_source()
        self.model = self.factory.create_model()
        self.slots = self.data_source.slots(self.model)
        self.input_slots, self.output_slots = self.slots

    @staticmethod
    def convert_to_data_values(values, slots):
        return [
            DataValue(type=slot.type, value=value)
            for slot, value in zip(slots, values)
        ]
