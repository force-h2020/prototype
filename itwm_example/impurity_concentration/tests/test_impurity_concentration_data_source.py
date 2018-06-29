import unittest

from force_bdss.api import DataValue
from itwm_example.example_plugin import ExamplePlugin


class TestImpurityConcentrationDataSource(unittest.TestCase):
    def setUp(self):
        self.plugin = ExamplePlugin()
        self.factory = self.plugin.data_source_factories[4]

    def test_basic_evaluation(self):
        data_source = self.factory.create_data_source()
        model = self.factory.create_model()
        values = [
            0.5,
            0.101,
            335.0,
            360.0,
            0.02,
            1.5,
            0.02,
            12.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        in_slots = data_source.slots(model)[0]
        data_values = [
            DataValue(type=slot.type, value=value)
            for slot, value in zip(in_slots, values)
        ]

        res = data_source.run(model, data_values)

        self.assertAlmostEqual(res[0].value, 0.372519388493335, 6)
