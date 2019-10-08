import unittest

from force_bdss.api import DataValue
from itwm_example.example_plugin import ExamplePlugin


class TestImpurityConcentrationDataSource(unittest.TestCase):
    def setUp(self):
        self.plugin = ExamplePlugin()
        self.factory = self.plugin.data_source_factories[4]

    def test_data_source_slots(self):
        data_source = self.factory.create_data_source()
        model = self.factory.create_model()

        input_slots, output_slots = data_source.slots(model)
        self.assertEqual(12, len(input_slots))
        self.assertEqual(2, len(output_slots))

    def test_basic_evaluation(self):
        data_source = self.factory.create_data_source()
        model = self.factory.create_model()
        values = [
            0.5,
            0.1,
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
        input_slots, output_slots = data_source.slots(model)

        self.assertEqual(len(values), len(input_slots))

        data_values = [
            DataValue(type=slot.type, value=value)
            for slot, value in zip(input_slots, values)
        ]

        res = data_source.run(model, data_values)

        self.assertEqual(len(output_slots), len(res))
        self.assertAlmostEqual(res[0].value, 0.372027129767, 6)
