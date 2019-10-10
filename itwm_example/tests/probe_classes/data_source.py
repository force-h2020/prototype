import unittest

from force_bdss.api import DataValue

from itwm_example.example_plugin import ExamplePlugin


class BaseTestDataSource(unittest.TestCase):
    _data_source_index = None
    test_case_values = None
    test_case_objectives = None
    _objective_precision = 6

    def setUp(self):
        self.plugin = ExamplePlugin()
        self.factory = self.plugin.data_source_factories[
            self._data_source_index
        ]
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

    def base_test_gradient_type(self):
        self.assertEqual(
            self.output_slots[0].type + "_GRADIENT",
            self.output_slots[1].type
        )

    def base_test_basic_evaluation(self):
        for i, values in enumerate(self.test_case_values):
            data_values = self.convert_to_data_values(values, self.input_slots)

            res = self.data_source.run(self.model, data_values)

            self.assertAlmostEqual(
                res[0].value,
                self.test_case_objectives[i],
                self._objective_precision
            )
