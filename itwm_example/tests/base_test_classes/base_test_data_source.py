import unittest

from force_bdss.api import DataValue

from itwm_example.example_plugin import ExamplePlugin


class BaseTestDataSource(unittest.TestCase):
    """ Base test class for a generic DataSource.
    """
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

    def basic_evaluation(self, test_case_index):
        values = self.test_case_values[test_case_index]
        data_values = self.convert_to_data_values(values, self.input_slots)
        res = self.data_source.run(self.model, data_values)
        return res

    @staticmethod
    def convert_to_data_values(values, slots):
        return [
            DataValue(type=slot.type, value=value)
            for slot, value in zip(slots, values)
        ]

    def base_test_basic_evaluation(self):
        for i in range(len(self.test_case_values)):
            res = self.basic_evaluation(i)
            self.assertAlmostEqual(
                res[0].value,
                self.test_case_objectives[i],
                self._objective_precision
            )


class BaseTestGradientDataSource(BaseTestDataSource):
    """ Base test class for DataSource that implements
    a pair of (objective value, gradient) calculation
    at a runtime.
    """
    def base_test_gradient_type(self):
        self.assertEqual(
            self.output_slots[0].type + "_GRADIENT",
            self.output_slots[1].type
        )

    def base_test_param_to_gradient(self):
        for i in range(len(self.test_case_values)):
            _, gradient = self.basic_evaluation(i)
            self.assertEqual(
                len(self.input_slots),
                len(gradient.value)
            )
