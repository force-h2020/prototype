import unittest

from force_bdss.api import DataValue

from itwm_example.example_plugin import ExamplePlugin
from itwm_example.unittest_tools.gradient_consistency.taylor_convergence \
    import TaylorTest


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

    def _evaluate_function(self, values):
        data_values = self.convert_to_data_values(
            values,
            self.input_slots
        )
        res = self.data_source.run(self.model, data_values)
        return res[0].value

    def basic_evaluation(self, values):
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
        for i, values in enumerate(self.test_case_values):
            res = self.basic_evaluation(values)
            self.assertAlmostEqual(
                res[0].value,
                self.test_case_objectives[i],
                self._objective_precision
            )

    def base_test_output_slots(self, values):
        self.assertEqual(
            len(self.basic_evaluation(values)),
            len(self.output_slots)
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
        for values in self.test_case_values:
            _, gradient = self.basic_evaluation(values)
            self.assertEqual(
                len(self.input_slots),
                len(gradient.value)
            )

    def _evaluate_gradient(self, values):
        data_values = self.convert_to_data_values(
            values,
            self.input_slots
        )
        res = self.data_source.run(self.model, data_values)
        return res[1].value

    def base_test_gradient_convergence(self):
        taylor_test = TaylorTest(
            self._evaluate_function,
            self._evaluate_gradient,
            len(self.input_slots)
        )
        self.assertTrue(
            taylor_test.is_correct_gradient(self.test_case_values[0])
        )
