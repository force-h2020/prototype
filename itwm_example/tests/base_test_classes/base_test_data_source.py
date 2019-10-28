import unittest

from force_bdss.api import DataValue

from itwm_example.example_plugin import ExamplePlugin
from itwm_example.unittest_tools.gradient_consistency.taylor_convergence \
    import TaylorTest


def convert_to_data_values(values, slots):
    return [
        DataValue(type=slot.type, value=value)
        for slot, value in zip(slots, values)
    ]


class BaseTestDataSource(unittest.TestCase):
    """ Base test class for a generic DataSource.
    """
    _data_source_index = None
    test_case_values = []
    test_case_objectives = []
    _objective_precision = 6

    def setUp(self):
        self.plugin = ExamplePlugin()
        if self._data_source_index is None:
            return

        self.factory = self.plugin.data_source_factories[
            self._data_source_index
        ]
        self.data_source = self.factory.create_data_source()
        self.model = self.factory.create_model()
        self.slots = self.data_source.slots(self.model)
        self.input_slots, self.output_slots = self.slots

    def _evaluate_function(self, values):
        data_values = convert_to_data_values(
            values,
            self.input_slots
        )
        res = self.data_source.run(self.model, data_values)
        return res[0].value

    def basic_evaluation(self, values):
        """ With `values` provided, we convert the numeric
        values into the DataSource acceptable format, and
        perform `DataSource().run` evaluation.

        Parameters
        ----------
        values: List[Int, Float]
            Generic type values to `run` with

        Returns
        ----------
        res: List[DataValues]
            `run` outputs
        """
        data_values = convert_to_data_values(values, self.input_slots)
        res = self.data_source.run(self.model, data_values)
        return res

    def test_basic_evaluation(self):
        """ Base test method for basic evaluation
        (see `basic_evaluation`)
        For the setup list of test case values, we verity
        that the `run` output is consistent with the known
        provided test case objectives.
        """
        for i, values in enumerate(self.test_case_values):
            res = self.basic_evaluation(values)

            for element, objective in zip(
                res,
                self.test_case_objectives[i]
            ):
                self.assertAlmostEqual(
                    element.value,
                    objective,
                    self._objective_precision
                )

    def base_test_output_slots(self, values):
        """ Base test to verify the number of output slots
        is consistent with the `run` output in terms of the
        number of output values.
        """
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

    def test_basic_evaluation(self):
        """ Overrides the base test evaluation method,
        specifically to verify the objective value consistency
        _only_, within the precision.
        Gradients consistency are NOT tested here.
        """
        for i, values in enumerate(self.test_case_values):
            res = self.basic_evaluation(values)
            self.assertAlmostEqual(
                res[0].value,
                self.test_case_objectives[i],
                self._objective_precision
            )

    def base_test_param_to_gradient(self):
        """ Base test to verify the consistency between
        the number of input slots and the length of the
        output gradients, as this should be injective.
        """
        for values in self.test_case_values:
            _, gradient = self.basic_evaluation(values)
            self.assertEqual(
                len(self.input_slots),
                len(gradient.value)
            )

    def _evaluate_gradient(self, values):
        data_values = convert_to_data_values(
            values,
            self.input_slots
        )
        res = self.data_source.run(self.model, data_values)
        return res[1].value

    def base_test_gradient_convergence(self):
        """ Base test for gradient consistency. Estimates the
        order of convergence for gradient versus known values.
        See implementation of `TaylorTest` for details.
        """
        taylor_test = TaylorTest(
            self._evaluate_function,
            self._evaluate_gradient,
            len(self.input_slots)
        )
        self.assertTrue(
            taylor_test.is_correct_gradient(self.test_case_values[0])
        )
