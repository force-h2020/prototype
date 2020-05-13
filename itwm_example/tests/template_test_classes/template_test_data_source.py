#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

import inspect
import unittest

from force_bdss.api import DataValue

from itwm_example.itwm_example_plugin import ITWMExamplePlugin
from itwm_example.unittest_tools.gradient_consistency.taylor_convergence \
    import TaylorTest


def convert_to_data_values(values, slots):
    return [
        DataValue(type=slot.type, value=value)
        for slot, value in zip(slots, values)
    ]


class TemplateTestDataSource(unittest.TestCase):
    """ Base test class for a generic DataSource.
    """

    _data_source_index = None
    test_inputs = []
    test_outputs = []
    _objective_precision = 6

    @property
    def _test_case_traceback(self):
        """Provides a traceback string containing source file of
        subclass, formatted for detection by IDE"""

        test_case_class = type(self).__name__
        source_file = inspect.getsourcefile(type(self))
        line = inspect.getsourcelines(type(self))[1]

        # Formatting of source file path and line
        location = f'\n{source_file}:{line}'

        # Return string with description of context
        return f'\n\n{test_case_class} source file:{location}'

    def setUp(self):
        self.plugin = ITWMExamplePlugin()
        if self._data_source_index is None:
            self.input_slots, self.output_slots = None, None
            return

        self.factory = self.plugin.data_source_factories[
            self._data_source_index
        ]
        self.data_source = self.factory.create_data_source()
        self.model = self.factory.create_model()
        self.slots = self.data_source.slots(self.model)
        self.input_slots, self.output_slots = self.slots

    def _evaluate_function(self, values):
        data_values = convert_to_data_values(values, self.input_slots)
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
        for input, output in zip(self.test_inputs, self.test_outputs):
            res = self.basic_evaluation(input)

            for element, objective in zip(res, output):
                self.assertAlmostEqual(
                    element.value, objective, self._objective_precision,
                    msg=self._test_case_traceback
                )

    def test_output_slots(self):
        """ Base test to verify the number of output slots
        is consistent with the `run` output in terms of the
        number of output values.
        """
        for input in self.test_inputs:
            self.assertEqual(
                len(self.basic_evaluation(input)), len(self.output_slots),
                msg=self._test_case_traceback
            )


class TemplateTestGradientDataSource(TemplateTestDataSource):
    """ Base test class for DataSource that implements
    a pair of (objective value, gradient) calculation
    at a runtime.
    """

    def test_gradient_type(self):
        if self.output_slots is not None:
            self.assertEqual(
                self.output_slots[0].type + "_GRADIENT",
                self.output_slots[1].type,
            )

    def test_basic_evaluation(self):
        """ Overrides the base test evaluation method,
        specifically to verify the objective value consistency
        _only_, within the precision.
        Gradients consistency are NOT tested here.
        """
        for input, output in zip(self.test_inputs, self.test_outputs):
            res = self.basic_evaluation(input)
            self.assertAlmostEqual(
                res[0].value, output, self._objective_precision,
                msg=self._test_case_traceback
            )

    def test_param_to_gradient(self):
        """ Base test to verify the consistency between
        the number of input slots and the length of the
        output gradients, as this should be injective.
        """
        for values in self.test_inputs:
            _, gradient = self.basic_evaluation(values)
            self.assertEqual(
                len(self.input_slots), len(gradient.value),
                msg=self._test_case_traceback)

    def _evaluate_gradient(self, values):
        data_values = convert_to_data_values(values, self.input_slots)
        res = self.data_source.run(self.model, data_values)
        return res[1].value

    def test_gradient_convergence(self):
        """ Base test for gradient consistency. Estimates the
        order of convergence for gradient versus known values.
        See implementation of `TaylorTest` for details.
        """
        for input in self.test_inputs:
            taylor_test = TaylorTest(
                self._evaluate_function,
                self._evaluate_gradient,
                len(self.input_slots),
            )
            self.assertTrue(
                taylor_test.is_correct_gradient(input),
                msg=self._test_case_traceback)
