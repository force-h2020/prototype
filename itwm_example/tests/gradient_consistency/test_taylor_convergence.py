import unittest

from itwm_example.tests.gradient_consistency.taylor_convergence import (
    TaylorTest
)


class TestTaylorTest(unittest.TestCase):
    def setUp(self):
        def default_function(x):
            return x[0] ** 2. + x[1] ** 3.

        def default_function_gradient(x):
            return [2. * x[0], 3. * x[1] ** 2.]

        self.default_function = default_function
        self.default_function_gradient = default_function_gradient

        self.default_taylor_tool = TaylorTest(
            self.default_function,
            self.default_function_gradient,
            2
        )

    def test_initialization(self):
        self.assertIs(
            self.default_taylor_tool._function,
            self.default_function
        )
        self.assertIs(
            self.default_taylor_tool._gradient,
            self.default_function_gradient
        )
        self.assertEqual(self.default_taylor_tool._input_dimension, 2)

    def test_evaluate(self):
        test_point = [1, 2]
        self.assertEqual(
            self.default_taylor_tool._evaluate_function(test_point),
            self.default_function(test_point)
        )
        self.assertEqual(
            self.default_taylor_tool._evaluate_gradient(test_point),
            self.default_function_gradient(test_point)
        )

    def test_generate_directions(self):
        test_directions = [
            [1, 0],
            [0, 1]
        ]
        for given, test in zip(
                self.default_taylor_tool._test_directions(),
                test_directions
        ):
            self.assertListEqual(given.tolist(), test)

    def test_taylor_remainders(self):
        for direction in self.default_taylor_tool._test_directions():
            slope = self.default_taylor_tool._fit_power_law(
                *self.default_taylor_tool._calculate_taylor_remainders(
                    [1., 2.],
                    direction
                )
            )
            self.assertAlmostEqual(slope, 2, 1)

    def test_taylor_run(self):
        slopes = self.default_taylor_tool.run_taylor_test([1., 2.])
        for slope in slopes:
            self.assertAlmostEqual(2, slope, 1)
