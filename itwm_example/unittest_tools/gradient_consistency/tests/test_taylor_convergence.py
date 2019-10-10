import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from itwm_example.unittest_tools.gradient_consistency.taylor_convergence \
    import TaylorTest


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
            np.array([1, 0]),
            np.array([0, 1])
        ]
        for given, test in zip(
                self.default_taylor_tool._test_directions(),
                test_directions
        ):
            assert_almost_equal(given, test)

    def test_taylor_remainders(self):
        known_remainders = np.array(
            [
                0.01,
                0.04,
                0.09,
                0.16,
                0.25
            ]
        )

        direction = self.default_taylor_tool._test_directions()[0]
        _, remainders = self.default_taylor_tool._calculate_taylor_remainders(
            [1., 2.],
            direction,
            step_size=0.1
        )
        assert_almost_equal(known_remainders, remainders)

    def test_taylor_run(self):
        self.assertTrue(self.default_taylor_tool.is_correct_gradient([1., 2.]))

        slopes = self.default_taylor_tool.run_taylor_test([1., 2.])
        for slope in slopes:
            self.assertAlmostEqual(2, slope, 1)

        def wrong_function_gradient(x, eps=1.e-4):
            return [2. * x[0] + eps, 3. * x[1] ** 2. + eps]

        wrong_taylor_tool = TaylorTest(
            self.default_function,
            wrong_function_gradient,
            2
        )

        failed_slopes = wrong_taylor_tool.run_taylor_test([1., 2.])
        for slope in failed_slopes:
            self.assertGreater(2, slope)

        taylor_test_sincos = TaylorTest(
            np.sin,
            np.cos,
            1
        )
        self.assertTrue(taylor_test_sincos.is_correct_gradient([1.]))
