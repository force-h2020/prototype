from scipy.stats import linregress
import numpy as np


class TaylorTest:
    """ Gradient consistency test.
    Estimates and verifies the order of the Taylor remainder
    convergence for a provided (function, function_gradient)
    object.
    """

    def __init__(self, function, gradient, input_dimension):
        self._function = function
        self._gradient = gradient
        self._input_dimension = input_dimension

        self._default_step_size = 1.e-6
        self._default_nof_evaluations = 5

    def _evaluate_function(self, point):
        return self._function(point)

    def _evaluate_gradient(self, point):
        return self._gradient(point)

    def _single_component_vector(self, index):
        vector = np.zeros(self._input_dimension)
        vector[index] = 1.
        return vector

    def _fit_power_law(self, x, y):
        regression = linregress(np.log(x), np.log(y))
        return regression.slope

    def _test_directions(self):
        """ Generates simple complete set of the test directions.

        Returns
        -------
        directions: list of np.array
            List of all single component vectors as test directions.
        """
        directions = [
            self._single_component_vector(i)
            for i in range(self._input_dimension)
        ]
        return directions

    def _generate_uniform_perturbations(self, step_size, length=None):
        if length is None:
            length = self._default_nof_evaluations
        perturbations = (
            (i + 1) * step_size for i in range(length)
        )
        return perturbations

    def _calculate_taylor_remainders(
            self,
            init_point,
            direction,
            step_size=None
    ):
        """ Runs a series of function evaluation at the given point,
        perturbing the evaluation point in the provided direction.

        Returns
        -------
        shifts: np.array
            Array of perturbation magnitude (shifts) used to produce
            perturbed function values
        taylor_remainders: np.array
            Array of calculated Taylor remainders
        """
        if step_size is None:
            step_size = self._default_step_size

        default_value = self._evaluate_function(init_point)
        perturbation_norm = np.dot(
            direction,
            self._evaluate_gradient(init_point)
        )

        taylor_remainders = np.zeros(self._default_nof_evaluations)
        shifts = np.zeros(self._default_nof_evaluations)
        for i, shift in enumerate(
                self._generate_uniform_perturbations(step_size)
        ):
            perturbed_point = init_point.copy()
            perturbed_point += shift * direction

            perturbed_value = self._evaluate_function(perturbed_point)
            taylor_remainders[i] = abs(
                    perturbed_value
                    - default_value
                    - shift * perturbation_norm
            )
            shifts[i] = shift

        return shifts, taylor_remainders

    def run_taylor_test(self, initial_point):
        """Estimates the slope of the Taylor reminders power fit
        for a complete set of possible directions.

        Returns
        -------
        slopes: np.array
            Array of slopes with respect to the perturbations
            in the unit directions.
        """
        initial_point = np.array(initial_point)

        slopes = np.zeros(self._input_dimension)
        for i, direction in enumerate(self._test_directions()):
            x, y = self._calculate_taylor_remainders(
                initial_point,
                direction
            )
            slopes[i] = self._fit_power_law(x, y)

        return slopes
