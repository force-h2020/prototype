#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from scipy.stats import linregress
import numpy as np


class TaylorTest:
    """Gradient consistency test.
    Estimates and verifies the order of the Taylor remainder
    convergence for a provided (function, function_gradient)
    object.

    Notes
    -----

    Consider a `function` and a `gradient_f` functions.
    The aim is to verify the consistency of the estimated
    (usually analytically derived) `gradient_f` with the
    true derivative of `function`.
    Note, that the Taylor expansion of the function near a
    point `x` in the direction of `v` is:

    .. math::
        f(x + v) = f(x) + f'(x) * v + O(|v^2|)

    or, equivalently,

    .. math::
        f(x + v) - f(x) - f'(x) * v = O(|v^2|)

    Then, given that the approximation `gradient_f` is correct,
    the following should hold:

    .. math::
        f(x + v) - f(x) - `gradient_f`(x) * v = O(|v^2|)

    We estimate the right hand side term, and if it is growing
    slower than `O(|v^2|)`, the approximation is wrong.
    This automatically implies that the `gradient_x`
    implementation is wrong.
    """

    def __init__(
            self,
            function,
            gradient,
            input_dimension,
            slope_tolerance=1.e-2
    ):
        self._function = function
        self._gradient = gradient
        self._input_dimension = input_dimension
        self.slope_tolerance = slope_tolerance

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

    @staticmethod
    def _fit_power_law(x, y):
        """Fit power function y = x**slope to data.

        Finds the best fit of `slope` for the pair of np.arrays (x,y)
        that minimises the standard l2-deviation from the power function.
        Both the (x, y) must contain positive, nonzero values.

        Parameters
        ----------
        x: np.array
            One dimensional array of abscissae
        x: np.array
            One dimensional array of ordinates

        Returns
        -------
        slope: float
            Linear regression fit coefficient
        """
        regression = linregress(np.log(x), np.log(y))
        return regression.slope

    def _test_directions(self):
        """ Generates simple complete set of the test directions.

        The individual directions are implemented in
            `_single_component_vector`.

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
        """Generate a tuple of uniform perturbation values,
        proportional to the `step_size`.

        For example,
            >>> perturbations = self._generate_uniform_perturbations(
            >>>     1., 3
            >>> )
            >>> perturbations
            (1.0, 2.0, 3.0)

        Parameters
        ----------
        step_size: float
            the amplitude of the perturbation
        length: int
            the number of the perturbation values

        Returns
        -------
        perturbations: tuple(float)
            Tuple of generated perturbation values
        """
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

        Parameters
        ----------
        init_point: np.array
            Initial point, around which the gradient consistency
            is explored
        direction: np.array
            Perturbation direction, e.g., element of _test_directions
        step_size: float
            Scales the amplitude of the perturbation direction

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

        # Preliminary setup:
        # Calculate the unperturbed solution and gradient
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
            # In order to avoid encountering np.log(0) case, the values of
            # `y` is shifted by an acceptable (machine precision) amount.
            taylor_remainders[i] = abs(
                    perturbed_value
                    - default_value
                    - shift * perturbation_norm
            ) + np.finfo(np.float64).eps
            shifts[i] = shift

        return shifts, taylor_remainders

    def run_taylor_test(self, initial_point, return_data=False):
        """An entry point to the Taylor testing.
        Generator, that estimates and yields the slope of the
        Taylor reminders power fit for a complete set of possible
        directions.

        return_data:bool If the Taylor remainders are requested,
            the generator yields them as well

        Yields
        -------
        slope: float
            Slopes of power fit with respect to the perturbation
            in the unit direction.
        (optional, return_data=True)
        data: [List(float), List(float)]
            Shift amplitudes and Taylor reminders for the slope
        """
        initial_point = np.asarray(initial_point, dtype=np.float64)

        for i, direction in enumerate(self._test_directions()):
            x, y = self._calculate_taylor_remainders(
                initial_point,
                direction
            )
            slope = self._fit_power_law(x, y)
            if return_data:
                yield slope, (x, y)
            else:
                yield slope

    def is_correct_gradient(self, inintial_point):
        """ Performs a simple Taylor test to verify the
        convergence rates are acceptable within the tolerance
        provided.

        Returns
        -------
        bool:
            If all of the tests are successful
        """
        _remainder_precision = 1.e-14
        slopes = self.run_taylor_test(inintial_point, return_data=True)
        for slope, data in slopes:
            if all(data[1] < _remainder_precision):
                continue
            elif slope < 2.0 - self.slope_tolerance:
                return False

        return True
