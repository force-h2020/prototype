import unittest
import numpy as np

from ..reaction_kinetics import Reaction_kinetics, _analytical_solution, \
                                _grad_x, _calc_k

A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
P = { "name": "product", "manufacturer": "", "pdi": 0 }
R = { "reactants": [A, B], "products": [P] }
M = (np.array([1, 1], float), np.array([1.5, 6.], float))
nptype = type(np.array([]))


class Reaction_kineticsTestCase(unittest.TestCase):

    def test_instance(self):
        rkin = Reaction_kinetics()
        self.assertIsInstance(rkin, Reaction_kinetics)

    def test_analytic_solutionreturn_type(self):
        params = [0.4, 0.5, 0., 0., 0.1, np.array([1, 2]), 20]
        self.assertEqual(type(_analytical_solution(*params)), nptype)

    def test_analytic_solutionreturn_shape(self):
        params = [0.4, 0.5, 0., 0., 0.1, np.array([1, 2]), 20]
        self.assertEqual(_analytical_solution(*params).shape, (5,))

    def test_grad_x_return_type(self):
        params = [0.4, 0.5, 0., 0., 0.1, np.array([1, 2]), 20]
        gradx = _grad_x(*params)
        self.assertEqual(type(gradx), nptype)

    def test_grad_x_return_shape(self):
        params = [0.4, 0.5, 0., 0., 0.1, np.array([1, 2]), 20]
        gradx = _grad_x(*params)
        self.assertEqual(gradx.shape, (5, 7))

    def test_calc_k_return_type(self):
        kps = _calc_k(330, M)
        self.assertEqual(type(kps), nptype)

    def test_calc_k_return_shape(self):
        kps = _calc_k(330, M)
        self.assertEqual(kps.shape, (2,))

    def test_calc_k_return(self):
        kps = _calc_k(330, M)
        self.assertTrue(kps[0] > kps[1])
