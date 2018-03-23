import unittest
import numpy as np

from ..reaction_kinetics import Reaction_kinetics, analytical_solution, \
                                dalpha, grad_x, calc_k

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

    def test_run_return_type(self):
        rkin = Reaction_kinetics()
        X_mat, grad_x_X_mat = rkin.run_default(R)
        self.assertEqual(type(X_mat), nptype)
        self.assertEqual(type(grad_x_X_mat), nptype)

    def test_run_return_shape(self):
        rkin = Reaction_kinetics()
        X_mat, grad_x_X_mat = rkin.run_default(R)
        self.assertEqual(X_mat.shape, (5,))
        self.assertEqual(grad_x_X_mat.shape, (5, 7))

    def test_analytic_solutionreturn_type(self):
        params = [0.4, 0.5, 0., 0., 0.1, np.array([1, 2]), 20]
        self.assertEqual(type(analytical_solution(*params)), nptype)

    def test_analytic_solutionreturn_shape(self):
        params = [0.4, 0.5, 0., 0., 0.1, np.array([1, 2]), 20]
        self.assertEqual(analytical_solution(*params).shape, (5,))

    def test_dalpha_return_type(self):
        da = dalpha(0.4, 0.5, 3., 20.)
        self.assertEqual(type(da), nptype)

    def test_dalpha_return_shape(self):
        da = dalpha(0.4, 0.5, 3., 20.)
        self.assertEqual(da.shape, (4,))

    def test_grad_x_return_type(self):
        params = [0.4, 0.5, 0., 0., 0.1, np.array([1, 2]), 20]
        gradx = grad_x(*params)
        self.assertEqual(type(gradx), nptype)

    def test_grad_x_return_shape(self):
        params = [0.4, 0.5, 0., 0., 0.1, np.array([1, 2]), 20]
        gradx = grad_x(*params)
        self.assertEqual(gradx.shape, (5, 7))

    def test_calc_k_return_type(self):
        kps = calc_k(330, M)
        self.assertEqual(type(kps), nptype)

    def test_calc_k_return_shape(self):
        kps = calc_k(330, M)
        self.assertEqual(kps.shape, (2,))

    def test_calc_k_return(self):
        kps = calc_k(330, M)
        self.assertTrue(kps[0] > kps[1])
