import unittest
import numpy as np

from ..MCOsolver import MCOsolver

y0 = np.array([0.5, 0.1, 330, 3600])
va_range = (0, 1)
ce_range = (0.001, 0.01)
T_range = (300, 600)
tau_range = (0, 3600)
constr = (va_range, ce_range, T_range, tau_range)
f = lambda y: (y[0] * y[1] * y[2] * y[3])**2
obj_f = lambda y: np.array([f(y), f(y), f(y)])
jac1 = lambda y: 2 * y[0] * (y[1] * y[2] * y[3])**2
jac2 = lambda y: 2 * y[1] * (y[0] * y[2] * y[3])**2
jac3 = lambda y: 2 * y[2] * (y[0] * y[1] * y[3])**2
jac4 = lambda y: 2 * y[3] * (y[0] * y[1] * y[2])**2
jac = lambda y: np.array([jac1(y), jac2(y), jac3(y), jac4(y)])
obj_jac = lambda y: np.array([jac(y), jac(y), jac(y)])
nptype = type(np.array([]))
calltype = type(lambda x: x)

class MCOsolverTestCase(unittest.TestCase):

    def test_instance(self):
        mcosolver = MCOsolver(y0, constr, obj_f, obj_jac)
        self.assertIsInstance(mcosolver, MCOsolver)

    def test_init_types(self):
        mcosolver = MCOsolver(y0, constr, obj_f, obj_jac)
        self.assertEqual(type(obj_f), calltype)
        self.assertEqual(type(obj_jac), calltype)
        self.assertEqual(type(obj_f(y0)), nptype)
        self.assertEqual(type(obj_jac(y0)), nptype)

    def test_init_dim(self):
        mcosolver = MCOsolver(y0, constr, obj_f, obj_jac)
        self.assertEqual(len(mcosolver.constr), 4)
        self.assertEqual(mcosolver.y0.shape, (4,))
        self.assertEqual(obj_jac(y0).shape, (3, 4))
        self.assertEqual(obj_f(y0).shape, (3,))

    def test_solve_return_type(self):
        mcosolver = MCOsolver(y0, constr, obj_f, obj_jac)
        self.assertEqual(type(mcosolver.solve(N=4)), nptype)

    def test_solve_return_shape(self):
        mcosolver = MCOsolver(y0, constr, obj_f, obj_jac)
        self.assertEqual(mcosolver.solve(N=4).shape, (10,4))

    def test_KKTsolver_return_type(self):
        mcosolver = MCOsolver(y0, constr, obj_f, obj_jac)
        self.assertEqual(type(mcosolver.KKTsolver(f, jac)), nptype)

    def test_KKTsolver_return_shape(self):
        mcosolver = MCOsolver(y0, constr, obj_f, obj_jac)
        self.assertEqual(mcosolver.KKTsolver(f, jac).shape, (4,))

    def test_store_curr_res_side_effects(self):
        mcosolver = MCOsolver(y0, constr, obj_f, obj_jac)
        iprime = mcosolver.i
        mcosolver.store_curr_res(y0)
        i = mcosolver.i
        mcosolver.i = mcosolver.res.shape[0]
        reslenprime = mcosolver.res.shape[0]
        mcosolver.store_curr_res(y0)
        self.assertEqual(iprime + 1, i)
        self.assertEqual(reslenprime * 2, mcosolver.res.shape[0])
