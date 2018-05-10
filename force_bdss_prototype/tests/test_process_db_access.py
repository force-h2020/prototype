import unittest
import numpy as np

from ..process_db_access import Process_db_access

A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
P = { "name": "product", "manufacturer": "", "pdi": 0 }
R = { "reactants": [A, B], "products": [P] }
nptype = type(np.array([]))

class Process_db_accessTestCase(unittest.TestCase):

    def test_instance(self):
        p_db = Process_db_access(R)
        self.assertIsInstance(p_db, Process_db_access)

    def test_prod_cost_return_type(self):
        p_db = Process_db_access(R)
        cost, grad_x_cost = p_db.get_prod_cost(np.array([350., 3600.]))
        self.assertEqual(type(grad_x_cost), nptype)

    def test_prod_cost_return_shape(self):
        p_db = Process_db_access(R)
        cost, grad_x_cost = p_db.get_prod_cost(np.array([350., 3600.]))
        self.assertEqual(grad_x_cost.shape, (7,))

    def test_contamination_range(self):
        p_db = Process_db_access(R)
        cmin, cmax = p_db.get_contamination_range(A)
        self.assertTrue(cmin < cmax)

    def test_temp_range(self):
        p_db = Process_db_access(R)
        tmin, tmax = p_db.get_temp_range()
        self.assertTrue(tmin < tmax)

    def test_reactor_vol(self):
        p_db = Process_db_access(R)
        V_r = p_db.get_reactor_vol()
        self.assertEqual(type(V_r), float)
