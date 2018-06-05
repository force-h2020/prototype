import unittest
import numpy as np

from ..material_db_access import Material_db_access

A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
P = { "name": "product", "manufacturer": "", "pdi": 0 }
R = { "reactants": [A, B], "products": [P] }
nptype = type(np.array([]))


class Material_db_accessTestCase(unittest.TestCase):

    def test_instance(self):
        m_db = Material_db_access()
        self.assertIsInstance(m_db, Material_db_access)

    def test_component_density_return_type(self):
        m_db = Material_db_access()
        self.assertEqual(type(m_db.get_pure_component_density(A)), float)

    def test_arrhenius_params_return_type(self):
        m_db = Material_db_access()
        v, delta_H = m_db.get_arrhenius_params(R)
        self.assertEqual(type(v), float)
        self.assertEqual(type(delta_H), float)

    def test_mat_cost_return_type(self):
        m_db = Material_db_access()
        cost, grad_y_cost = m_db.get_mat_cost(0.5, 0.1, 1, 1)
        self.assertEqual(type(cost), float)
        self.assertEqual(type(grad_y_cost), nptype)

    def test_mat_cost_return_shape(self):
        m_db = Material_db_access()
        _, grad_y_cost = m_db.get_mat_cost(0.5, 0.1, 1, 1)
        self.assertEqual(grad_y_cost.shape, (4,))
