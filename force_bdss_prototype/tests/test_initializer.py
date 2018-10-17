import unittest
import numpy as np

from ..initializer import Initializer
from ..material_db_access import Material_db_access

A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
C = { "name": "contamination", "manufacturer": "", "pdi": 0 }
P = { "name": "product", "manufacturer": "", "pdi": 0 }
R = { "reactants": [A, B], "products": [P] }
m_db_access = Material_db_access()
p_A = m_db_access.get_pure_component_density(A)
p_B = m_db_access.get_pure_component_density(B)
p_C = m_db_access.get_pure_component_density(C)
p_array = np.array([p_A, p_B, p_C])
nptype = type(np.array([]))

class InitializerTestCase(unittest.TestCase):

    def test_instance(self):
        ini = Initializer()
        self.assertIsInstance(ini, Initializer)

    def test_init_data_return_type(self):
        ini = Initializer()
        self.assertEqual(type(ini.get_init_data_kin_model(R, C)), nptype)

    def test_init_data_return_shape(self):
        ini = Initializer()
        self.assertEqual(ini.get_init_data_kin_model(R, C).shape, (7,))

    def test_init_data_concentration_consistency(self):
        ini = Initializer()
        concentrations = ini.get_init_data_kin_model(R, C)[np.array([0, 1, 4])]
        conservation = np.sum(concentrations / p_array)
        self.assertTrue(conservation - 1 < 1e-6)

    def test_mat_relation_return_type(self):
        ini = Initializer()
        M_v, M_delta_H = ini.get_material_relation_data(R)
        self.assertEqual(type(M_v), nptype)
        self.assertEqual(type(M_delta_H), nptype)

    def test_mat_relation_return_shape(self):
        ini = Initializer()
        M_v, M_delta_H = ini.get_material_relation_data(R)
        self.assertEqual(M_v.shape, (2,))
        self.assertEqual(M_delta_H.shape, (2,))

    def test_mat_relation_delta_H(self):
        ini = Initializer()
        _, M_delta_H = ini.get_material_relation_data(R)
        self.assertTrue(M_delta_H[0] < M_delta_H[1])
