import unittest
import numpy as np

from ..initializer import Initializer

A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
C = { "name": "contamination", "manufacturer": "", "pdi": 0 }
P = { "name": "product", "manufacturer": "", "pdi": 0 }
R = { "reactants": [A, B], "products": [P] }
p_array = np.array([1, 1, 1])
nptype = type(np.array([]))

class InitializerTestCase(unittest.TestCase):

    def test_instance(self):
        ini = Initializer()
        self.assertIsInstance(ini, Initializer)

    def test_init_data_return_type(self):
        ini = Initializer()
        self.assertEqual(type(ini.get_init_data_kin_model(R)), nptype)

    def test_init_data_return_shape(self):
        ini = Initializer()
        self.assertEqual(ini.get_init_data_kin_model(R).shape, (7,))

    def test_init_data_concentration_consistency(self):
        ini = Initializer()
        concentrations = ini.get_init_data_kin_model(R)[np.array([0, 1, 4])]
        conservation = np.sum(concentrations / p_array)
        self.assertEqual(conservation, 1)

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
