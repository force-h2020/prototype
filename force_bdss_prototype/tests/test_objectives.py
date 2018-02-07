import unittest
import numpy as np

from ..objectives import Objectives

A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
C = { "name": "contamination", "manufacturer": "", "pdi": 0 }
P = { "name": "product", "manufacturer": "", "pdi": 0 }
R = { "reactants": [A, B], "products": [P] }
nptype = type(np.array([]))

class ObjectiveTestCase(unittest.TestCase):

    def test_instance(self):
        objectives = Objectives(R, C)
        self.assertIsInstance(objectives, Objectives)

    def test_calc_return_type(self):
        objectives = Objectives(R, C)
        y = np.ones(4)
        self.assertEqual(type(objectives.obj_calc(y)[0]), nptype)
        self.assertEqual(type(objectives.obj_calc(y)[1]), nptype)

    def test_calc_return_shape(self):
        objectives = Objectives(R, C)
        y = np.ones(4)
        self.assertEqual(objectives.obj_calc(y)[0].shape, (3,))
        self.assertEqual(objectives.obj_calc(y)[1].shape, (3, 4))

    def test_x_to_y_return_type(self):
        objectives = Objectives(R, C)
        x = np.ones(7)
        self.assertEqual(type(objectives.x_to_y(x)), nptype)

    def test_x_to_y_return_shape(self):
        objectives = Objectives(R, C)
        x = np.ones(7)
        self.assertEqual(objectives.x_to_y(x).shape, (4,))
