import unittest
import numpy as np

from ..KPI import KPI

A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
C = { "name": "contamination", "manufacturer": "", "pdi": 0 }
P = { "name": "product", "manufacturer": "", "pdi": 0 }
R = { "reactants": [A, B], "products": [P] }
X0 = np.array([0.495, 0.5, 0, 0, 0.005, 330, 3600], float)
nptype = type(np.array([]))

class KPITestCase(unittest.TestCase):

    def test_instance(self):
        kpi = KPI(R)
        self.assertIsInstance(kpi, KPI)

    def test_kpi_calc_return_type(self):
        kpi = KPI(R)
        I, grad_x_I = kpi.kpi_calc(X0)
        self.assertEqual(type(I), float)
        self.assertEqual(type(grad_x_I), nptype)

    def test_kpi_calc_return_shape(self):
        kpi = KPI(R)
        _, grad_x_I = kpi.kpi_calc(X0)
        self.assertEqual(type(grad_x_I), nptype)
