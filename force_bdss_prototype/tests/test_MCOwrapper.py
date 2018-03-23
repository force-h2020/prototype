import unittest
import numpy as np

from ..MCOwrapper import MCOwrapper

A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
C = { "name": "contamination", "manufacturer": "", "pdi": 0 }
P = { "name": "product", "manufacturer": "", "pdi": 0 }
R = { "reactants": [A, B], "products": [P] }
nptype = type(np.array([]))

class MCOwrapperTestCase(unittest.TestCase):

    def test_instance(self):
        mcowrapper = MCOwrapper(R, C)
        self.assertIsInstance(mcowrapper, MCOwrapper)
