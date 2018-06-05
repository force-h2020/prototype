import unittest
import numpy as np

from ..reaction_knowledge_access import Reaction_knowledge_access

A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
C = { "name": "contamination", "manufacturer": "", "pdi": 0 }
P = { "name": "product", "manufacturer": "", "pdi": 0 }
R = { "reactants": [A, B], "products": [P] }
nptype = type(np.array([]))


class Reaction_knowledge_accessTestCase(unittest.TestCase):

    def test_instance(self):
        react_knowldg = Reaction_knowledge_access()
        self.assertIsInstance(react_knowldg, Reaction_knowledge_access)

    def test_sideproducts_return_type(self):
        react_knowldg = Reaction_knowledge_access()
        self.assertEqual(type(react_knowldg.get_side_products(R)), nptype)

    def test_estimate_reaction_time_return_type(self):
        react_knowldg = Reaction_knowledge_access()
        self.assertEqual(type(react_knowldg.estimate_reaction_time(R)), float)

    def test_estimate_reaction_time_return(self):
        react_knowldg = Reaction_knowledge_access()
        self.assertTrue(react_knowldg.estimate_reaction_time(R) > 0.)
