import unittest

from ..objectives import Objectives

class ObjectiveTestCase(unittest.TestCase):

    def test_something(self):
        objectives = Objectives()
        self.assertIsInstance(objectives, Objectives)
