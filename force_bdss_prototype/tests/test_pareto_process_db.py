import unittest
import numpy as np

from ..pareto_process_db import Pareto_process_db

data = np.arange(10)

class Pareto_process_dbTestCase(unittest.TestCase):

    def test_instance(self):
        pp_db = Pareto_process_db(data)
        self.assertIsInstance(pp_db, Pareto_process_db)

    def test_init(self):
        pp_db = Pareto_process_db(np.copy(data))
        self.assertTrue(np.all(data == pp_db.data))
