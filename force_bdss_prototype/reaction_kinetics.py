import numpy as np
from .initializer import Initializer

class Reaction_kinetics:

    def __init__(self):
        self.ini = Initializer()

    def run_default(self, R):
        X = self.ini.get_init_data_kin_model(R)
        M = self.ini.get_material_relation_data(R)
        return self.run(X, M)

    def run(self, X, M):
        # solver of kinetic module
        X_mat = np.zeros(5)
        grad_x_X_mat = np.array([[0, 0, 0] for x in range(4)])
        return (X_mat, grad_x_X_mat)
