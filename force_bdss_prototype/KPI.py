import numpy as np
from .initializer import Initializer
from .reaction_kinetics import Reaction_kinetics

class KPI:
    #default constructor
    def __init__(self, R):
        self.R = R
        self.ini = Initializer()
        self.react_kinetics = Reaction_kinetics()
        self.M = self.ini.get_material_relation_data(R)

    # returns impurity concentration I and its x-Gradient
    def kpi_calc(self, X):
        X_mat, grad_x_X_mat = self.react_kinetics.run(X, self.M)
        I = X_mat[3] + X_mat[4] + X_mat[0] + X_mat[1]
        # grad_x_I to be calculated properly
        grad_x_I = np.array([0, 0, 0, 0, 0],float)
        return (I, grad_x_I)
