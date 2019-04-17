import numpy as np
from .initializer import Initializer
from .reaction_kinetics import Reaction_kinetics
from .process_db_access import Process_db_access


class KPI:
    # Transferred
    #default constructor
    def __init__(self, R):
        self.R = R
        self.ini = Initializer()
        self.react_kinetics = Reaction_kinetics()
        self.M = self.ini.get_material_relation_data(R)
        p_db_access = Process_db_access(R)
        self.C_supplier = p_db_access.get_C_supplier()

    # returns impurity concentration I and its x-Gradient
    def kpi_calc(self, X):
        X_mat, grad_x_X_mat = self.react_kinetics.run(X, self.M)
        I = float(X_mat[3] + X_mat[4] + X_mat[0] + X_mat[1])
        dIda = np.sum(grad_x_X_mat[0:2, 0]) + np.sum(grad_x_X_mat[3:5, 0])
        dIdb = np.sum(grad_x_X_mat[0:2, 1]) + np.sum(grad_x_X_mat[3:5, 1])
        dIdp = np.sum(grad_x_X_mat[0:2, 2]) + np.sum(grad_x_X_mat[3:5, 2])
        dIds = np.sum(grad_x_X_mat[0:2, 3]) + np.sum(grad_x_X_mat[3:5, 3])
        dIdc = np.sum(grad_x_X_mat[0:2, 4]) + np.sum(grad_x_X_mat[3:5, 4])
        dIdT = np.sum(grad_x_X_mat[0:2, 5]) + np.sum(grad_x_X_mat[3:5, 5])
        dIdt = np.sum(grad_x_X_mat[0:2, 6]) + np.sum(grad_x_X_mat[3:5, 6])
        grad_x_I = np.array([dIda, dIdb, dIdp, dIds, dIdc, dIdT, dIdt]) / I
        return (float(np.log(I / self.C_supplier)), grad_x_I)
