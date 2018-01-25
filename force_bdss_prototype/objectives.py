import numpy as np
from .process_db_access import Process_db_access
from .material_db_access import Material_db_access
from .KPI import KPI

class Objectives:
    # default constructur
    def __init__(self, R, C):
        self.R = R
        self.C = C
        self.p_db_access = Process_db_access(self.R)
        self.kpi = KPI(self.R)
        self.m_db_access = Material_db_access()

    def obj_calc(self, Y):
        p_A = self.m_db_access.get_pure_component_density(self.R["reactants"][0])
        p_B = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
        p_C = self.m_db_access.get_pure_component_density(self.C)
        V_r = self.p_db_access.get_reactor_vol()
        X = np.zeros(7, float)
        X[0] = p_A*(1 - Y[1]/p_C)*Y[1]/V_r
        X[1] = p_B*(V_r - Y[0])/V_r
        X[2] = 0
        X[3] = 0
        X[4] = Y[1]*Y[0]/V_r
        X[5] = Y[2]
        X[6] = Y[3]
        O = np.zeros(3, float)
        (O[0], grad_x_O1) = self.kpi.kpi_calc(X)
        (O[1], grad_x_O2) = self.p_db_access.get_mat_cost(X[:5])
        (O[2], grad_x_O3) = self.p_db_access.get_prod_cost(X[5:])
        grad_x_O = np.array([grad_x_O1, grad_x_O2, grad_x_O3])
        grad_y_O = np.zeros((3, 5))
        for i in range(3):
            # chain role: convert to grad_x to grad_y
            grad_y_O[i] = grad_x_O[i]
        return(O, grad_y_O)
