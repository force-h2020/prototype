import numpy as np
from .process_db_access import Process_db_access
from .material_db_access import Material_db_access
from .KPI import KPI

class Objectives:
    """ Objectives class


    """
    # default constructur
    def __init__(self, R, C):
        """ Constructor requires ...

        Parmeters
        ---------
        R : type
            Description of R

        """
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
        (O[1], grad_x_O2) = self.p_db_access.get_prod_cost(X[5:])
        (O[2], grad_y_O3) = self.m_db_access.get_mat_cost(*Y[:2], V_r, p_C)
        grad_x_O = np.array([grad_x_O1, grad_x_O2])
        dadVa = p_A * (1 - Y[1] / p_C) / V_r
        dadCe = - p_A * Y[0] / (p_C * V_r)
        da = np.array([dadVa, dadCe])
        dbdVa = - p_B / V_r
        dcdVa = Y[1] / V_r
        dcdCe = Y[0] / V_r
        dc = np.array([dcdVa, dcdCe])
        grad_y_x = np.zeros((7, 4))
        grad_y_x[0, :2] = da
        grad_y_x[1, 0] = dbdVa
        grad_y_x[4, :2] = dc
        grad_y_x[5, 2] = 1
        grad_y_x[6, 3] = 1
        grad_y_O = np.empty((3, 4))
        grad_y_O[:2] = np.dot(grad_x_O, grad_y_x)
        grad_y_O[2] = grad_y_O3
        return(O, grad_y_O)

    def x_to_y(self, X):
        V_r = self.p_db_access.get_reactor_vol()
        p_B = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
        y = np.zeros(4)
        y[0] = V_r - X[1]*V_r/p_B
        y[1] = V_r/y[0]*X[4]
        y[2] = X[5]
        y[3] = X[6]
        return y
