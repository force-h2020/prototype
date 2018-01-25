import numpy as np
from .material_db_access import Material_db_access

class Process_db_access:

    class __Process_db:

        def __init__(self, R):
            self.R = R
            self.V_r = 1
            self.W = 1
            self.cost_A = 1
            self.cost_B = 1
            self.m_db_access = Material_db_access()

        def get_prod_cost(self, X_proc):
            cost = self.V_r*X_proc[1]*(X_proc[0] - 20)**2*self.W
            grad_x_cost = np.zeros(5, float)
            return (cost, grad_x_cost)

        def get_mat_cost(self, X_0_mat):
            cost_A_tilde = 1
            cost_B = 1
            p_B = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
            theta_m = X_0_mat[1]/p_B
            cost = self.V_r*((1 - theta_m)*cost_A_tilde + theta_m*cost_B)
            grad_x_cost = np.zeros(5, float)
            return (cost, grad_x_cost)

        def get_contamination_range(self, A):
            c_min = 0
            c_max = 1
            return (c_min, c_max)

        def get_temp_range(self):
            T_min = 0
            T_max = 1000
            return (T_min, T_max)

        def get_reactor_vol(self):
            return self.V_r

    instance = None

    def __init__(self, R):
        # init of Process_db to be done
        if not Process_db_access.instance:
            Process_db_access.instance = Process_db_access.__Process_db(R)
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)
