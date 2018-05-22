import numpy as np


class Process_db_access:

    class __Process_db:

        def __init__(self, R):
            self.R = R
            self.V_r = 1.
            self.W = 1.
            self.cost_A = 1.
            self.cost_B = 1.

        def get_prod_cost(self, X_proc):
            # Transferred
            cost = X_proc[1] * (X_proc[0] - 290)**2 * self.W
            grad_x_cost = np.zeros(7, float)
            grad_x_cost[5] = X_proc[1] * (2 * X_proc[0] - 2 * 290) * self.W
            grad_x_cost[6] = (X_proc[0] - 290)**2 * self.W
            return (cost, grad_x_cost)

        def get_contamination_range(self, A):
            # [C] in mol/l
            c_min = 0.001
            c_max = 0.1
            return (c_min, c_max)

        def get_temp_range(self):
            # T in Kelvin
            T_min = 270
            T_max = 400
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
