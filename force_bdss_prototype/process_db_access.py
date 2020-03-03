import numpy as np


class Process_db_access:

    class __Process_db:

        def __init__(self, R):
            self.R = R
            self.V_r = 1000.
            self.W = 1e-8
            self.const_A = 5e-3
            self.cost_B = 4e-3
            self.quad_coeff = 1e-5
            self.C_supplier = .01 * 5
            self.cost_purification = 0.5e-3

        def get_prod_cost(self, X_proc):
            # Transferred
            cost = X_proc[1] * (X_proc[0] - 290)**2 * self.W
            grad_x_cost = np.zeros(7, float)
            grad_x_cost[5] = X_proc[1] * (2 * X_proc[0] - 2 * 290) * self.W
            grad_x_cost[6] = (X_proc[0] - 290)**2 * self.W
            return (cost, grad_x_cost)

        def get_contamination_range(self, A):
            # Transferred to json
            # [C] in mol/l
            c_min = self.C_supplier * 1e-9
            c_max = self.C_supplier
            return (c_min, c_max)

        def get_temp_range(self):
            # Transferred to json
            # T in Kelvin
            T_min = 270
            T_max = 400
            return (T_min, T_max)

        def get_reactor_vol(self):
            # Transferred to json
            return self.V_r

        def get_C_supplier(self):
            return self.C_supplier

        def get_process_params(self):
            return self.V_r, self.W, self.const_A, self.cost_B, self.quad_coeff, self.C_supplier, self.cost_purification

        def get_mat_cost(self, V_a, C_e, V_b, p_C):
            # Transferred
            # V_a + V_b <= V_r
            V_r = self.V_r
            const_A = self.const_A
            cost_purification = self.cost_purification
            cost_B = self.cost_B
            tot_cost_A = cost_purification * (C_e / self.C_supplier - 1)**2
            tot_cost_A += const_A
            tot_cost_A *= V_a
            tot_cost_A += V_r * self.quad_coeff * (V_a - 600)**2
            tot_cost_B = V_b * cost_B
            cost = float(tot_cost_A + tot_cost_B)
            dva = const_A - cost_B + cost_purification * (C_e / self.C_supplier - 1)**2
            dva += V_r * self.quad_coeff * 2 * (V_a - 600)
            dce = V_a * cost_purification * (2 * C_e / self.C_supplier**2 - 2 / self.C_supplier)
            grad_y_cost = np.array([dva, dce, 0, 0])
            return (cost / 100, grad_y_cost / 100)

    instance = None

    def __init__(self, R):
        # init of Process_db to be done
        if not Process_db_access.instance:
            Process_db_access.instance = Process_db_access.__Process_db(R)
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)
