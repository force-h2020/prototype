import numpy as np

class Material_db_access:

    class __Material_db:

        def __init__(self):
            pass

        def get_component_dat(self, X):
            m = 1
            return m

        def get_pure_component_density(self, X):
            # p in mol/l
            p = 30
            return p

        def get_arrhenius_params(self, R):
            v = 1e10
            delta_H = 300
            return (v, delta_H)

        def get_mat_cost(self, V_a, C_e, V_r, p_C):
            const_A = 1
            const_C = 1
            const_B = 1
            cost_A = V_a * (1 - C_e / p_C) * const_A + const_C * p_C / C_e
            cost_B = (V_r - V_a) * const_B
            cost = cost_A + cost_B
            dva = ((1 - C_e / p_C) * const_A + const_C * p_C / C_e) - const_B
            dce = - V_a * const_A / p_C - const_C * p_C / (C_e)**2
            grad_x_cost = np.array([dva, dce, 0, 0])
            return (cost, grad_x_cost)

    instance = None

    def __init__(self):
        # init of Material_db to be done
        if not Material_db_access.instance:
            Material_db_access.instance = Material_db_access.__Material_db()
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)
