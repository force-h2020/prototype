import numpy as np

class Material_db_access:

    class __Material_db:

        def __init__(self):
            pass

        def get_component_dat(self, X):
            m = 1
            return m

        def get_pure_component_density(self, X):
            p = 1
            return p

        def get_arrhenius_params(self, R):
            v = 0.1
            grad_H = np.zeros(3, float)
            return (v, grad_H)

    instance = None

    def __init__(self):
        # init of Material_db to be done
        if not Material_db_access.instance:
            Material_db_access.instance = Material_db_access.__Material_db()
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)
