# Transferred
import numpy as np


class Material_db_access:

    class __Material_db:

        def __init__(self):
            pass

        def get_component_molec_mass(self, X):
            mass = 1.
            return mass

        def get_pure_component_density(self, X):
            # Transferred to json
            # p in particles/l
            if X["name"] == "eductA":
                p = 5
            if X["name"] == "eductB":
                p = 10
            if X["name"] == "contamination":
                p = 55
            return p

        def get_arrhenius_params(self, R):
            # Transferred to json
            if R["products"][0]["name"] == "product":
                # delta H in kJ/mol
                delta_H = 1000 * 8.3144598e-3
                v = 2e-2
            else:
                delta_H = 4500 * 8.3144598e-3
                v = 2e-2
            return (v, delta_H)

        def get_supplier_cost_educt(self, X):
            cost = 1.
            return cost

    instance = None

    def __init__(self):
        # init of Material_db to be done
        if not Material_db_access.instance:
            Material_db_access.instance = Material_db_access.__Material_db()
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)
