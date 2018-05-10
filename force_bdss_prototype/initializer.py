import numpy as np
from .material_db_access import Material_db_access
from .reaction_knowledge_access import Reaction_knowledge_access
from .process_db_access import Process_db_access

class Initializer:

    class __Init:

        def __init__(self):
            self.m_db_access = Material_db_access()
            self.react_knowledge = Reaction_knowledge_access()

        def get_init_data_kin_model(self, R):
            A = R["reactants"][0]
            B = R["reactants"][1]
            p_db_access = Process_db_access(R)
            X = np.zeros(7)
            T_min, T_max = p_db_access.get_temp_range()
            X[5] = T_min + 0.5 * (T_max - T_min)
            C_min, C_max = p_db_access.get_contamination_range(A)
            X[4] = C_min + 0.5 * (C_max - C_min)
            X[2] = 0
            X[3] = 0
            info = self.react_knowledge.good_practice4reaction(R)
            X[0] = 0.5 - X[4]
            X[1] = 0.5
            tau = self.react_knowledge.estimate_reaction_time(R)
            X[6] = tau
            return X

        def get_material_relation_data(self, R):
            S = self.react_knowledge.get_side_products(R)[0]
            R_S = { "reactants": R["reactants"], "products": [S] }
            vp, grad_Hp = self.m_db_access.get_arrhenius_params(R)
            vs, grad_Hs = self.m_db_access.get_arrhenius_params(R_S)
            M_v = np.array([vp, vs])
            M_delta_H = np.array([grad_Hp, grad_Hs])
            return (M_v, M_delta_H)

    instance = None

    def __init__(self):
        # init of Process_db to be done
        if not Initializer.instance:
            Initializer.instance = Initializer.__Init()
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)
