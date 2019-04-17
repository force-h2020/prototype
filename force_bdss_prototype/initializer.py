import numpy as np
from .material_db_access import Material_db_access
from .reaction_knowledge_access import Reaction_knowledge_access
from .process_db_access import Process_db_access


class Initializer:

    class __Init:

        def __init__(self):
            self.m_db_access = Material_db_access()
            self.react_knowledge = Reaction_knowledge_access()

        def get_init_data_kin_model(self, R, C):
            # Transferred to json
            A = R["reactants"][0]
            B = R["reactants"][1]
            p_A = self.m_db_access.get_pure_component_density(A)
            p_B = self.m_db_access.get_pure_component_density(B)
            p_C = self.m_db_access.get_pure_component_density(C)
            p_db_access = Process_db_access(R)
            X = np.zeros(7)
            T_min, T_max = p_db_access.get_temp_range()
            X[5] = 0.5 * (T_max + T_min)
            C_min, C_max = p_db_access.get_contamination_range(A)
            C_bar = 0.5 * (C_max + C_min)
            E = p_C - C_bar
            E /= p_C / p_A + p_C / p_B - C_bar * p_C / (p_B * p_C)
            X[2] = 0
            X[3] = 0
            X[0] = E
            X[1] = E
            X[4] = C_bar * (1 - X[1] / p_B)
            tau = self.react_knowledge.estimate_reaction_time(R)
            X[6] = tau
            return X

        def get_material_relation_data(self, R):
            # Transferred to json
            S = self.react_knowledge.get_side_products(R)[0]
            R_S = { "reactants": R["reactants"], "products": [S] }
            vp, delta_Hp = self.m_db_access.get_arrhenius_params(R)
            vs, delta_Hs = self.m_db_access.get_arrhenius_params(R_S)
            M_v = np.array([vp, vs])
            M_delta_H = np.array([delta_Hp, delta_Hs])
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
