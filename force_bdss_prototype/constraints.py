from .reaction_knowledge_access import Reaction_knowledge_access
from .process_db_access import Process_db_access


class Constraints:
    # default constructor
    def __init__(self, R):
        self.R = R
        self.react_knowledge = Reaction_knowledge_access()
        self.p_db_access = Process_db_access(self.R)

    def get_linear_constraints(self):
        # Transferred to json
        va_range = self.get_va_range()
        C_range = self.get_contamination_range(self.R["reactants"][0])
        T_range = self.get_temp_range()
        tau_range = (0, self.get_max_reaction_time())
        return (va_range, C_range, T_range, tau_range)

    def get_va_range(self):
        # Transferred to json
        return (0, self.p_db_access.get_reactor_vol())

    def get_contamination_range(self, educt):
        # Transferred to json
        C_min, C_max = self.p_db_access.get_contamination_range(educt)
        return (C_min, C_max)

    def get_temp_range(self):
        # Transferred to json
        T_min, T_max = self.p_db_access.get_temp_range()
        return (T_min, T_max)

    def get_max_reaction_time(self):
        tau = self.react_knowledge.estimate_reaction_time(self.R)
        # Translators guess
        tau *= 10
        return tau
