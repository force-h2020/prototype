from .reaction_knowledge_access import Reaction_knowledge_access
from .process_db_access import Process_db_access
from .constraints_editor import EditorApp


class Constraints:
    # default constructor
    def __init__(self, R):
        self.R = R
        self.react_knowledge = Reaction_knowledge_access()
        self.p_db_access = Process_db_access(self.R)

    def get_editor_constraints(self):
        ## Volume A without Slider hint ///
        ## Contamination e with Slider <-- C_bar
        ## Temperature with Slider <-- t
        ## Reaction Time with Slider <-- tau

        va_range = self.get_va_range()
        C_range = self.get_contamination_range(self.R["reactants"][0])
        T_range = self.get_temp_range()
        tau_range = (1e-2, self.get_max_reaction_time())


        constraints = [{"name": "Volume A","unit": "m³", "min": va_range[0], "max": va_range[1]},
                   {"name": "Concentration e","unit": "ppm", "min": C_range[0], "max": C_range[1]},
                   {"name": "Temperature", "unit": "K", "min": T_range[0], "max": T_range[1]},
                   {"name": "Reaction time", "unit": "s", "min": tau_range[0], "max": tau_range[1]}]

        constraints = EditorApp().runWithOutput(constraints, constraints)

        va_range  = (constraints[0].get('min'), constraints[0].get('max'))
        C_range   = (constraints[1].get('min'), constraints[1].get('max'))
        T_range   = (constraints[2].get('min'), constraints[2].get('max'))
        tau_range = (constraints[3].get('min'), constraints[3].get('max'))
        return (va_range, C_range, T_range, tau_range)

    def get_linear_constraints(self):
        # Transferred to json
        va_range = self.get_va_range()
        C_range = self.get_contamination_range(self.R["reactants"][0])
        T_range = self.get_temp_range()
        tau_range = (1e-2, self.get_max_reaction_time())
        return (va_range, C_range, T_range, tau_range)

    def get_va_range(self):
        # Transferred to json
        V_r = self.p_db_access.get_reactor_vol()
        return (1e-9 * V_r, V_r)

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
        tau *= 3
        return tau
