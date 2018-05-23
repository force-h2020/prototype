import numpy as np


class Reaction_knowledge_access:

    class __Reaction_knowledge:

        def __init__(self):
            pass

        def get_side_products(self, R):
            # Useless
            S = {"name": "sideproduct", "manufacturer": "", "pdi": 0}
            return np.array([S])

        def estimate_reaction_time(self, R):
            # Transferred to json
            # estimated reaction time in s
            e_time = 360.
            return e_time

    instance = None

    def __init__(self):
        # init of Reaction_knowledge to be done
        if not Reaction_knowledge_access.instance:
            Reaction_knowledge_access.instance = \
                Reaction_knowledge_access.__Reaction_knowledge()
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)
