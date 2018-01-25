import numpy as np
from .objectives import Objectives
from .constraints import Constraints


class MCOwrapper:
    # default constructor
    def __init__(self, R, C):
        # mco setup: trasform to impl. data structures.
        self.R = R
        self.C = C
        self.obj = Objectives(self.R, self.C)
        self.constraints = Constraints(self.R, self.C)
        bounds = self.constraints.get_linear_constraints(5)

    def solve(self):
        pass
