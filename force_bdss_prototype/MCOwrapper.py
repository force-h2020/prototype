import numpy as np
from .objectives import Objectives
from .constraints import Constraints
from .initializer import Initializer
from .MCOsolver import MCOsolver


class MCOwrapper:
    # default constructor
    def __init__(self, R, C):
        # mco setup: trasform to impl. data structures.
        self.R = R
        self.C = C
        self.obj = Objectives(self.R, self.C)
        self.constraints = Constraints(self.R, self.C)
        self.ini = Initializer()
        constr = self.constraints.get_linear_constraints(5)
        obj_f = lambda y: self.obj.obj_calc(y)[0]
        obj_jac = lambda y: self.obj.obj_calc(y)[1]
        X0 = self.ini.get_init_data_kin_model(self.R)
        y0 = self.obj.x_to_y(X0)
        self.mcosolver = MCOsolver(y0, constr, obj_f, obj_jac)

    def solve(self):
        self.mcosolver.solve()
