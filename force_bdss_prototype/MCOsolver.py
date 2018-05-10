import numpy as np
import scipy.optimize as sp_opt
from .pareto_process_db import Pareto_process_db

class MCOsolver:

    w = np.array([0.4, 0.4, 0.2])
    res = np.zeros((100, 4))
    i = 0

    def __init__(self, y0, constr, obj_f, obj_jac):
        self.constr = constr
        self.y0 = y0
        self.obj_f = obj_f
        self.obj_jac = obj_jac

    def solve(self, N=7):
        new_obj = lambda y: np.dot(self.w, self.obj_f(y))
        new_obj_jac = lambda y: np.dot(self.w, self.obj_jac(y))
        for self.w[0] in np.linspace(0, 1, N):
            for self.w[1] in np.linspace(0, 1 - self.w[0],
                                         N - int((N - 1)*self.w[0])):
                self.w[2] = 1 - self.w[0] - self.w[1]
                #if not np.any(self.w == 0):
                self.store_curr_res(self.KKTsolver(new_obj, new_obj_jac))
        return self.res[:self.i]

    def KKTsolver(self, new_obj, new_obj_jac):
        opt_res = sp_opt.minimize(new_obj, self.y0, method="SLSQP",
                                  jac=new_obj_jac , bounds=self.constr).x
        return opt_res

    def store_curr_res(self, y):
        if self.i >= self.res.shape[0]:
            res = np.zeros((2*self.res.shape[0], 4))
            res[:self.res.shape[0]] = self.res
            self.res = res
        self.res[self.i] = y
        self.i = self.i + 1
