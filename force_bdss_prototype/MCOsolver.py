import numpy as np

class MCOsolver:

    def __init__(self, y0, constr, obj_f, obj_jac):
        self.constr = constr
        self.y0 = y0
        self.obj_f = obj_f
        self.obj_jac = obj_jac
        self.w = np.empty(3)
        self.res = np.zeros((100, 4))
        self.i = 0

    def solve(self):
        new_obj = lambda y: np.dot(self.w, self.obj_f(y))
        new_obj_jac = lambda y: np.dot(self.w, self.obj_jac(y))
        N = 3
        for self.w[0] in np.linspace(0, 1, N):
            for self.w[1] in np.linspace(0, 1 - self.w[0],
                                         N - (N - 1)*self.w[0]):
                self.w[2] = 1 - self.w[0] - self.w[1]
                self.store_curr_res(self.KKTsolver(new_obj, new_obj_jac))
        self.pp_db = Pareto_process_db(self.res[:self.i])
        self.pp_db.write_csv()

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
