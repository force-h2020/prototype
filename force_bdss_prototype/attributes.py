import numpy as np
from .reaction_kineticswrapper import Reaction_kineticswrapper

class Attributes:
    def __init__(self, R, C):
        self.R = R
        self.C = C
        self.reaction_kineticswrapper = Reaction_kineticswrapper(self.R, self.C)

    def calc_attributes(self, y):
        a = np.zeros(9, dtype=np.float)
        a[:4] = y
        x, grad_y_x = self.reaction_kineticswrapper.calc_x(y)
        a[4:9] = x[:5]
        grad_y_a = np.zeros((4, 9), dtype=np.float)
        grad_y_a[0, 0] = 1
        grad_y_a[1, 1] = 1
        grad_y_a[2, 2] = 1
        grad_y_a[3, 3] = 1
        grad_y_a[:, 4:] = grad_y_x[:,:5]
        return a, grad_y_a