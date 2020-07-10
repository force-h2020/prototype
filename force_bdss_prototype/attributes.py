import numpy as np
from .process_db_access import Process_db_access
from .material_db_access import Material_db_access
from .initializer import Initializer
from .reaction_kinetics import Reaction_kinetics
from .reaction_kineticswrapper import Reaction_kineticswrapper
from .function_editor import FunctionApp
from sympy import symbols, Matrix, sympify, diff, evalf, lambdify

class Attributes:
    instance = None

    class __Attributes_Singelton:
        def __init__(self, R, C):
            self.R = R
            self.C = C
            self.p_db_access = Process_db_access(self.R)
            self.reaction_kinetics = Reaction_kinetics()
            self.reaction_kineticswrapper = Reaction_kineticswrapper(self.R, self.C)
            self.ini = Initializer()
            self.M = self.ini.get_material_relation_data(self.R)
            self.m_db_access = Material_db_access()

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
    
    def __init__(self, R, C):
        # init of Process_db to be done
        if not Attributes.instance:
            Attributes.instance = Attributes.__Attributes_Singelton(R,C)
        else:
            pass
    
    def __getattr__(self, name):
        return getattr(self.instance, name)
    
    
    
    
    
    
    
    
    
    
    
    
        