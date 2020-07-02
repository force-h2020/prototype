import numpy as np
from .process_db_access import Process_db_access
from .material_db_access import Material_db_access
from .initializer import Initializer
from .reaction_kinetics import Reaction_kinetics
from .function_editor import FunctionApp
from .attributes import Attributes

class Objectives:
    """ Objectives class
    """
    # default constructur
    def __init__(self, R, C, attributes):
        """ Constructor requires ...
        Parmeters
        ---------
        R : type
            Description of R
        """
        self.R = R
        self.C = C
        self.p_db_access = Process_db_access(self.R)
        self.reaction_kinetics = Reaction_kinetics()
        self.ini = Initializer()
        self.M = self.ini.get_material_relation_data(self.R)
        self.m_db_access = Material_db_access()
        self.attributes = attributes

    def obj_calc(self, y):
        O, grad_y_O = self.attributes.attributes_calc(y)
        return (O, grad_y_O)

    def x_to_y(self, X):
        V_r = self.p_db_access.get_reactor_vol()
        p_B = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
        y = np.zeros(4)
        y[0] = V_r - X[1] * V_r / p_B
        y[1] = V_r / y[0] * X[4]
        y[2] = X[5]
        y[3] = X[6]
        return y

    def y_to_x(self, y):
        p_A = self.m_db_access.get_pure_component_density(self.R["reactants"][0])
        p_B = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
        p_C = self.m_db_access.get_pure_component_density(self.C)
        V_r = self.p_db_access.get_reactor_vol()
        X = np.zeros(7, float)
        X[0] = p_A*(1 - y[1]/p_C)*y[0]/V_r
        X[1] = p_B*(V_r - y[0])/V_r
        X[2] = 0
        X[3] = 0
        X[4] = y[1]*y[0]/V_r
        X[5] = y[2]
        X[6] = y[3]
        return X
