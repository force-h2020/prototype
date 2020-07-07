import numpy as np
from .process_db_access import Process_db_access
from .material_db_access import Material_db_access
from .initializer import Initializer
from .reaction_kinetics import Reaction_kinetics
from .function_editor import FunctionApp
from .attributes import Attributes
from sympy import symbols, Matrix, sympify, diff, evalf, lambdify

class Objectives:
    """ Objectives class
    """
    # default constructur
    def __init__(self, R, C):
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
        self.XO, self.yO, self.grad_x_XO, self.grad_y_yO = FunctionApp().run_with_output(self.function_editor_input(), -1)
        self.attributes = Attributes(R, C, self.XO, self.yO, self.grad_x_XO, self.grad_y_yO)
        self.obj_calc_init()

    def obj_calc_init(self):
        #retrieve fixed values
        p_A_value = self.m_db_access.get_pure_component_density(self.R["reactants"][0])
        p_B_value = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
        p_C_value = self.m_db_access.get_pure_component_density(self.C)
        V_r_value, W_value, const_A_value, cost_B_value, quad_coeff_value, C_supplier_value, cost_purification_value = self.p_db_access.get_process_params()
        #fixed value symbols
        p_A, p_B, p_C, V_r, W, const_A, cost_B, quad_coeff, C_supplier, cost_purification = symbols("p_A, p_B, p_C, V_r, W, const_A, cost_B, quad_coeff, C_supplier, cost_purification")
    
        #y setup
        V_a, C_e, T, t = symbols("V_a, C_e, T, t")
        self.y = [V_a, C_e, T, t]

        #X-Dim setup
        conc_A, conc_B, conc_P, conc_S, conc_C, T, t = symbols("conc_A, conc_B, conc_P, conc_S, conc_C, T, t")
        self.X_Dim = [conc_A, conc_B, conc_P, conc_S, conc_C, T, t]

        #substitute fixed values
        self.yO = self.yO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()
        self.XO = self.XO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()
        self.grad_y_yO = self.grad_y_yO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()
        self.grad_x_XO = self.grad_x_XO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()
        self.yO = lambdify(self.y, self.yO)
        self.XO = lambdify(self.X_Dim , self.XO)
        self.grad_y_yO = lambdify(self.y, self.grad_y_yO)
        self.grad_x_XO = lambdify(self.X_Dim, self.grad_x_XO)

        # delete when function editor updated
        self.O = lambda a: np.array([self.yO(*a[:4])[0], self.yO(*a[:4])[1], self.XO(*a[4:])], dtype=np.float).flatten()

    # delete when function editor is updated
    def grad_a_O_calc(self, a):
        grad_a_O = np.zeros((11, 3), dtype=np.float)
        grad_a_O[:4, :2] = self.grad_y_yO(*a[:4])
        grad_a_O[4:11, 2] = self.grad_x_XO(*a[4:]).flatten()
        return grad_a_O


    def obj_calc(self, y):
        a, grad_y_a = self.attributes.calc_attributes(y)
        O = self.O(a)
        grad_a_O = self.grad_a_O_calc(a)
        grad_y_O = np.dot(grad_y_a, grad_a_O)
        return (O, grad_y_O.T)

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

    def function_editor_input(self):
        #functions: key: id, value[0]: description, value[1]: function, value[2]: isEditable
        functions = {"pc" : ["Production Cost","t * (T - 290)^2 * W", False],
                        "mc" : ["Material Cost" , "mcA + mcB", False],
                        "mcA" : ["Mat cost A" , "(cost_purification * (C_e / C_supplier -1)^2 + const_A) * V_a + V_r * quad_coeff * (V_a - 0.6 * V_r)**2", True],
                        "mcB" : ["Mat cost B" , "(V_r - V_a) * cost_B", True],
                        "imp" : ["Impurity Concentration" , "ln((conc_A + conc_B + conc_C + conc_S )/ C_supplier)", False]}
        #variables: key: id, value[0]: description, value[1]: isFixedParameter
        var = {"conc_A" : ("concentration of A","X"),
                "conc_B" : ("concentration of B","X"),
                "conc_P" : ("concentration of P","X"),
                "conc_S" : ("concentration of S","X"),
                "conc_C" : ("concentration of C","X"),
                "V_a" : ("volume of A","y"),
                "C_e" : ("impurity of A","y"),
                "T" : ("temperature","y, X"),
                "t" : ("reaction time","y, X"),
                "p_A" : ("pure density of A","fixed"),
                "p_B" : ("pure density of A","fixed"),
                "p_C" : ("pure density of A","fixed"),
                "V_r" : ("reactor volume","fixed"),
                "W" : ("heating cost","fixed"),
                "const_A" : ("description","fixed"),
                "cost_B" : ("description","fixed"),
                "quad_coeff" : ("description","fixed"),
                "C_supplier" : ("description","fixed"),
                "cost_purification" : ("description","fixed")}
        return [functions, var]
