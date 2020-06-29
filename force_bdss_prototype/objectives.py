import numpy as np
from .process_db_access import Process_db_access
from .material_db_access import Material_db_access
from .KPI import KPI
from .initializer import Initializer
from .reaction_kinetics import Reaction_kinetics
from .function_editor import FunctionApp
from sympy import symbols, Matrix, sympify, diff, evalf

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
        self.kpi = KPI(self.R)
        self.reaction_kinetics = Reaction_kinetics()
        self.ini = Initializer()
        self.M = self.ini.get_material_relation_data(self.R)
        self.m_db_access = Material_db_access()
        self.obj_calc_init()

    def obj_calc_init(self):
        p_A_value = self.m_db_access.get_pure_component_density(self.R["reactants"][0])
        p_B_value = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
        p_C_value = self.m_db_access.get_pure_component_density(self.C)
        V_r_value, W_value, const_A_value, cost_B_value, quad_coeff_value, C_supplier_value, cost_purification_value = self.p_db_access.get_process_params()

        dAdA0, dAdB0, dAdP0, dAdS0, dAdC0, dAdT0, dAdt0 = symbols("dAdA0, dAdB0, dAdP0, dAdS0, dAdC0, dAdT0, dAdt0")
        dBdA0, dBdB0, dBdP0, dBdS0, dBdC0, dBdT0, dBdt0 = symbols("dBdA0, dBdB0, dBdP0, dBdS0, dBdC0, dBdT0, dBdt0")
        dPdA0, dPdB0, dPdP0, dPdS0, dPdC0, dPdT0, dPdt0 = symbols("dPdA0, dPdB0, dPdP0, dPdS0, dPdC0, dPdT0, dPdt0")
        dSdA0, dSdB0, dSdP0, dSdS0, dSdC0, dSdT0, dSdt0 = symbols("dSdA0, dSdB0, dSdP0, dSdS0, dSdC0, dSdT0, dSdt0")
        dCdA0, dCdB0, dCdP0, dCdS0, dCdC0, dCdT0, dCdt0 = symbols("dCdA0, dCdB0, dCdP0, dCdS0, dCdC0, dCdT0, dCdt0")
        dTdA0, dTdB0, dTdP0, dTdS0, dTdC0, dTdT0, dTdt0 = symbols("dTdA0, dTdB0, dTdP0, dTdS0, dTdC0, dTdT0, dTdt0")
        dtdA0, dtdB0, dtdP0, dtdS0, dtdC0, dtdT0, dtdt0 = symbols("dtdA0, dtdB0, dtdP0, dtdS0, dtdC0, dtdT0, dtdt0")

        conc_A, conc_B, conc_P, conc_S, conc_C, T, t = symbols("conc_A, conc_B, conc_P, conc_S, conc_C, T, t")
        V_a, C_e, T, t = symbols("V_a, C_e, T, t")


        self.X = (conc_A, conc_B, conc_P, conc_S, conc_C, T, t)
        self.y = (V_a, C_e, T, t)
        p_A, p_B, p_C, V_r, W, const_A, cost_B, quad_coeff, C_supplier, cost_purification = symbols("p_A, p_B, p_C, V_r, W, const_A, cost_B, quad_coeff, C_supplier, cost_purification")
        conc_A = sympify("p_A * (1 - C_e / p_C) * V_a / V_r").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        conc_B = sympify("p_B * (V_r - V_a) / V_r").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        conc_P = sympify("0").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        conc_S = sympify("0").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        conc_C = sympify("C_e * V_a / V_r").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        T = sympify("T").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        t = sympify("t").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        grad_y_X = []
        for y_h in self.y:
            grad_y_X.append([conc_A.diff(y_h), conc_B.diff(y_h), conc_P.diff(y_h), conc_S.diff(y_h), conc_C.diff(y_h), T.diff(y_h), t.diff(y_h)])
        #print(np.array(grad_y_X)).subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)ubs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        grad_y_X = Matrix(grad_y_X)

        grad_X_x = Matrix([
            [dAdA0, dBdA0, dPdA0, dSdA0, dCdA0, dTdA0, dtdA0],
            [dAdB0, dBdB0, dPdB0, dSdB0, dCdB0, dTdB0, dtdB0],
            [dAdP0, dBdP0, dPdP0, dSdP0, dCdP0, dTdP0, dtdP0],
            [dAdS0, dBdS0, dPdS0, dSdS0, dCdS0, dTdS0, dtdS0],
            [dAdC0, dBdC0, dPdC0, dSdC0, dCdC0, dTdC0, dtdC0],
            [dAdT0, dBdT0, dPdT0, dSdT0, dCdT0, dTdT0, dtdT0],
            [dAdt0, dBdt0, dPdt0, dSdt0, dCdt0, dTdt0, dtdt0]
        ])

        #functions: key: id, value[0]: description, value[1]: function, value[2]: isEditable
        functions = {"pc" : ["Production Cost","t * (T - 290)^2 * W", False],
                        "mc" : ["Material Cost" , "mcA + mcB", False],
                        "mcA" : ["Mat cost A" , "(cost_purification * (C_e / C_supplier -1)^2 + const_A) * V_a", True],
                        "mcB" : ["Mat cost B" , "(V_r - V_a) * cost_B", True],
                        "imp" : ["Impurity Concentration" , "conc_A + conc_B + conc_C + conc_S", False]}
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

        editorInput = [functions, var]
        self.XO, self.yO, self.grad_x_XO, self.grad_y_yO = FunctionApp().run_with_output(editorInput, -1)

        # self.XO, self.yO, self.grad_y_yO aus function editor
        self.grad_y_XO = grad_y_X * grad_X_x * self.grad_x_XO
        self.X = lambda y: np.array([conc_A.subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]).evalf(),
                                     conc_B.subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]).evalf(),
                                     conc_P.subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]).evalf(),
                                     conc_S.subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]).evalf(),
                                     conc_C.subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]).evalf(),
                                     T.subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]).evalf(),
                                     t.subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]).evalf()
                                     ], dtype=np.float)

        self.yO = self.yO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()
        self.XO = self.XO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()
        self.grad_y_yO = self.grad_y_yO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()
        self.grad_y_XO = self.grad_y_XO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()


    def obj_calc(self, y):
        dAdA0, dAdB0, dAdP0, dAdS0, dAdC0, dAdT0, dAdt0 = symbols("dAdA0, dAdB0, dAdP0, dAdS0, dAdC0, dAdT0, dAdt0")
        dBdA0, dBdB0, dBdP0, dBdS0, dBdC0, dBdT0, dBdt0 = symbols("dBdA0, dBdB0, dBdP0, dBdS0, dBdC0, dBdT0, dBdt0")
        dPdA0, dPdB0, dPdP0, dPdS0, dPdC0, dPdT0, dPdt0 = symbols("dPdA0, dPdB0, dPdP0, dPdS0, dPdC0, dPdT0, dPdt0")
        dSdA0, dSdB0, dSdP0, dSdS0, dSdC0, dSdT0, dSdt0 = symbols("dSdA0, dSdB0, dSdP0, dSdS0, dSdC0, dSdT0, dSdt0")
        dCdA0, dCdB0, dCdP0, dCdS0, dCdC0, dCdT0, dCdt0 = symbols("dCdA0, dCdB0, dCdP0, dCdS0, dCdC0, dCdT0, dCdt0")
        dTdA0, dTdB0, dTdP0, dTdS0, dTdC0, dTdT0, dTdt0 = symbols("dTdA0, dTdB0, dTdP0, dTdS0, dTdC0, dTdT0, dTdt0")
        dtdA0, dtdB0, dtdP0, dtdS0, dtdC0, dtdT0, dtdt0 = symbols("dtdA0, dtdB0, dtdP0, dtdS0, dtdC0, dtdT0, dtdt0")

        conc_A, conc_B, conc_P, conc_S, conc_C, T, t = symbols("conc_A conc_B conc_P conc_S conc_C T t")
        V_a, C_e, T, t = symbols("V_a, C_e, T, t")

        X_value = self.X(y)
        x_mat, grad_X_x_mat = self.reaction_kinetics.run(X_value, self.M)
        grad_X_x = np.zeros((X_value.shape[0], grad_X_x_mat.shape[1]))
        grad_X_x[:5, :] = grad_X_x_mat
        grad_X_x[5, 5] = 1
        grad_X_x[6, 6] = 1
        hgrad_y_XO = self.grad_y_XO.subs(dAdA0, grad_X_x[0,0]).subs(dBdA0, grad_X_x[1,0]).subs(dPdA0, grad_X_x[2,0]).subs(dSdA0, grad_X_x[3,0]).subs(dCdA0, grad_X_x[4,0]).subs(dTdA0, grad_X_x[5,0]).subs(dtdA0, grad_X_x[6,0])
        hgrad_y_XO = hgrad_y_XO.subs(dAdB0, grad_X_x[0,1]).subs(dBdB0, grad_X_x[1,1]).subs(dPdB0, grad_X_x[2,1]).subs(dSdB0, grad_X_x[3,1]).subs(dCdB0, grad_X_x[4,1]).subs(dTdB0, grad_X_x[5,1]).subs(dtdB0, grad_X_x[6,1])
        hgrad_y_XO = hgrad_y_XO.subs(dAdP0, grad_X_x[0,2]).subs(dBdP0, grad_X_x[1,2]).subs(dPdP0, grad_X_x[2,2]).subs(dSdP0, grad_X_x[3,2]).subs(dCdP0, grad_X_x[4,2]).subs(dTdP0, grad_X_x[5,2]).subs(dtdP0, grad_X_x[6,2])
        hgrad_y_XO = hgrad_y_XO.subs(dAdS0, grad_X_x[0,3]).subs(dBdS0, grad_X_x[1,3]).subs(dPdS0, grad_X_x[2,3]).subs(dSdS0, grad_X_x[3,3]).subs(dCdS0, grad_X_x[4,3]).subs(dTdS0, grad_X_x[5,3]).subs(dtdS0, grad_X_x[6,3])
        hgrad_y_XO = hgrad_y_XO.subs(dAdC0, grad_X_x[0,4]).subs(dBdC0, grad_X_x[1,4]).subs(dPdC0, grad_X_x[2,4]).subs(dSdC0, grad_X_x[3,4]).subs(dCdC0, grad_X_x[4,4]).subs(dTdC0, grad_X_x[5,4]).subs(dtdC0, grad_X_x[6,4])
        hgrad_y_XO = hgrad_y_XO.subs(dAdT0, grad_X_x[0,5]).subs(dBdT0, grad_X_x[1,5]).subs(dPdT0, grad_X_x[2,5]).subs(dSdT0, grad_X_x[3,5]).subs(dCdT0, grad_X_x[4,5]).subs(dTdT0, grad_X_x[5,5]).subs(dtdT0, grad_X_x[6,5])
        hgrad_y_XO = hgrad_y_XO.subs(dAdt0, grad_X_x[0,6]).subs(dBdt0, grad_X_x[1,6]).subs(dPdt0, grad_X_x[2,6]).subs(dSdt0, grad_X_x[3,6]).subs(dCdt0, grad_X_x[4,6]).subs(dTdt0, grad_X_x[5,6]).subs(dtdt0, grad_X_x[6,6])
        O = np.zeros(len(self.yO) + len(self.XO), dtype=np.float)
        grad_y_O = np.zeros((len(self.yO) + len(self.XO), 4), dtype=np.float)
        for i in range(len(self.yO)):
            # print(self.yO[i].subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]))
            O[i + len(self.XO)] = self.yO[i].subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]).evalf()
        for i in range(len(self.XO)):
            O[i] = self.XO[i].subs(conc_A, x_mat[0]).subs(conc_B, x_mat[1]).subs(conc_P, x_mat[2]).subs(conc_S, x_mat[3]).subs(conc_C, x_mat[4]).evalf()
        for i in range(len(self.yO)):
            for j in range(4):
                grad_y_O[i + len(self.XO), j] = self.grad_y_yO[j, i].subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]).evalf()
        for i in range(len(self.XO)):
            for j in range(4):
                grad_y_O[i, j] = hgrad_y_XO[j, i].subs(V_a, y[0]).subs(C_e, y[1]).subs(T, y[2]).subs(t, y[3]).subs(conc_A, x_mat[0]).subs(conc_B, x_mat[1]).subs(conc_P, x_mat[2]).subs(conc_S, x_mat[3]).subs(conc_C, x_mat[4]).evalf().subs(conc_A, x_mat[0]).subs(conc_B, x_mat[1]).subs(conc_P, x_mat[2]).subs(conc_S, x_mat[3]).subs(conc_C, x_mat[4]).evalf()
        return (O, grad_y_O)

    def obj_calc_old(self, y):
        p_A = self.m_db_access.get_pure_component_density(self.R["reactants"][0])
        p_B = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
        p_C = self.m_db_access.get_pure_component_density(self.C)
        V_r = self.p_db_access.get_reactor_vol()
        X = np.zeros(7, float)
        X[0] = p_A * (1 - y[1] / p_C) * y[0] / V_r
        X[1] = p_B * (V_r - y[0]) / V_r
        X[2] = 0
        X[3] = 0
        X[4] = y[1] * y[0] / V_r
        X[5] = y[2]
        X[6] = y[3]
        O = np.zeros(3, float)
        (O[0], grad_x_O1) = self.kpi.kpi_calc(X)
        (O[1], grad_x_O2) = self.p_db_access.get_prod_cost(X[5:])
        (O[2], grad_y_O3) = self.p_db_access.get_mat_cost(y[0], y[1], (V_r - y[0]), p_C)
        grad_x_O = np.array([grad_x_O1, grad_x_O2])
        dadVa = p_A * (1 - y[1] / p_C) / V_r
        dadCe = - p_A * y[0] / (p_C * V_r)
        da = np.array([dadVa, dadCe])
        dbdVa = - p_B / V_r
        dcdVa = y[1] / V_r
        dcdCe = y[0] / V_r
        dc = np.array([dcdVa, dcdCe])
        grad_y_x = np.zeros((7, 4))
        grad_y_x[0, :2] = da
        grad_y_x[1, 0] = dbdVa
        grad_y_x[4, :2] = dc
        grad_y_x[5, 2] = 1
        grad_y_x[6, 3] = 1
        grad_y_O = np.empty((3, 4))
        grad_y_O[:2] = np.dot(grad_x_O, grad_y_x)
        grad_y_O[2] = grad_y_O3
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
