import numpy as np
from .process_db_access import Process_db_access
from .material_db_access import Material_db_access
from .initializer import Initializer
from .reaction_kinetics import Reaction_kinetics
from sympy import symbols, Matrix, sympify, diff, evalf, lambdify

class Reaction_kineticswrapper():
    def __init__(self, R, C):
        self.R = R
        self.C = C
        self.reaction_kinetics = Reaction_kinetics()
        self.p_db_access = Process_db_access(self.R)
        self.m_db_access = Material_db_access()
        self.ini = Initializer(self.R)
        self.M = self.ini.get_material_relation_data(R)
        self.calc_init()

    def calc_init(self):
        #retrieve fixed values
        p_A_value = self.m_db_access.get_pure_component_density(self.R["reactants"][0])
        p_B_value = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
        p_C_value = self.m_db_access.get_pure_component_density(self.C)
        V_r_value, W_value, const_A_value, cost_B_value, quad_coeff_value, C_supplier_value, cost_purification_value = self.p_db_access.get_process_params()
        #fixed value symbols
        p_A, p_B, p_C, V_r, W, const_A, cost_B, quad_coeff, C_supplier, cost_purification = symbols("p_A, p_B, p_C, V_r, W, const_A, cost_B, quad_coeff, C_supplier, cost_purification")

        #grad_X_x setup
        grad_X_x = Matrix([
            [*symbols("dAdA0, dBdA0, dPdA0, dSdA0, dCdA0, dTdA0, dtdA0")],
            [*symbols("dAdB0, dBdB0, dPdB0, dSdB0, dCdB0, dTdB0, dtdB0]")],
            [*symbols("dAdP0, dBdP0, dPdP0, dSdP0, dCdP0, dTdP0, dtdP0]")],
            [*symbols("dAdS0, dBdS0, dPdS0, dSdS0, dCdS0, dTdS0, dtdS0]")],
            [*symbols("dAdC0, dBdC0, dPdC0, dSdC0, dCdC0, dTdC0, dtdC0]")],
            [*symbols("dAdT0, dBdT0, dPdT0, dSdT0, dCdT0, dTdT0, dtdT0]")],
            [*symbols("dAdt0, dBdt0, dPdt0, dSdt0, dCdt0, dTdt0, dtdt0]")]
        ])
    
        #y setup
        V_a, C_e, T, t = symbols("V_a, C_e, T, t")
        self.y = [V_a, C_e, T, t]

        #X-Dim setup
        conc_A, conc_B, conc_P, conc_S, conc_C, T, t = symbols("conc_A, conc_B, conc_P, conc_S, conc_C, T, t")
        self.X_Dim = [conc_A, conc_B, conc_P, conc_S, conc_C, T, t]
        conc_A = sympify("p_A * (1 - C_e / p_C) * V_a / V_r").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        conc_B = sympify("p_B * (V_r - V_a) / V_r").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        conc_P = sympify("0").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        conc_S = sympify("0").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        conc_C = sympify("C_e * V_a / V_r").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        T = sympify("T").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)
        t = sympify("t").subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value)

        #grad_y_X setup
        grad_y_X = []
        for y_h in self.y: grad_y_X.append([conc_A.diff(y_h), conc_B.diff(y_h), conc_P.diff(y_h), conc_S.diff(y_h), conc_C.diff(y_h), T.diff(y_h), t.diff(y_h)])
        grad_y_X = Matrix(grad_y_X)

        #grad_y_XO setup
        self.grad_y_x = grad_y_X * grad_X_x

        #lambdify X-Dim
        conc_A = lambdify(self.y, conc_A)
        conc_B = lambdify(self.y, conc_B)
        conc_P = lambdify(self.y, conc_P)
        conc_S = lambdify(self.y, conc_S)
        conc_C = lambdify(self.y, conc_C)
        T      = lambdify(self.y, T)
        t      = lambdify(self.y, t)

        #X setup
        self.X = lambda y: np.array([conc_A(*y),
                                     conc_B(*y),
                                     conc_P(*y),
                                     conc_S(*y),
                                     conc_C(*y),
                                     T(*y),
                                     t(*y),
                                    ], dtype=np.float)

        #substitute fixed values
        self.grad_y_x = self.grad_y_x.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()


        #lambdify objectives
        #use grad_X_x.transpose() because grad_X_x in calc is transposed
        self.grad_y_x = lambdify(self.y+self.X_Dim[:5]+grad_X_x.transpose().values(), self.grad_y_x)


    def calc_x(self, y):
        x_mat, grad_X_x_mat = self.reaction_kinetics.run(self.X(y), self.M)
        x = np.zeros(7, dtype=np.float)
        x[:5] = x_mat
        x[5] = y[2]
        x[6] = y[3]
        #fix grad_X_x_mat missing T,t
        grad_X_x = np.zeros((x.shape[0], grad_X_x_mat.shape[1]))
        grad_X_x[:5, :] = grad_X_x_mat
        grad_X_x[5, 5] = 1
        grad_X_x[6, 6] = 1
        grad_y_x = Matrix(self.grad_y_x(*y,*x_mat, *grad_X_x.flatten()))
        return (x, grad_y_x)


