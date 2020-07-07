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
        def __init__(self, R, C, XO, yO, grad_x_XO, grad_y_yO):
            self.R = R
            self.C = C
            self.p_db_access = Process_db_access(self.R)
            self.reaction_kinetics = Reaction_kinetics()
            self.reaction_kineticswrapper = Reaction_kineticswrapper(self.R, self.C)
            self.ini = Initializer()
            self.M = self.ini.get_material_relation_data(self.R)
            self.m_db_access = Material_db_access()
            #runs function editor to get User defined attribute functions
            self.XO, self.yO, self.grad_x_XO, self.grad_y_yO = XO, yO, grad_x_XO, grad_y_yO
            self.attributes_calc_init_alt()

        def calc_attributes(self, y):
            a = np.zeros(11, dtype=np.float)
            a[:4] = y

            x, grad_y_x = self.reaction_kineticswrapper.calc_x(y)
            a[4:11] = x

            grad_y_a = np.zeros((4, 11), dtype=np.float)
            grad_y_a[0, 0] = 1
            grad_y_a[1, 1] = 1
            grad_y_a[2, 2] = 1
            grad_y_a[3, 3] = 1

            grad_y_a[:, 4:] = grad_y_x

            return a, grad_y_a

        def attributes_calc_init_alt(self):
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
            self.grad_y_XO = grad_y_X * grad_X_x * self.grad_x_XO

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
            self.yO = self.yO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()
            self.XO = self.XO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()
            self.grad_y_yO = self.grad_y_yO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()
            self.grad_y_XO = self.grad_y_XO.subs(p_A, p_A_value).subs(p_B, p_B_value).subs(p_C, p_C_value).subs(V_r, V_r_value).subs(W, W_value).subs(const_A, const_A_value).subs(cost_B, cost_B_value).subs(quad_coeff, quad_coeff_value).subs(C_supplier,C_supplier_value).subs(cost_purification, cost_purification_value).evalf()

            # #lambdify objectives
            self.yO = lambdify(self.y, self.yO)
            self.XO = lambdify(self.X_Dim , self.XO)
            self.grad_y_yO = lambdify(self.y, self.grad_y_yO)
            #use grad_X_x.transpose() because grad_X_x in calc is transposed
            self.grad_y_XO = lambdify(self.y+self.X_Dim[:5]+grad_X_x.transpose().values(), self.grad_y_XO)

        def attributes_calc_alt(self, y):
            X_value = self.X(y)
            x_mat, grad_X_x_mat = self.reaction_kinetics.run(X_value, self.M)
            #fix grad_X_x_mat missing T,t
            grad_X_x = np.zeros((X_value.shape[0], grad_X_x_mat.shape[1]))
            grad_X_x[:5, :] = grad_X_x_mat
            grad_X_x[5, 5] = 1
            grad_X_x[6, 6] = 1

            #calc yO
            yO = self.yO(*y)
            #calc XO // *y[2:] <--- fixes missing T,t
            XO = self.XO(*x_mat,*y[2:])
            #calc grad_y_yO
            grad_y_yO = Matrix(self.grad_y_yO(*y))
            #calc grad_y_XO
            grad_y_XO = Matrix(self.grad_y_XO(*y,*x_mat, *grad_X_x.flatten()))

            #insert
            O = np.zeros(len(yO) + len(XO), dtype=np.float)
            grad_y_O = np.zeros((len(yO) + len(XO), 4), dtype=np.float)
            for i in range(len(yO)):
                O[i] = yO[i]
            for i in range(len(XO)):
                O[i + len(yO)] = XO[i]
            for i in range(len(yO)):
                for j in range(4):
                    grad_y_O[i, j] = grad_y_yO[j, i]
            for i in range(len(XO)):
                for j in range(4):
                    grad_y_O[i + len(yO), j] = grad_y_XO[j,i]
            return (O, grad_y_O) 

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
    
    def __init__(self, R, C, XO, yO, grad_x_XO, grad_y_yO):
        # init of Process_db to be done
        if not Attributes.instance:
            Attributes.instance = Attributes.__Attributes_Singelton(R,C, XO, yO, grad_x_XO, grad_y_yO)
        else:
            pass
    
    def __getattr__(self, name):
        return getattr(self.instance, name)
    
    
    
    
    
    
    
    
    
    
    
    
        