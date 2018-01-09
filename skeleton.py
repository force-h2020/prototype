import numpy as np

class Reaction_knowledge_access:

    class __Reaction_knowledge:

        def __init__(self):
            pass

        def get_educts4product(self, P):
            A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
            B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
            return (A, B)

        def get_side_products(self, R):
            A = R["reactants"][0]
            B = R["reactants"][1]
            P = R["products"][0]
            S = { "name": "sideproduct", "manufacturer": "", "pdi": 0 }
            return S

        def good_practice4reaction(self, R):
            A = R["reactants"][0]
            B = R["reactants"][1]
            P = R["products"][0]
            info = ""
            return info

        def estimate_reaction_time(self, R):
            A = R["reactants"][0]
            B = R["reactants"][1]
            P = R["products"][0]
            e_time = 20
            return e_time

    instance = None

    def __init__(self):
        # init of Reaction_knowledge to be done
        if not Reaction_knowledge_access.instance:
            Reaction_knowledge_access.instance = \
                Reaction_knowledge_access.__Reaction_knowledge()
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)


class Material_db_access:

    class __Material_db:

        def __init__(self):
            pass

        def get_component_dat(self, X):
            m = 1
            return m

        def get_pure_component_density(self, X):
            print(X)
            p = 1
            return p

        def get_arrhenius_params(self, R):
            v = 0.1
            grad_H = np.zeros(3, float)
            return (v, grad_H)

    instance = None

    def __init__(self):
        # init of Material_db to be done
        if not Material_db_access.instance:
            Material_db_access.instance = Material_db_access.__Material_db()
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)


class Process_db_access:

    class __Process_db:

        def __init__(self, R):
            self.R = R
            self.V_r = 1
            self.W = 1
            self.cost_A = 1
            self.cost_B = 1
            self.m_db_access = Material_db_access()

        def get_prod_cost(self, X_proc):
            cost = self.V_r*X_proc[1]*(X_proc[0] - 20)**2*self.W
            grad_x_cost = np.zeros(5, float)
            return (cost, grad_x_cost)

        def get_mat_cost(self, X_0_mat):
            cost_A_tilde = 1
            cost_B = 1
            p_B = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
            theta_m = X_0_mat[1]/p_B
            cost = self.V_r*((1 - theta_m)*cost_A_tilde + theta_m*cost_B)
            grad_x_cost = np.zeros(5, float)
            return (cost, grad_x_cost)

        def get_contamination_range(self, A):
            c_min = 0
            c_max = 1
            return (c_min, c_max)

        def get_temp_range(self):
            T_min = 0
            T_max = 1000
            return (T_min, T_max)

        def get_reactor_vol(self):
            return self.V_r

    instance = None

    def __init__(self, R):
        # init of Process_db to be done
        if not Process_db_access.instance:
            Process_db_access.instance = Process_db_access.__Process_db(R)
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)


class Initializer:

    class __Init:

        def __init__(self):
            self.m_db_access = Material_db_access()
            self.react_knowledge = Reaction_knowledge_access()

        def get_init_data_kin_model(self, R):
            A = R["reactants"][0]
            p_db_access = Process_db_access(R)
            X = np.zeros(7)
            T_min, T_max = p_db_access.get_temp_range()
            X[5] = 0.5*(T_max - T_min)
            C_min, C_max = p_db_access.get_contamination_range(A)
            X[4] = 0.5*(C_max - C_min)
            X[2] = 0
            X[3] = 0
            info = self.react_knowledge.good_practice4reaction(R)
            X[0] = 0.5*(1 - X[4])
            X[1] = 0.5*(1 - X[4])
            tau = self.react_knowledge.estimate_reaction_time(R)
            X[6] = tau
            return X

        def get_material_relation_data(self, R):
            S = self.react_knowledge.get_side_products(R)
            R_S = { "reactants": R["reactants"], "products": [S] }
            vp, grad_Hp = self.m_db_access.get_arrhenius_params(R)
            vs, grad_Hs = self.m_db_access.get_arrhenius_params(R_S)
            M_v = np.array([vp, vs])
            M_grad_H = np.array([grad_Hp, grad_Hs])
            return (M_v, M_grad_H)

    instance = None

    def __init__(self):
        # init of Process_db to be done
        if not Initializer.instance:
            Initializer.instance = Initializer.__Init()
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)


class Reaction_kinetics:

    def __init__(self):
        self.ini = Initializer()

    def run_default(self, R):
        X = self.ini.get_init_data_kin_model(R)
        M = self.ini.get_material_relation_data(R)
        return self.run(X, M)

    def run(self, X, M):
        # solver of kinetic module
        X_mat = np.zeros(5)
        grad_x_X_mat = np.array([[0, 0, 0] for x in range(4)])
        return (X_mat, grad_x_X_mat)


class KPI:
    #default constructor
    def __init__(self, R):
        self.R = R
        self.ini = Initializer()
        self.react_kinetics = Reaction_kinetics()
        self.M = self.ini.get_material_relation_data(R)

    # returns impurity concentration I and its x-Gradient
    def kpi_calc(self, X):
        X_mat, grad_x_X_mat = self.react_kinetics.run(X, self.M)
        I = X_mat[3] + X_mat[4] + X_mat[0] + X_mat[1]
        # grad_x_I to be calculated properly
        grad_x_I = np.array([0, 0, 0, 0, 0],float)
        return (I, grad_x_I)


class Objectives:
    # default constructur
    def __init__(self, R, C):
        self.R = R
        self.C = C
        self.p_db_access = Process_db_access(self.R)
        self.kpi = KPI(self.R)
        self.m_db_access = Material_db_access()

    def obj_calc(self, Y):
        p_A = self.m_db_access.get_pure_component_density(self.R["reactants"][0])
        p_B = self.m_db_access.get_pure_component_density(self.R["reactants"][1])
        p_C = self.m_db_access.get_pure_component_density(self.C)
        V_r = self.p_db_access.get_reactor_vol()
        X = np.zeros(7, float)
        X[0] = p_A*(1 - Y[1]/p_C)*Y[1]/V_r
        X[1] = p_B*(V_r - Y[0])/V_r
        X[2] = 0
        X[3] = 0
        X[4] = Y[1]*Y[0]/V_r
        X[5] = Y[2]
        X[6] = Y[3]
        O = np.zeros(3, float)
        (O[0], grad_x_O1) = self.kpi.kpi_calc(X)
        (O[1], grad_x_O2) = self.p_db_access.get_mat_cost(X[:5])
        (O[2], grad_x_O3) = self.p_db_access.get_prod_cost(X[5:])
        grad_x_O = np.array([grad_x_O1, grad_x_O2, grad_x_O3])
        grad_y_O = np.zeros((3, 5))
        for i in range(3):
            # chain role: convert to grad_x to grad_y
            grad_y_O[i] = grad_x_O[i]
        return(O, grad_y_O)


class Constraints:
    # default constructor
    def __init__(self, R, C):
        self.R = R
        self.C = C
        self.react_knowledge = Reaction_knowledge_access()
        self.p_db_access = Process_db_access(self.R)

    def get_linear_constraints(self, N):
        C_range = self.get_contamination_range(self.R["reactants"][0])
        T_range = self.get_temp_range()
        tau_range = (0, self.get_max_reaction_time())
        return (C_range, T_range, tau_range)

    def get_contamination_range(self, educt):
        C_min, C_max = self.p_db_access.get_contamination_range(educt)
        return (C_min, C_max)

    def get_temp_range(self):
        T_min, T_max = self.p_db_access.get_temp_range()
        return (T_min, T_max)

    def get_max_reaction_time(self):
        tau = self.react_knowledge.estimate_reaction_time(self.R)
        # Translators guess
        tau *= 10
        return tau


class MCOwrapper:
    # default constructor
    def __init__(self, R, C):
        # mco setup: trasform to impl. data structures.
        self.R = R
        self.C = C
        self.obj = Objectives(self.R, self.C)
        self.constraints = Constraints(self.R, self.C)
        bounds = self.constraints.get_linear_constraints(5)

    def solve(self):
        pass


# MCO description
# interface of a material
P = { "name": "product", "manufacturer": "", "pdi": 0 }
react_knowledge = Reaction_knowledge_access()
A, B = react_knowledge.get_educts4product(P)
C = { "name": "contamination", "manufacturer": "", "pdi": 0 }
# interface of a reaction
R = { "reactants": [A, B], "products": [P] }
mco_solver = MCOwrapper(R, C)
#mco_solver.solve()
