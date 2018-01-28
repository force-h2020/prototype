import numpy as np
from scipy.special import factorial
from .initializer import Initializer

    

def analytical_solution(A0, B0, P0, S0, C0, kp, ks, time):
    a = alpha(time, A0, B0, kp, ks, time)
    A = A0 - a
    B = B0 - a
    P = P0 + (kp / (kp + ks)) * a
    S = S0 + (ks / (kp + ks)) * a
    C = C0

    return A, B, P, S, C


def alpha(time, A0, B0, kp, ks, t):

    k = kp + ks
    epsilon = np.abs((A0 - B0) * (kp + ks) * time)
    if epsilon > 0.4E-2:
        multiplier = np.exp((B0 - A0) ** (k * time))
        result = A0 * B0 * (multiplier - 1) / (B0 * multiplier - A0)
    else:
        def factorization(x, k, t, n=4):
            denominator = factorial(n)
            exponent = np.arange(n, 0, -1)
            return np.sum(((x ** exponent) * (k * t) ** n) / denominator)
        result = (1. / 2.) * (
            (
                (A0 * B0 * factorization(B0 - A0, k, t, 4)) /
                (1. + (B0 * factorization(B0 - A0, k, t, 4)))
            ) +
            (
                A0 * B0 * factorization(A0 - B0, k, t, 4) /
                (1. + (A0 * factorization(A0 - B0, k, t, 4)))
            )
        )
    return result


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



