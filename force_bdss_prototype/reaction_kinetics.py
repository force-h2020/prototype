import numpy as np
from scipy.special import factorial
from .initializer import Initializer

def analytical_solution(A0, B0, P0, S0, C0, k_ps, t):
    a = alpha(A0, B0, np.sum(k_ps), t)
    A = A0 - a
    B = B0 - a
    P = P0 + k_ps[0] / np.sum(k_ps) * a
    S = S0 + k_ps[1] / np.sum(k_ps) * a
    C = C0
    return np.array([A, B, P, S, C])

def alpha(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > 0.4e-2:
        multiplier = np.exp((B0 - A0) * k * t)
        result = A0 * B0 * (multiplier - 1) / (B0 * multiplier - A0)
    else:
        def factorization(a, b, k, t, n=4):
            exponent = np.arange(1, n + 1, 1)
            denominator = factorial(exponent)
            return np.sum((((b - a) ** (exponent - 1)) * (k * t) ** exponent) / denominator)
        result = (1. / 2.) * (
            (
                (A0 * B0 * factorization(A0, B0, k, t, 4)) /
                (1. + (B0 * factorization(A0, B0, k, t, 4)))
            ) +
            (
                A0 * B0 * factorization(B0, A0, k, t, 4) /
                (1. + (A0 * factorization(B0, A0, k, t, 4)))
            )
        )
    return result

def dalpha(A0, B0, k, t):
    expo = np.exp((B0 - A0) * k * t)
    B0expo = B0 * expo
    da = B0expo * (B0expo + k * t * A0**2 - k * t * A0 * B0 - B0)
    da = da / (B0expo - A0)**2
    db = A0 * ((k * t * B0**2 - k * t * A0 * B0 - A0) * expo + A0)
    db = db / (B0 * expo - A0)**2
    factor = A0 * B0expo * (B0 - A0)**2 / (B0expo - A0)**2
    dk = t * factor
    dt = k * factor
    return np.array([da, db, dk, dt])

def grad_x(A0, B0, P0, S0, C0, k_ps, t):
    grad_x_X_mat = np.empty((5, 7))
    kp, ks = k_ps
    k = np.sum(k_ps)
    kpk = kp / k
    ksk = ks / k
    da, db, dk, dt = dalpha(A0, B0, k, t)
    a = alpha(A0, B0, k, t)
    dada = 1 - da
    dadb = - db
    dadk = - dk
    dadt = - dt
    grad_x_X_mat[0, :] = np.array([dada, dadb, 0, 0, 0, dadk, dadt])
    dbda = - da
    dbdb = 1 - db
    dbdk = - dk
    dbdt = - dt
    grad_x_X_mat[1, :] = np.array([dbda, dbdb, 0, 0, 0, dbdk, dbdt])
    dpda = kpk * da
    dpdb = kpk * db
    dpdp = 1
    dpdk = ksk / k * a + kpk * dk
    dpdt = kpk * dt
    grad_x_X_mat[2, :] = np.array([dpda, dpdb, dpdp, 0, 0, dpdk, dpdt])
    dsda = ksk * da
    dsdb = ksk * db
    dsds = 1
    dsdk = kpk / k * a + ksk * dk
    dsdt = ksk * dt
    grad_x_X_mat[3, :] = np.array([dsda, dsdb, 0, dsds, 0, dsdk, dsdt])
    grad_x_X_mat[4, :] = np.array([0, 0, 0, 0, 1, 0, 0])
    return grad_x_X_mat

def calc_k(T, M):
    M_v, M_delta_H = M
    R = 8.3144598e-3
    k_ps = M_v * np.exp(-M_delta_H / (R * T))
    return k_ps

class Reaction_kinetics:

    def __init__(self):
        self.ini = Initializer()

    def run_default(self, R):
        X = self.ini.get_init_data_kin_model(R)
        M = self.ini.get_material_relation_data(R)
        return self.run(X, M)

    def run(self, X0, M):
        # solver of kinetic module
        R = 8.3144598e-3
        k_ps = calc_k(X0[5], M)
        X_mat = analytical_solution(*X0[:5], k_ps, X0[6])
        grad_x_X_mat = grad_x(*X0[:5], k_ps, X0[6])
        dkdT = 1 / (R * X0[5])**2 * np.sum(k_ps * M[0])
        grad_x_X_mat[:, 5] = dkdT * grad_x_X_mat[:, 5]
        return (X_mat, grad_x_X_mat)
