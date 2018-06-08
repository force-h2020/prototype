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

def sum1(A0, B0, k, t, n=4):
    exponent = np.arange(1, n + 1, 1)
    denominator = factorial(exponent)
    result = ((B0 - A0) ** (exponent - 1)) * (k * t) ** exponent
    return np.sum(result / denominator)

def sum2(A0, B0, k, t, n=4):
    exponent = np.arange(2, n + 1, 1)
    denominator = factorial(exponent) / (exponent - 1)
    result = ((B0 - A0) ** (exponent - 2)) * (k * t) ** exponent
    return np.sum(result / denominator)

def sum3(A0, B0, k, t, n=4):
    exponent = np.arange(1, n + 1, 1)
    denominator = factorial(exponent) / exponent
    result = ((B0 - A0) ** (exponent - 1)) * (k * t) ** (exponent - 1)
    return np.sum(result / denominator)

def alpha(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > 0.4e-2:
        multiplier = np.exp((B0 - A0) * k * t)
        result = A0 * B0 * (multiplier - 1) / (B0 * multiplier - A0)
    else:
        sum1ab = sum1(A0, B0, k, t)
        sum1ba = sum1(B0, A0, k, t)
        result = (A0 * B0 * sum1ab) / (1. + (B0 * sum1ab))
        result += (A0 * B0 * sum1ba) / (1. + (A0 * sum1ba))
        result /= 2
    return result

def dalda(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > 0.4e-2:
        B0expo = B0 * np.exp((B0 - A0) * k * t)
        result = B0expo * (B0expo + k * t * A0**2 - k * t * A0 * B0 - B0)
        result /= (B0expo - A0)**2
    else:
        sum1ab = sum1(A0, B0, k, t)
        sum1ba = sum1(B0, A0, k, t)
        sum2ab = sum2(A0, B0, k, t)
        sum2ba = sum2(B0, A0, k, t)
        p1 = (B0 * sum1ab - A0 * B0 * sum2ab) * (1 + B0 * sum1ab)
        p2 = (-B0 * sum2ab) * (A0 * B0 * sum1ab)
        p3 = (B0 * sum1ba + A0 * B0 * sum2ba) * (1 + A0 * sum1ba)
        p4 = (sum1ba + A0 * sum2ba) * (A0 * B0 * sum1ba)
        result = (p1 - p2) / (1 + B0 * sum1ab)**2
        result += (p3 - p4) / (1 + A0 * sum1ba)**2
        result /= 2
    return result

def daldb(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > 0.4e-2:
        expo = np.exp((B0 - A0) * k * t)
        B0expo = B0 * expo
        result = A0 * ((k * t * B0**2 - k * t * A0 * B0 - A0) * expo + A0)
        result /= (B0expo - A0)**2
    else:
        sum1ab = sum1(A0, B0, k, t)
        sum1ba = sum1(B0, A0, k, t)
        sum2ab = sum2(A0, B0, k, t)
        sum2ba = sum2(B0, A0, k, t)
        p1 = (A0 * sum1ab + A0 * B0 * sum2ab) * (1 + B0 * sum1ab)
        p2 = (sum1ab + B0 * sum2ab) * A0 * B0 * sum1ab
        p3 = (A0 * sum1ba - A0 * B0 * sum2ba) * (1 + A0 * sum1ba)
        p4 = A0 * (-sum2ba) * A0 * B0 * sum1ba
        result = (p1 - p2) / (1 + B0 * sum1ab)**2
        result += (p3 - p4) / (1 + A0 * sum1ba)**2
        result /= 2
    return result

def daldk(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > 0.4e-2:
        B0expo = B0 * np.exp((B0 - A0) * k * t)
        result = t * A0 * B0expo * (B0 - A0)**2 / (B0expo - A0)**2
    else:
        sum1ab = sum1(A0, B0, k, t)
        sum1ba = sum1(B0, A0, k, t)
        sum3ab = sum3(A0, B0, k, t)
        sum3ba = sum3(B0, A0, k, t)
        p1 = A0 * B0 * t * sum3ab * (1 + B0 * sum1ab)
        p2 = B0 * t * sum3ab * A0 * B0 * sum1ab
        p3 = A0 * B0 * t * sum3ba * (1 + A0 * sum1ba)
        p4 = A0 * t * sum3ba * A0 * B0 * sum1ba
        result = (p1 - p2) / (1 + B0 * sum1ab)**2
        result += (p3 - p4) / (1 + A0 * sum1ba)**2
        result /= 2
    return result

def daldt(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > 0.4e-2:
        B0expo = B0 * np.exp((B0 - A0) * k * t)
        result = k * A0 * B0expo * (B0 - A0)**2 / (B0expo - A0)**2
    else:
        sum1ab = sum1(A0, B0, k, t)
        sum1ba = sum1(B0, A0, k, t)
        sum3ab = sum3(A0, B0, k, t)
        sum3ba = sum3(B0, A0, k, t)
        p1 = A0 * B0 * k * sum3ab * (1 + B0 * sum1ab)
        p2 = B0 * k * sum3ab * A0 * B0 * sum1ab
        p3 = A0 * B0 * k * sum3ba * (1 + A0 * sum1ba)
        p4 = A0 * k * sum3ba * A0 * B0 * sum1ba
        result = (p1 - p2) / (1 + B0 * sum1ab)**2
        result += (p3 - p4) / (1 + A0 * sum1ba)**2
        result /= 2
    return result

def grad_x(A0, B0, P0, S0, C0, k_ps, t):
    grad_x_X_mat = np.empty((5, 7))
    kp, ks = k_ps
    k = np.sum(k_ps)
    kpk = kp / k
    ksk = ks / k
    da = dalda(A0, B0, k, t)
    db = daldb(A0, B0, k, t)
    dk = daldk(A0, B0, k, t)
    dt = daldt(A0, B0, k, t)
    al = alpha(A0, B0, k, t)
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
    dpdk = ksk / k * al + kpk * dk
    dpdt = kpk * dt
    grad_x_X_mat[2, :] = np.array([dpda, dpdb, dpdp, 0, 0, dpdk, dpdt])
    dsda = ksk * da
    dsdb = ksk * db
    dsds = 1
    dsdk = kpk / k * al + ksk * dk
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
