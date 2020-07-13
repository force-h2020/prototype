import numpy as np
from scipy.special import factorial
from .initializer import Initializer

tol = 4e-1

def _analytical_solution(A0, B0, P0, S0, C0, k_ps, t):
    a = _alpha(A0, B0, np.sum(k_ps), t)
    A = A0 - a
    B = B0 - a
    P = P0 + k_ps[0] / np.sum(k_ps) * a
    S = S0 + k_ps[1] / np.sum(k_ps) * a
    C = C0
    return np.array([A, B, P, S, C])


def _sum1(A0, B0, k, t, n=4):
    exponent = np.arange(1, n + 1, 1)
    denominator = factorial(exponent)
    result = ((B0 - A0) ** (exponent - 1)) * (k * t) ** exponent
    return np.sum(result / denominator)


def _sum2(A0, B0, k, t, n=4):
    exponent = np.arange(2, n + 1, 1)
    denominator = factorial(exponent) / (exponent - 1)
    result = ((B0 - A0) ** (exponent - 2)) * (k * t) ** exponent
    return np.sum(result / denominator)


def _sum3(A0, B0, k, t, n=4):
    exponent = np.arange(1, n + 1, 1)
    denominator = factorial(exponent) / exponent
    result = ((B0 - A0) ** (exponent - 1)) * (k * t) ** (exponent - 1)
    return np.sum(result / denominator)


def _alpha(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > tol:
        multiplier = np.exp((B0 - A0) * k * t)
        result = A0 * B0 * (multiplier - 1) / (B0 * multiplier - A0)
    else:
        sum1ab = _sum1(A0, B0, k, t)
        sum1ba = _sum1(B0, A0, k, t)
        result = (A0 * B0 * sum1ab) / (1. + (B0 * sum1ab))
        result += (A0 * B0 * sum1ba) / (1. + (A0 * sum1ba))
        result /= 2
    return result


def _dalda(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > tol:
        B0expo = B0 * np.exp((B0 - A0) * k * t)
        result = B0expo * (B0expo + k * t * A0**2 - k * t * A0 * B0 - B0)
        result /= (B0expo - A0)**2
    else:
        sum1ab = _sum1(A0, B0, k, t)
        sum1ba = _sum1(B0, A0, k, t)
        sum2ab = _sum2(A0, B0, k, t)
        sum2ba = _sum2(B0, A0, k, t)
        p1 = (B0 * sum1ab - A0 * B0 * sum2ab) * (1 + B0 * sum1ab)
        p2 = (-B0 * sum2ab) * (A0 * B0 * sum1ab)
        p3 = (B0 * sum1ba + A0 * B0 * sum2ba) * (1 + A0 * sum1ba)
        p4 = (sum1ba + A0 * sum2ba) * (A0 * B0 * sum1ba)
        result = (p1 - p2) / (1 + B0 * sum1ab)**2
        result += (p3 - p4) / (1 + A0 * sum1ba)**2
        result /= 2
    return result


def _daldb(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > tol:
        expo = np.exp((B0 - A0) * k * t)
        B0expo = B0 * expo
        result = A0 * ((k * t * B0**2 - k * t * A0 * B0 - A0) * expo + A0)
        result /= (B0expo - A0)**2
    else:
        sum1ab = _sum1(A0, B0, k, t)
        sum1ba = _sum1(B0, A0, k, t)
        sum2ab = _sum2(A0, B0, k, t)
        sum2ba = _sum2(B0, A0, k, t)
        p1 = (A0 * sum1ab + A0 * B0 * sum2ab) * (1 + B0 * sum1ab)
        p2 = (sum1ab + B0 * sum2ab) * A0 * B0 * sum1ab
        p3 = (A0 * sum1ba - A0 * B0 * sum2ba) * (1 + A0 * sum1ba)
        p4 = A0 * (-sum2ba) * A0 * B0 * sum1ba
        result = (p1 - p2) / (1 + B0 * sum1ab)**2
        result += (p3 - p4) / (1 + A0 * sum1ba)**2
        result /= 2
    return result


def _daldk(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > tol:
        B0expo = B0 * np.exp((B0 - A0) * k * t)
        result = t * A0 * B0expo * (B0 - A0)**2 / (B0expo - A0)**2
    else:
        sum1ab = _sum1(A0, B0, k, t)
        sum1ba = _sum1(B0, A0, k, t)
        sum3ab = _sum3(A0, B0, k, t)
        sum3ba = _sum3(B0, A0, k, t)
        p1 = A0 * B0 * t * sum3ab * (1 + B0 * sum1ab)
        p2 = B0 * t * sum3ab * A0 * B0 * sum1ab
        p3 = A0 * B0 * t * sum3ba * (1 + A0 * sum1ba)
        p4 = A0 * t * sum3ba * A0 * B0 * sum1ba
        result = (p1 - p2) / (1 + B0 * sum1ab)**2
        result += (p3 - p4) / (1 + A0 * sum1ba)**2
        result /= 2
    return result


def _daldt(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > tol:
        B0expo = B0 * np.exp((B0 - A0) * k * t)
        result = k * A0 * B0expo * (B0 - A0)**2 / (B0expo - A0)**2
    else:
        sum1ab = _sum1(A0, B0, k, t)
        sum1ba = _sum1(B0, A0, k, t)
        sum3ab = _sum3(A0, B0, k, t)
        sum3ba = _sum3(B0, A0, k, t)
        p1 = A0 * B0 * k * sum3ab * (1 + B0 * sum1ab)
        p2 = B0 * k * sum3ab * A0 * B0 * sum1ab
        p3 = A0 * B0 * k * sum3ba * (1 + A0 * sum1ba)
        p4 = A0 * k * sum3ba * A0 * B0 * sum1ba
        result = (p1 - p2) / (1 + B0 * sum1ab)**2
        result += (p3 - p4) / (1 + A0 * sum1ba)**2
        result /= 2
    return result


def _grad_x(A0, B0, P0, S0, C0, k_ps, t):
    grad_x_X_mat = np.empty((5, 7))
    kp, ks = k_ps
    k = np.sum(k_ps)
    kpk = kp / k
    ksk = ks / k
    da = _dalda(A0, B0, k, t)
    db = _daldb(A0, B0, k, t)
    dk = _daldk(A0, B0, k, t)
    dt = _daldt(A0, B0, k, t)
    al = _alpha(A0, B0, k, t)
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
    dpdk = kpk * dk
    dpdt = kpk * dt
    grad_x_X_mat[2, :] = np.array([dpda, dpdb, dpdp, 0, 0, dpdk, dpdt])
    dsda = ksk * da
    dsdb = ksk * db
    dsds = 1
    dsdk = ksk * dk
    dsdt = ksk * dt
    grad_x_X_mat[3, :] = np.array([dsda, dsdb, 0, dsds, 0, dsdk, dsdt])
    grad_x_X_mat[4, :] = np.array([0, 0, 0, 0, 1, 0, 0])
    return grad_x_X_mat


def _calc_k(T, M):
    M_v, M_delta_H = M
    R = 8.3144598e-3
    k_ps = M_v * np.exp(-M_delta_H / (R * T))
    return k_ps


class Reaction_kinetics:

    def run_default(self, R, C):
        self.ini = Initializer(R)
        X = self.ini.get_init_data_kin_model(R, C)
        M = self.ini.get_material_relation_data(R)
        return self.run(X, M)

    def run(self, X0, M):
        # solver of kinetic module
        R = 8.3144598e-3
        k_ps = _calc_k(X0[5], M)
        X_mat = _analytical_solution(X0[0],
                                     X0[1],
                                     X0[2],
                                     X0[3],
                                     X0[4],
                                     k_ps,
                                     X0[6])
        grad_x_X_mat = _grad_x(X0[0],
                               X0[1],
                               X0[2],
                               X0[3],
                               X0[4],
                               k_ps,
                               X0[6])
        dkdT = 1 / (R * X0[5]**2) * np.sum(k_ps * M[1])
        grad_x_X_mat[:, 5] = dkdT * grad_x_X_mat[:, 5]
        dkskdT = (M[1][1] - M[1][0]) * k_ps[0] * k_ps[1]
        dkskdT /= R * X0[5]**2 * (k_ps[0] * M[0][1] / M[0][0] + k_ps[1] * M[0][0] / M[0][1])**2
        dkpkdT = - dkskdT
        grad_x_X_mat[2, 5] += _alpha(X0[0], X0[1], np.sum(k_ps), X0[6]) * dkpkdT
        grad_x_X_mat[3, 5] += _alpha(X0[0], X0[1], np.sum(k_ps), X0[6]) * dkskdT
        return (X_mat, grad_x_X_mat)
