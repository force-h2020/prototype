import numpy as np
from scipy.special import factorial

from force_bdss.core.slot import Slot
from force_bdss.data_sources.base_data_source import BaseDataSource


class ImpurityConcentrationDataSource(BaseDataSource):
    def run(self, model, parameters):
        V_a_tilde = parameters[0].value
        C_conc_e = parameters[1].value
        temperature = parameters[2].value
        reaction_time = parameters[3].value
        arrhenius_nu_main_reaction = parameters[4].value
        arrhenius_delta_H_main_reaction = parameters[5].value
        arrhenius_nu_secondary_reaction = parameters[6].value
        arrhenius_delta_H_secondary_reaction = parameters[7].value
        reactor_volume = parameters[8].value
        A_density = parameters[9].value
        B_density = parameters[10].value
        C_density = parameters[11].value

        X = np.zeros(7, float)
        X[0] = A_density * (1 -
                            C_conc_e / C_density) * V_a_tilde / reactor_volume
        X[1] = B_density * (reactor_volume - V_a_tilde) / reactor_volume
        X[2] = 0
        X[3] = 0
        X[4] = C_conc_e * V_a_tilde / reactor_volume
        X[5] = temperature
        X[6] = reaction_time

        M = (
            np.array([arrhenius_nu_main_reaction,
                      arrhenius_nu_secondary_reaction]),
            np.array([arrhenius_delta_H_main_reaction,
                      arrhenius_delta_H_secondary_reaction])
        )
        X_mat, grad_x_X_mat = _run(X, M)
        impurity_conc = float(X_mat[3] + X_mat[4] + X_mat[0] + X_mat[1])
        dIda = np.sum(grad_x_X_mat[0:2, 0] + grad_x_X_mat[3:5, 0])
        dIdb = np.sum(grad_x_X_mat[0:2, 1] + grad_x_X_mat[3:5, 1])
        dIdp = np.sum(grad_x_X_mat[0:2, 2] + grad_x_X_mat[3:5, 2])
        dIds = np.sum(grad_x_X_mat[0:2, 3] + grad_x_X_mat[3:5, 3])
        dIdc = np.sum(grad_x_X_mat[0:2, 4] + grad_x_X_mat[3:5, 4])
        dIdT = np.sum(grad_x_X_mat[0:2, 5] + grad_x_X_mat[3:5, 5])
        dIdt = np.sum(grad_x_X_mat[0:2, 6] + grad_x_X_mat[3:5, 6])
        grad_x_I = np.array([dIda, dIdb, dIdp, dIds, dIdc, dIdT, dIdt])
        return impurity_conc, grad_x_I

    def slots(self, model):
        return (
            (
                Slot(description="V_A_tilde", type="VOLUME"),
                Slot(description="C_e concentration", type="CONCENTRATION"),
                Slot(description="Temperature", type="TEMPERATURE"),
                Slot(description="Reaction time", type="TIME"),
                Slot(description="Arrhenius nu main reaction",
                     type="ARRHENIUS_NU"),
                Slot(description="Arrhenius delta H main reaction",
                     type="ARRHENIUS_DELTA_H"),
                Slot(description="Arrhenius nu secondary reaction",
                     type="ARRHENIUS_NU"),
                Slot(description="Arrhenius delta H secondary reaction",
                     type="ARRHENIUS_DELTA_H"),
                Slot(description="Reactor volume", type="VOLUME"),
                Slot(description="A pure density", type="DENSITY"),
                Slot(description="B pure density", type="DENSITY"),
                Slot(description="C pure density", type="DENSITY"),
            ),
            (
                Slot(description="Impurity concentration",
                     type="CONCENTRATION"),
                Slot(description="Impurity concentration gradient",
                     type="CONCENTRATION_GRADIENT")
            )
        )


def _analytical_solution(A0, B0, P0, S0, C0, k_ps, t):
    a = _alpha(A0, B0, np.sum(k_ps), t)
    A = A0 - a
    B = B0 - a
    P = P0 + k_ps[0] / np.sum(k_ps) * a
    S = S0 + k_ps[1] / np.sum(k_ps) * a
    C = C0
    return np.array([A, B, P, S, C])


def _sum1(A0, B0, k, t, n=5):
    exponent = np.arange(1, n + 1, 1)
    denominator = factorial(exponent)
    result = ((B0 - A0) ** (exponent - 1)) * (k * t) ** exponent
    return np.sum(result / denominator)


def _sum2(A0, B0, k, t, n=5):
    exponent = np.arange(2, n + 1, 1)
    denominator = factorial(exponent) / (exponent - 1)
    result = ((B0 - A0) ** (exponent - 2)) * (k * t) ** exponent
    return np.sum(result / denominator)


def _sum3(A0, B0, k, t, n=5):
    exponent = np.arange(1, n + 1, 1)
    denominator = factorial(exponent) / exponent
    result = ((B0 - A0) ** (exponent - 1)) * (k * t) ** (exponent - 1)
    return np.sum(result / denominator)


def _alpha(A0, B0, k, t):
    epsilon = np.abs((A0 - B0) * k * t)
    if epsilon > 8e-2:
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
    if epsilon > 8e-2:
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
    if epsilon > 8e-2:
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
    if epsilon > 8e-2:
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
    if epsilon > 8e-2:
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


def _calc_k(T, M):
    M_v, M_delta_H = M
    R = 8.3144598e-3
    k_ps = M_v * np.exp(-M_delta_H / (R * T))
    return k_ps


def _run(X0, M):
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
    dkdT = 1 / (R * X0[5])**2 * np.sum(k_ps * M[0])
    grad_x_X_mat[:, 5] = dkdT * grad_x_X_mat[:, 5]
    return X_mat, grad_x_X_mat
