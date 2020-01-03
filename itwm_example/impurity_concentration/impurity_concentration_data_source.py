from jax.config import config
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

from force_bdss.api import DataValue, Slot, BaseDataSource

config.update("jax_enable_x64", True)


class ImpurityConcentrationDataSource(BaseDataSource):
    def run(self, model, parameters):

        input_data = [p.value for p in parameters]

        gradient = grad(objective)

        objective_value = np.asarray(objective(input_data), dtype=np.float64)
        gradient_value = np.asarray(gradient(input_data), dtype=np.float64)

        return [
            DataValue(value=objective_value, type="CONCENTRATION"),
            DataValue(value=gradient_value, type="CONCENTRATION_GRADIENT"),
        ]

    def slots(self, model):
        return (
            (
                Slot(description="V_A_tilde", type="VOLUME"),
                Slot(description="C_e concentration", type="CONCENTRATION"),
                Slot(description="Temperature", type="TEMPERATURE"),
                Slot(description="Reaction time", type="TIME"),
                Slot(
                    description="Arrhenius nu main reaction",
                    type="ARRHENIUS_NU",
                ),
                Slot(
                    description="Arrhenius delta H main reaction",
                    type="ARRHENIUS_DELTA_H",
                ),
                Slot(
                    description="Arrhenius nu secondary reaction",
                    type="ARRHENIUS_NU",
                ),
                Slot(
                    description="Arrhenius delta H secondary reaction",
                    type="ARRHENIUS_DELTA_H",
                ),
                Slot(description="Reactor volume", type="VOLUME"),
                Slot(description="A pure density", type="DENSITY"),
                Slot(description="B pure density", type="DENSITY"),
                Slot(description="C pure density", type="DENSITY"),
            ),
            (
                Slot(
                    description="Impurity concentration", type="CONCENTRATION"
                ),
                Slot(
                    description="Impurity concentration gradient",
                    type="CONCENTRATION_GRADIENT",
                ),
            ),
        )


def objective(inputs):
    transformed_inputs = preliminary_transformation(inputs)

    result_vector = analytical_solution(transformed_inputs)

    return jnp.dot(jnp.array([1.0, 1.0, 0.0, 1.0, 1.0]), result_vector)


@jit
def preliminary_transformation(inputs):
    v_a_tilde = inputs[0]
    c_conc_e = inputs[1]
    temperature = inputs[2]
    reaction_time = inputs[3]
    arrhenius_nu_main_reaction = inputs[4]
    arrhenius_delta_h_main_reaction = inputs[5]
    arrhenius_nu_secondary_reaction = inputs[6]
    arrhenius_delta_h_secondary_reaction = inputs[7]
    reactor_volume = inputs[8]
    density_a = inputs[9]
    density_b = inputs[10]
    density_c = inputs[11]

    constant_r = 8.3144598e-3
    k_ps_1 = arrhenius_nu_main_reaction * jnp.exp(
        -arrhenius_delta_h_main_reaction / constant_r / temperature
    )
    k_ps_2 = arrhenius_nu_secondary_reaction * jnp.exp(
        -arrhenius_delta_h_secondary_reaction / constant_r / temperature
    )

    return jnp.array(
        [
            density_a
            * (1 - c_conc_e / density_c)
            * v_a_tilde
            / reactor_volume,
            density_b * (reactor_volume - v_a_tilde) / reactor_volume,
            0.0,
            0.0,
            c_conc_e * v_a_tilde / reactor_volume,
            k_ps_1,
            k_ps_2,
            reaction_time,
        ]
    )


def analytical_solution(input):
    a = concentration_alpha(input[0], input[1], input[5] + input[6], input[7])
    update = jnp.array(
        [
            -a,
            -a,
            input[5] / (input[5] + input[6]) * a,
            input[6] / (input[5] + input[6]) * a,
            0,
        ]
    )
    return input[:5] + update


def concentration_alpha(concentration_a, concentration_b, k, t):
    epsilon = jnp.abs((concentration_a - concentration_b) * k * t)
    if epsilon > 8.0e-2:
        multiplier = jnp.exp((concentration_b - concentration_a) * k * t)
        result = (
            concentration_a
            * concentration_b
            * (multiplier - 1.0)
            / (concentration_b * multiplier - concentration_a)
        )
    else:
        sum1ab = alpha_internal_sum(concentration_a, concentration_b, k, t)
        sum1ba = alpha_internal_sum(concentration_b, concentration_a, k, t)
        result = (concentration_a * concentration_b * sum1ab) / (
            1.0 + (concentration_b * sum1ab)
        )
        result += (concentration_a * concentration_b * sum1ba) / (
            1.0 + (concentration_a * sum1ba)
        )
        result /= 2.0
    return result


@jit
def alpha_internal_sum(concentration_a, concentration_b, k, t):
    exponent = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    denominator = jnp.array([1.0, 2.0, 6.0, 24.0, 120.0])
    result = ((concentration_b - concentration_a) ** (exponent - 1)) * (
        k * t
    ) ** exponent
    return jnp.sum(result / denominator)
