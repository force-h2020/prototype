from jax.config import config
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

from force_bdss.api import DataValue, Slot, BaseDataSource

config.update("jax_enable_x64", True)


class ImpurityConcentrationDataSource(BaseDataSource):
    def run(self, model, parameters):

        input_data = [p.value for p in parameters]

        objective_value = np.asarray(
            self.objective(input_data), dtype=np.float64
        )
        gradient_value = np.asarray(
            self.gradient(input_data), dtype=np.float64
        )

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

    def objective(self, inputs):
        transformed_inputs = preliminary_transformation(inputs)

        result_vector = analytical_solution(transformed_inputs)

        return jnp.dot(jnp.array([1.0, 1.0, 0.0, 1.0, 1.0]), result_vector)

    def gradient(self, inputs):
        return grad(self.objective)(inputs)


@jit
def preliminary_transformation(inputs):
    """ The preliminary transformation of the input chemical parameters.
    The transformations are described in section 3.1. 'The reaction
    model', equations (3.1, 3.7).
    """
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

    # Volume concentrations in the reactor (see equations 3.1)
    initial_conc_a = (
        density_a * (1 - c_conc_e / density_c) * v_a_tilde / reactor_volume
    )
    initial_conc_b = density_b * (reactor_volume - v_a_tilde) / reactor_volume
    initial_conc_c = c_conc_e * v_a_tilde / reactor_volume
    initial_conc_p = 0.0
    initial_conc_s = 0.0

    # Materials related (see equations 3.7)
    constant_r = 8.3144598e-3
    k_ps_1 = arrhenius_nu_main_reaction * jnp.exp(
        -arrhenius_delta_h_main_reaction / constant_r / temperature
    )
    k_ps_2 = arrhenius_nu_secondary_reaction * jnp.exp(
        -arrhenius_delta_h_secondary_reaction / constant_r / temperature
    )

    return jnp.array(
        [
            initial_conc_a,
            initial_conc_b,
            initial_conc_p,
            initial_conc_s,
            initial_conc_c,
            k_ps_1,
            k_ps_2,
            reaction_time,
        ]
    )


def analytical_solution(input):
    """ Analytical solution of the kinetic reaction model (see section 3.1,
    equation (3.6)). The solution is provided by equation (3.9).
    The analytical solution updates the initial concentrations of the
    chemicals at t = 0 by the values, defined by alpha(t).
    """
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
    """ Calculates the concentrations change (denoted by alpha(t).
    The analytical solution is provided by the equation (3.9) and
    (3.10). The later one is a continuous approximation of the actual
    analytical solution.
    """
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
    """ Auxiliary method to calculate the sum introduced in equation (3.10).
    """
    exponent = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    denominator = jnp.array([1.0, 2.0, 6.0, 24.0, 120.0])
    result = ((concentration_b - concentration_a) ** (exponent - 1)) * (
        k * t
    ) ** exponent
    return jnp.sum(result / denominator)
