#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from force_bdss.api import BaseDataSource, DataValue, Slot


class MaterialCostDataSource(BaseDataSource):
    """Defines a data source that returns the cost
    of materials to perform the reaction.
    """

    def run(self, model, parameters):
        V_a = parameters[0].value
        C_e = parameters[1].value
        V_r = parameters[2].value
        rho_C = parameters[3].value

        tot_cost_A = V_a * ((1 - C_e / rho_C) * model.const_A +
                            model.const_C * rho_C / C_e)
        tot_cost_B = (V_r - V_a) * model.cost_B
        cost = tot_cost_A + tot_cost_B

        grad_cost = [
            ((1 - C_e / rho_C) * model.const_A
             + model.const_C * rho_C / C_e) - model.cost_B,
            - V_a * model.const_A / rho_C - model.const_C * rho_C / C_e**2,
            model.cost_B,
            V_a * (model.const_A * C_e / rho_C ** 2.0 + model.const_C / C_e),
        ]
        return [
            DataValue(
                type="COST",
                value=cost
            ),
            DataValue(
                type="COST_GRADIENT",
                value=grad_cost
            ),
        ]

    def slots(self, model):
        return (
            (
                Slot(description="Volume of contaminated A", type="VOLUME"),
                Slot(description="Concentration of contaminant",
                     type="CONCENTRATION"),
                Slot(description="Reactor Volume", type="VOLUME"),
                Slot(description="Density of contaminant", type="DENSITY"),
            ),
            (
                Slot(description="Material cost", type="COST"),
                Slot(description="Material cost gradient",
                     type="COST_GRADIENT"),
            )
        )
