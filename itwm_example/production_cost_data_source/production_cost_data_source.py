from force_bdss.api import BaseDataSource, DataValue, Slot


class ProductionCostDataSource(BaseDataSource):
    """Defines a data source that returns the production
    cost of the heat required in the process.
    """
    temperature_zero_kelvin = -270

    def run(self, model, parameters):
        temperature = parameters[0].value
        reaction_time = parameters[1].value

        temperature_celsius = self.kelvin_to_celsius(temperature)
        temperature_shifted = temperature_celsius - model.temperature_shift
        cost = (
                reaction_time
                * temperature_shifted**2
                * model.W
        )
        cost_gradient = [
                reaction_time * 2.0 * temperature_shifted * model.W,
                temperature_shifted**2 * model.W
        ]

        return [
            DataValue(
                type="COST",
                value=cost
            ),
            DataValue(
                type="COST_GRADIENT",
                value=cost_gradient
            )
        ]

    def slots(self, model):
        return (
            (
                Slot(
                    description="Temperature",
                    type="TEMPERATURE"
                ),
                Slot(
                    description="Reaction time",
                    type="TIME"
                ),
            ),
            (
                Slot(description="Cost of the process",
                     type="COST"),
                Slot(description="Gradient of the cost of the process",
                     type="COST_GRADIENT"),
            )
        )

    def kelvin_to_celsius(self, temperature):
        return temperature + self.temperature_zero_kelvin
