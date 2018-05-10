from force_bdss.api import BaseDataSource, DataValue, Slot


class ProductionCostDataSource(BaseDataSource):
    """Defines a data source that returns the production
    cost of the heat required in the process.
    """

    def run(self, model, parameters):
        temperature = parameters[0].value
        reaction_time = parameters[1].value

        cost = reaction_time * (temperature - 290)**2 * model.W
        cost_gradient = [
                reaction_time * (2 * temperature - 2 * 290) * model.W,
                (temperature - 290)**2 * self.W
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
