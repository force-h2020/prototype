from force_bdss.api import BaseMCOModel, PositiveInt


class MCOModel(BaseMCOModel):
    num_points = PositiveInt(7)
