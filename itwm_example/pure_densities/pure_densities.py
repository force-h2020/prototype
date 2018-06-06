from force_bdss.api import BaseDataSource, DataValue, Slot


class PureDensities(BaseDataSource):
    def run(self, model, parameters):
        return [
            DataValue(type="DENSITY", value=model.a_pure_density),
            DataValue(type="DENSITY", value=model.b_pure_density),
            DataValue(type="DENSITY", value=model.c_pure_density)
        ]

    def slots(self, model):
        return (
            (
            ),
            (
                Slot(type="DENSITY", description="A pure density"),
                Slot(type="DENSITY", description="B pure density"),
                Slot(type="DENSITY", description="C pure density"),
            )
        )
