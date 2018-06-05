from force_bdss.api import BaseDataSource, DataValue, Slot


class ArrheniusParameters(BaseDataSource):
    def run(self, model, parameters):
        return [
            DataValue(type="ARRHENIUS_NU", value=model.nu_main_reaction),
            DataValue(type="ARRHENIUS_DELTA_H",
                      value=model.delta_H_main_reaction),
            DataValue(type="ARRHENIUS_NU", value=model.nu_secondary_reaction),
            DataValue(type="ARRHENIUS_DELTA_H",
                      value=model.delta_H_secondary_reaction)
        ]

    def slots(self, model):
        return (
            (
            ),
            (
                Slot(type="ARRHENIUS_NU"),
                Slot(type="ARRHENIUS_DELTA_H"),
                Slot(type="ARRHENIUS_NU"),
                Slot(type="ARRHENIUS_DELTA_H"),
            )
        )
