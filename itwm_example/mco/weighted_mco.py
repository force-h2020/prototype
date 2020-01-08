import logging

from force_bdss.api import BaseMCO, DataValue, WeightedOptimizerEngine


log = logging.getLogger(__name__)


class WeightedMCO(BaseMCO):
    def run(self, evaluator):

        model = evaluator.mco_model

        optimizer = WeightedOptimizerEngine(
            kpis=model.kpis,
            parameters=model.parameters,
            num_points=model.num_points,
            algorithms=model.algorithms,
            space_search_mode=model.space_search_mode,
            single_point_evaluator=evaluator,
            verbose_run=model.verbose_run,
        )

        for (
            optimal_point,
            optimal_kpis,
            scaled_weights,
        ) in optimizer.optimize():

            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            self.notify_new_point(
                [DataValue(value=v) for v in optimal_point],
                [DataValue(value=v) for v in optimal_kpis],
                scaled_weights,
            )
