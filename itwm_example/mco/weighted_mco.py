#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

import logging

from force_bdss.api import BaseMCO, DataValue

from force_bdss.mco.optimizer_engines.weighted_optimizer_engine import (
    WeightedOptimizerEngine
)
from force_bdss.mco.optimizers.scipy_optimizer import ScipyOptimizer


log = logging.getLogger(__name__)


class WeightedMCO(BaseMCO):
    def run(self, evaluator):

        model = evaluator.mco_model

        optim = ScipyOptimizer(algorithms=model.algorithms)

        optimizer = WeightedOptimizerEngine(
            kpis=model.kpis,
            parameters=model.parameters,
            num_points=model.num_points,
            space_search_mode=model.space_search_mode,
            single_point_evaluator=evaluator,
            verbose_run=model.verbose_run,
            optimizer=optim,
        )

        for (
            optimal_point,
            optimal_kpis,
            scaled_weights,
        ) in optimizer.optimize():

            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            evaluator.mco_model.notify_progress_event(
                [DataValue(value=v) for v in optimal_point],
                [DataValue(value=v) for v in optimal_kpis],
                weights=scaled_weights,
            )
