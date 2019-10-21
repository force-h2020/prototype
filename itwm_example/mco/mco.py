import logging
import sys

import numpy as np

from traits.api import Type

from force_bdss.api import BaseMCO, DataValue

from .subprocess_workflow_evaluator import SubprocessWorkflowEvaluator
from .scaling_tools.kpi_scaling import sen_scaling_method
from .optimizers.optimizers import IOptimizer, WeightedOptimizer

log = logging.getLogger(__name__)


class MCO(BaseMCO):

    optimizer = Type(IOptimizer)

    def _optimizer_default(self):
        return WeightedOptimizer

    @staticmethod
    def optimization_wrapper(evaluator):
        def inner(weights):
            evaluator.weights = list(weights)
            return evaluator.optimize()

        return inner

    def get_scaling_factors(
        self, optimizer, kpis, parameters, scaling_method=None
    ):
        """Calculates scaling factors for KPIs, defined in MCO.
        Scaling factors are calculated (as required) by the provided scaling
        method. In general, this provides normalization values for the possible
        range of each KPI..
        Performs scaling for all KPIs that have `auto_scale == True`.
        Otherwise, keeps the default scale factor.

        Parameters
        ----------
        optimizer: IOptimizer
            Instance that provides optimization functionality
        kpis: List[KPISpecification]
            List of KPI objects to scale
        parameters: List[MCOParameters]
            MCO parameters, required by the optimizer
        scaling_method: callable
            A method to scale KPI weights. Default set to the Sen's
            "Multi-Objective Programming Method"
        """
        if scaling_method is None:
            scaling_method = sen_scaling_method

        evaluator = self.optimizer(optimizer, [1.0 for _ in kpis], parameters)

        #: Get default scaling weights for each KPI variable
        default_scaling_factors = np.array([kpi.scale_factor for kpi in kpis])

        #: Apply a wrapper for the evaluator weights assignment and
        #: call of the .optimize method.
        #: Then, calculate scaling factors defined by the `scaling_method`
        scaling_factors = scaling_method(
            len(evaluator.weights), self.optimization_wrapper(evaluator)
        )

        #: Apply the scaling factors where necessary
        auto_scales = [kpi.auto_scale for kpi in kpis]
        default_scaling_factors[auto_scales] = scaling_factors[auto_scales]

        log.info(
            "Using KPI scaling factors: {}".format(default_scaling_factors)
        )

        return default_scaling_factors.tolist()

    def run(self, evaluator):

        model = evaluator.mco_model
        parameters = model.parameters
        kpis = model.kpis

        if model.evaluation_mode == "Subprocess":
            # Here we create an instance of our WorkflowEvaluator subclass
            # that allows for evaluation of a state in the workflow via calling
            # force_bdss on a new subprocess running in 'evaluate' mode.
            # Note: a BaseMCOCommunicator must be present to pass in parameter
            # values and returning the KPI for a force_bdss run in 'evaluate'
            # mode
            single_point_evaluator = SubprocessWorkflowEvaluator(
                workflow=evaluator.workflow,
                workflow_filepath=evaluator.workflow_filepath,
                executable_path=sys.argv[0],
            )
        else:
            single_point_evaluator = evaluator

        #: Get scaling factors and non-zero weight combinations for each KPI
        scaling_factors = self.get_scaling_factors(
            single_point_evaluator, kpis, parameters
        )

        optimizer = self.optimizer(
            single_point_evaluator, [1.0 for _ in kpis], parameters
        )
        optimizer = self.optimization_wrapper(optimizer)

        for weights in model.weights_samples(with_zero_values=False):

            log.info("Doing MCO run with weights: {}".format(weights))

            generator = zip(weights, scaling_factors)
            scaled_weights = [weight * scale for weight, scale in generator]

            optimal_point, optimal_kpis = optimizer(scaled_weights)

            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            self.notify_new_point(
                [DataValue(value=v) for v in optimal_point],
                [DataValue(value=v) for v in optimal_kpis],
                scaled_weights,
            )
