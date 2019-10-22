import logging
import sys

import numpy as np

from traits.api import Type

from force_bdss.api import BaseMCO, DataValue

from .subprocess_workflow_evaluator import SubprocessWorkflowEvaluator
from .scaling_tools.kpi_scaling import sen_scaling_method
from .optimizers.optimizers import IOptimizer, WeightedOptimizer
from .space_sampling.space_samplers import (
    UniformSpaceSampler,
    DirichletSpaceSampler,
)

log = logging.getLogger(__name__)


class MCO(BaseMCO):

    optimizer = Type(IOptimizer)

    scaling_method = staticmethod(sen_scaling_method)

    def _optimizer_default(self):
        return WeightedOptimizer

    def get_scaling_factors(self, optimizer, kpis, scaling_method=None):
        """Calculates scaling factors for KPIs, defined in MCO.
        Scaling factors are calculated (as required) by the provided scaling
        method. In general, this provides normalization values for the possible
        range of each KPI..
        Performs scaling for all KPIs that have `auto_scale == True`.
        Otherwise, keeps the default scale factor.

        Parameters
        ----------
        optimizer: IOptimizer instance
            Instance that provides optimization functionality
        kpis: List[KPISpecification]
            List of KPI objects to scale
        scaling_method: callable
            A method to scale KPI weights. Default set to the Sen's
            "Multi-Objective Programming Method"
        """
        if scaling_method is None:
            scaling_method = self.scaling_method

        #: Get default scaling weights for each KPI variable
        default_scaling_factors = np.array([kpi.scale_factor for kpi in kpis])

        #: Apply a wrapper for the evaluator weights assignment and
        #: call of the .optimize method.
        #: Then, calculate scaling factors defined by the `scaling_method`
        scaling_factors = scaling_method(len(kpis), optimizer.optimize)

        #: Apply the scaling factors where necessary
        auto_scales = [kpi.auto_scale for kpi in kpis]
        default_scaling_factors[auto_scales] = scaling_factors[auto_scales]

        log.info(
            "Using KPI scaling factors: {}".format(default_scaling_factors)
        )

        return default_scaling_factors.tolist()

    @staticmethod
    def _space_search_distribution(model, **kwargs):
        """ Generates space search distribution object, based on
        the user settings of the `space_search_strategy` trait."""

        if model.space_search_mode == "Uniform":
            distribution = UniformSpaceSampler
        elif model.space_search_mode == "Dirichlet":
            distribution = DirichletSpaceSampler
        else:
            raise NotImplementedError
        return distribution(len(model.kpis), model.num_points, **kwargs)

    def weights_samples(self, model, **kwargs):
        """ Generates necessary number of search space sample points
        from the internal search strategy."""
        return self._space_search_distribution(
            model, **kwargs
        ).generate_space_sample()

    def run(self, evaluator):

        model = evaluator.mco_model
        kpis = model.kpis

        if model.evaluation_mode == "Subprocess":
            # Here we create an instance of our WorkflowEvaluator subclass
            # that allows for evaluation of a state in the workflow via calling
            # force_bdss on a new subprocess running in 'evaluate' mode.
            # Note: a BaseMCOCommunicator must be present to pass in parameter
            # values and returning the KPI for a force_bdss run in 'evaluate'
            # mode
            evaluator = SubprocessWorkflowEvaluator(
                workflow=evaluator.workflow,
                workflow_filepath=evaluator.workflow_filepath,
                executable_path=sys.argv[0],
            )

        optimizer = self.optimizer(evaluator, model.parameters)

        #: Get scaling factors and non-zero weight combinations for each KPI
        scaling_factors = self.get_scaling_factors(optimizer, kpis)

        for weights in self.weights_samples(model):

            log.info("Doing MCO run with weights: {}".format(weights))

            scaled_weights = [
                weight * scale
                for weight, scale in zip(weights, scaling_factors)
            ]

            optimal_point, optimal_kpis = optimizer.optimize(scaled_weights)

            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            self.notify_new_point(
                [DataValue(value=v) for v in optimal_point],
                [DataValue(value=v) for v in optimal_kpis],
                scaled_weights,
            )
