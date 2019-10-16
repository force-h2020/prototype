import testfixtures
import unittest
from unittest import mock

from force_bdss.api import BaseMCOFactory, Workflow

from itwm_example.mco.mco_model import MCOModel
from itwm_example.mco.subprocess_workflow_evaluator import (
    SubprocessWorkflowEvaluator
)


class TestSubprocessWorkflowEvaluator(unittest.TestCase):

    def setUp(self):

        self.evaluator = SubprocessWorkflowEvaluator(
            workflow=Workflow(),
            workflow_filepath="test_probe.json"
        )
        self.mock_process = mock.Mock()
        self.mock_process.communicate = mock.Mock(
            return_value=(b"2", b"1 0")
        )

    def test___call_subprocess(self):

        # Test simple bash command
        stdout = self.evaluator._call_subprocess(
            ['uniq'], ['Hello', 'World']
        )
        self.assertEqual(
            'Hello World', stdout.decode("utf-8").strip()
        )

    def test__subprocess_solve(self):
        factory = mock.Mock(spec=BaseMCOFactory)
        self.evaluator.workflow.mco = MCOModel(factory)

        with mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = self.mock_process
            kpi_results = self.evaluator._subprocess_evaluate([1.0])

        self.assertEqual(1, len(kpi_results))

    def test_solve_error_mco_communicator(self):

        def mock_subprocess_evaluate(self, *args):
            raise Exception

        factory = mock.Mock(spec=BaseMCOFactory)
        self.evaluator.workflow.mco = MCOModel(factory)

        with mock.patch('itwm_example.mco.subprocess_workflow_evaluator'
                        '.SubprocessWorkflowEvaluator._subprocess_evaluate',
                        side_effect=mock_subprocess_evaluate):
            with testfixtures.LogCapture() as log:
                with self.assertRaises(RuntimeError):
                    self.evaluator.evaluate([1.0])
                log.check(
                    ('itwm_example.mco.subprocess_workflow_evaluator',
                     'ERROR',
                     'SubprocessWorkflowEvaluator failed to run. '
                     'This is likely due to an error '
                     'in the BaseMCOCommunicator assigned to <class '
                     "'force_bdss.mco.base_mco_factory.BaseMCOFactory'>.")
                )