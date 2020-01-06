from unittest import TestCase

from itwm_example.mco.driver_enents import ITWMMCOStartEvent


class TestProgressEvents(TestCase):
    def test_getstate_start_event(self):
        event = ITWMMCOStartEvent(
            parameter_names=["p1", "p2", "p3"], kpi_names=["kpi"]
        )
        self.assertDictEqual(
            event.__getstate__(),
            {"parameter_names": ["p1", "p2", "p3"], "kpi_names": ["kpi"]},
        )
        self.assertEqual(
            ["p1", "p2", "p3", "kpi", "kpi weight"], event.serialize()
        )
