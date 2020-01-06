from unittest import TestCase

from force_bdss.api import BaseCSVWriterModel, DataValue, MCOProgressEvent

from itwm_example.csv_writer.csv_writer import CSVWriter, CSVWriterFactory
from itwm_example.example_plugin import ExamplePlugin
from itwm_example.mco.driver_enents import ITWMMCOStartEvent


class TestCSVWriter(TestCase):
    def setUp(self):
        self.plugin = ExamplePlugin()
        self.factory = self.plugin.notification_listener_factories[0]
        self.notification_listener = self.factory.create_listener()
        self.model = self.factory.create_model()

        self.notification_listener.initialize(self.model)

        self.parameters = [
            DataValue(name="p1", value=1.0),
            DataValue(name="p2", value=5.0),
        ]
        self.kpis = [
            DataValue(name="kpi1", value=5.7),
            DataValue(name="kpi2", value=10),
        ]

    def test_factory(self):
        self.assertEqual("csv_writer", self.factory.get_identifier())
        self.assertEqual("CSV Writer", self.factory.get_name())
        self.assertIs(self.factory.listener_class, CSVWriter)
        self.assertIs(self.factory.model_class, BaseCSVWriterModel)
        self.assertIsInstance(self.factory, CSVWriterFactory)

    def test_parse_progress_event(self):
        event = MCOProgressEvent(
            optimal_point=self.parameters, optimal_kpis=self.kpis
        )
        self.assertListEqual(
            [1.0, 5.0, 5.7, 10],
            self.notification_listener.parse_progress_event(event),
        )

        event = MCOProgressEvent(
            optimal_point=self.parameters,
            optimal_kpis=self.kpis,
            weights=[1, 2, 3],
        )
        self.assertListEqual(
            [1.0, 5.0, 5.7, 10, 1, 2, 3],
            self.notification_listener.parse_progress_event(event),
        )

    def test_parse_start_event(self):
        event = ITWMMCOStartEvent(
            parameter_names=[p.name for p in self.parameters],
            kpi_names=[k.name for k in self.kpis],
        )
        self.assertListEqual(
            ["p1", "p2", "kpi1", "kpi2", "kpi1 weight", "kpi2 weight"],
            self.notification_listener.parse_start_event(event),
        )
