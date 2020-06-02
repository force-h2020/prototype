#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from unittest import TestCase

from force_bdss.api import (
    BaseCSVWriterModel,
    DataValue,
    BaseCSVWriter,
    WeightedMCOProgressEvent
)

from itwm_example.csv_writer.csv_writer import CSVWriterFactory
from itwm_example.itwm_example_plugin import ITWMExamplePlugin
from itwm_example.mco.driver_events import ITWMMCOStartEvent


class TestCSVWriter(TestCase):
    def setUp(self):
        self.plugin = ITWMExamplePlugin()
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
        self.assertIs(self.factory.listener_class, BaseCSVWriter)
        self.assertIs(self.factory.model_class, BaseCSVWriterModel)
        self.assertIsInstance(self.factory, CSVWriterFactory)

    def test_parse_progress_event(self):
        event = WeightedMCOProgressEvent(
            optimal_point=self.parameters,
            optimal_kpis=self.kpis
        )
        self.assertListEqual(
            [1.0, 5.0, 5.7, 0.5, 10, 0.5],
            self.notification_listener.parse_progress_event(event),
        )

        event = WeightedMCOProgressEvent(
            optimal_point=self.parameters,
            optimal_kpis=self.kpis,
            weights=[1, 2],
        )
        self.assertListEqual(
            [1.0, 5.0, 5.7, 1, 10, 2],
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
