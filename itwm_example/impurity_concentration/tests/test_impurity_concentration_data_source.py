from itwm_example.tests.probe_classes.data_source import ProbeDataSource


class TestImpurityConcentrationDataSource(ProbeDataSource):
    def setUp(self):
        self._data_source_index = 4
        super().setUp()

    def test_basic_evaluation(self):
        values = [
            0.5,
            0.1,
            335.0,
            360.0,
            0.02,
            1.5,
            0.02,
            12.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        data_values = self.convert_to_data_values(values, self.input_slots)

        res = self.data_source.run(self.model, data_values)

        self.assertAlmostEqual(res[0].value, 0.372027129767, 6)
