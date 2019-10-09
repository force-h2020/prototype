from itwm_example.tests.probe_classes.data_source import ProbeDataSource


class TestImpurityConcentrationDataSource(ProbeDataSource):
    def setUp(self):
        self._data_source_index = 4
        super().setUp()
        self.test_case_values = [
            [
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
        ]

    def test_slots_signature(self):
        self.assertEqual(12, len(self.input_slots))
        self.assertEqual(2, len(self.output_slots))

        self.assertEqual("CONCENTRATION", self.output_slots[0].type)
        self.assertEqual(
            self.output_slots[0].type+"_GRADIENT",
            self.output_slots[1].type
        )

    def test_basic_evaluation(self):
        values = self.test_case_values[0]
        data_values = self.convert_to_data_values(values, self.input_slots)

        res = self.data_source.run(self.model, data_values)

        self.assertAlmostEqual(res[0].value, 0.372027129767, 6)
