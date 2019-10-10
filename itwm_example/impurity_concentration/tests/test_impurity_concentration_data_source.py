from itwm_example.tests.probe_classes.data_source import BaseTestDataSource


class TestImpurityConcentrationDataSource(BaseTestDataSource):
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
        self.test_case_objectives = [0.372027129767]

    def test_slots_signature(self):
        self.assertEqual(12, len(self.input_slots))
        self.assertEqual(2, len(self.output_slots))

        self.assertEqual("CONCENTRATION", self.output_slots[0].type)

    def test_gradient_type(self):
        super().base_test_gradient_type()

    def test_basic_evaluation(self):
        super().base_test_basic_evaluation()
