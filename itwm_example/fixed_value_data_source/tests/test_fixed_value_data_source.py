from itwm_example.tests.base_test_classes.base_test_data_source \
    import BaseTestDataSource


class TestFixedValueDataSource(BaseTestDataSource):
    def setUp(self):
        self._data_source_index = 0
        super().setUp()

        self.test_case_values = [
            []
        ]
        self.test_case_objectives = [
            [
                self.model.value
            ]
        ]

    def test_basic_evaluation(self):
        super().base_test_basic_evaluation()

    def test_output_slots(self):
        super().base_test_output_slots(self.test_case_values[0])
