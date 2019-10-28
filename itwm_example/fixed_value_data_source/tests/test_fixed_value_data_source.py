from itwm_example.tests.base_test_classes.base_test_data_source \
    import TemplateTestDataSource


class TestFixedValueDataSource(TemplateTestDataSource):
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

    def test_output_slots(self):
        super().base_test_output_slots(self.test_case_values[0])
