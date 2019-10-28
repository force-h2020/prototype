from itwm_example.tests.base_test_classes.base_test_data_source \
    import TemplateTestDataSource


class TestFixedValueDataSource(TemplateTestDataSource):
    def setUp(self):
        self._data_source_index = 0
        super().setUp()

        self.test_inputs = [
            []
        ]
        self.test_outputs = [
            [
                self.model.value
            ]
        ]
