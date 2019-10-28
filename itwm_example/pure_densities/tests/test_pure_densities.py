from itwm_example.tests.base_test_classes.base_test_data_source \
    import TemplateTestDataSource


class TestPureDensities(TemplateTestDataSource):
    def setUp(self):
        self._data_source_index = 5
        super().setUp()

        self.test_inputs = [
            []
        ]
        self.test_outputs = [
            [
                self.model.a_pure_density,
                self.model.b_pure_density,
                self.model.c_pure_density
            ]
        ]
