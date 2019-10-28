from itwm_example.tests.base_test_classes.base_test_data_source \
    import TemplateTestGradientDataSource


class TestProductionCostDataSource(TemplateTestGradientDataSource):
    def setUp(self):
        self._data_source_index = 1
        super().setUp()

        self.test_inputs = [
            [290, 1],
            [291, 1]
        ]
        self.test_outputs = [
            0.,
            self.model.W
        ]
