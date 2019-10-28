from itwm_example.tests.base_test_classes.base_test_data_source \
    import TemplateTestGradientDataSource


class TestMaterialCostDataSource(TemplateTestGradientDataSource):
    def setUp(self):
        self._data_source_index = 3
        super().setUp()
        self.test_inputs = [
            [1., 1., 1., 1.]
        ]
        self.test_outputs = [
            1.,
        ]
