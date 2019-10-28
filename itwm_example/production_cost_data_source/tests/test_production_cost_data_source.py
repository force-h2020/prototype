from itwm_example.tests.base_test_classes.base_test_data_source \
    import TemplateTestGradientDataSource


class TestProductionCostDataSource(TemplateTestGradientDataSource):
    def setUp(self):
        self._data_source_index = 1
        super().setUp()

        self.test_case_values = [
            [290, 1],
            [291, 1]
        ]
        self.test_case_objectives = [
            0.,
            self.model.W
        ]

    def test_gradient_type(self):
        super().base_test_gradient_type()

    def test_param_to_gradient(self):
        super().base_test_param_to_gradient()

    def test_gradient_convergence(self):
        super().base_test_gradient_convergence()

    def test_output_slots(self):
        super().base_test_output_slots(self.test_case_values[0])
