from itwm_example.tests.probe_classes.data_source import BaseTestDataSource


class TestProductionCostDataSource(BaseTestDataSource):
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

    def test_basic_evaluation(self):
        super().base_test_basic_evaluation()

    def test_gradient_type(self):
        super().base_test_gradient_type()
