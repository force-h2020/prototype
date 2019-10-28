from itwm_example.tests.base_test_classes.base_test_data_source import (
    TemplateTestDataSource,
)


class TestPureDensities(TemplateTestDataSource):
    _data_source_index = 5
    test_inputs = [[]]

    def setUp(self):
        super().setUp()

        self.test_outputs = [
            [
                self.model.a_pure_density,
                self.model.b_pure_density,
                self.model.c_pure_density,
            ]
        ]
