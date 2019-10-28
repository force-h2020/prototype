from itwm_example.tests.base_test_classes.base_test_data_source \
    import TemplateTestDataSource


class TestArrheniusParameters(TemplateTestDataSource):
    def setUp(self):
        self._data_source_index = 2
        super().setUp()

        self.test_inputs = [
            []
        ]
        self.test_outputs = [
            [
                self.model.nu_main_reaction,
                self.model.delta_H_main_reaction,
                self.model.nu_secondary_reaction,
                self.model.delta_H_secondary_reaction
            ]
        ]
