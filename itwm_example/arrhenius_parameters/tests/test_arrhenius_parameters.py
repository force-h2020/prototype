from itwm_example.tests.base_test_classes.base_test_data_source \
    import TemplateTestDataSource


class TestArrheniusParameters(TemplateTestDataSource):
    def setUp(self):
        self._data_source_index = 2
        super().setUp()

        self.test_case_values = [
            []
        ]
        self.test_case_objectives = [
            [
                self.model.nu_main_reaction,
                self.model.delta_H_main_reaction,
                self.model.nu_secondary_reaction,
                self.model.delta_H_secondary_reaction
            ]
        ]

    def test_output_slots(self):
        super().base_test_output_slots(self.test_case_values[0])
