from itwm_example.tests.template_test_classes.template_test_data_source \
    import TemplateTestDataSource


class TestFixedValueDataSource(TemplateTestDataSource):
    _data_source_index = 0
    test_inputs = [[]]

    @property
    def test_outputs(self):
        return [[self.model.value]]
