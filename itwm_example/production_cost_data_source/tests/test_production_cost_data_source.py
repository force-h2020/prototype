from itwm_example.tests.template_test_classes.template_test_data_source \
    import TemplateTestGradientDataSource


class TestProductionCostDataSource(TemplateTestGradientDataSource):
    _data_source_index = 1
    test_inputs = [[290, 1], [291, 1]]

    @property
    def test_outputs(self):
        return [0.0, self.model.W]
