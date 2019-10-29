from itwm_example.tests.template_test_classes.template_test_data_source \
    import TemplateTestGradientDataSource


class TestMaterialCostDataSource(TemplateTestGradientDataSource):
    _data_source_index = 3
    test_inputs = [[1.0, 1.0, 1.0, 1.0]]
    test_outputs = [1.0]
