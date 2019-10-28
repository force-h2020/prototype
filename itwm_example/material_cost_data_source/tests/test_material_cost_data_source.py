from itwm_example.tests.base_test_classes.base_test_data_source import (
    TemplateTestGradientDataSource,
)


class TestMaterialCostDataSource(TemplateTestGradientDataSource):
    _data_source_index = 3
    test_inputs = [[1.0, 1.0, 1.0, 1.0]]
    test_outputs = [1.0]
