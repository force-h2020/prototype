#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from itwm_example.tests.template_test_classes.template_test_data_source \
    import TemplateTestGradientDataSource


class TestImpurityConcentrationDataSource(TemplateTestGradientDataSource):
    _data_source_index = 4
    test_inputs = [
        [0.5, 0.1, 335.0, 360.0, 0.02, 1.5, 0.02, 12.0, 1.0, 1.0, 1.0, 1.0]
    ]
    test_outputs = [0.372027129767]

    def test_slots_signature(self):
        self.assertEqual(12, len(self.input_slots))
        self.assertEqual(2, len(self.output_slots))

        self.assertEqual("CONCENTRATION", self.output_slots[0].type)

    def test_param_to_gradient(self):
        """ This test requires refactoring this DataSource"""
        pass

    def test_gradient_convergence(self):
        """ This test requires refactoring this DataSource"""
        pass
