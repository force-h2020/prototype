#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from itwm_example.tests.template_test_classes.template_test_data_source \
    import TemplateTestDataSource


class TestArrheniusParameters(TemplateTestDataSource):
    _data_source_index = 2
    test_inputs = [[]]

    @property
    def test_outputs(self):
        return [
            [
                self.model.nu_main_reaction,
                self.model.delta_H_main_reaction,
                self.model.nu_secondary_reaction,
                self.model.delta_H_secondary_reaction,
            ]
        ]
