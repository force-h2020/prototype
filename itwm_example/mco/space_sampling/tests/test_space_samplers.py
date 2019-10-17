import unittest

from itwm_example.mco.space_sampling.space_samplers import (
    DirichletSpaceSampler,
    convert_samples_pp_to_samples_total
)


class TestSamplesConvert(unittest.TestCase):
    def test_convert(self):
        samples_total = convert_samples_pp_to_samples_total(
            space_dimension=3,
            nof_points=5
        )
        known_samples_total = 15
        self.assertEqual(samples_total, known_samples_total)


class TestDirichletSpaceSampler(unittest.TestCase):
    def setUp(self):
        self.dimensions = [3, 1, 5]
        self.alphas = [1, 0.5, 10]
        self.nof_points = [3, 10, 6]

    def generate_samplers(self):
        for dimension in self.dimensions:
            for alpha in self.alphas:
                yield DirichletSpaceSampler(
                    dimension=dimension,
                    alpha=alpha
                )

    def test__get_sample_point(self):
        for sampler in self.generate_samplers():
            self.assertAlmostEqual(
                sum(sampler._get_sample_point()),
                1.
            )

    def test_generate_space_sample(self):
        for sampler in self.generate_samplers():
            for nof_points in self.nof_points:

                space_sample = list(
                    sampler.generate_space_sample(
                        nof_points
                    )
                )
                self.assertEqual(
                    len(space_sample),
                    convert_samples_pp_to_samples_total(
                        sampler.dimension,
                        nof_points
                    )
                )

                for sample in space_sample:
                    self.assertAlmostEqual(
                        sum(sample),
                        1.
                    )
