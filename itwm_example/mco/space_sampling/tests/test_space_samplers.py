import unittest

from itwm_example.mco.space_sampling.space_samplers import (
    SpaceSampler,
    DirichletSpaceSampler,
    UniformSpaceSampler,
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


class BaseTestSampler(unittest.TestCase):
    distribution = SpaceSampler

    def generate_values(self, *args, **kwargs):
        return self.distribution(
            *args,
            **kwargs
        ).generate_space_sample(**kwargs)


class TestDirichletSpaceSampler(BaseTestSampler):
    distribution = DirichletSpaceSampler

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


class TestUniformSpaceSampler(BaseTestSampler):
    distribution = UniformSpaceSampler

    def setUp(self):
        self.sampler = self.distribution(
            dimension=3,
            resolution=5
        )

    def test_space_sample(self):
        self.assertEqual(
            list(self.generate_values(3, 5)),
            [
                [1.0, 0.0, 0.0],
                [0.75, 0.25, 0.0],
                [0.75, 0.0, 0.25],
                [0.50, 0.50, 0.0],
                [0.50, 0.25, 0.25],
                [0.50, 0.0, 0.50],
                [0.25, 0.75, 0.0],
                [0.25, 0.50, 0.25],
                [0.25, 0.25, 0.50],
                [0.25, 0.0, 0.75],
                [0.0, 1.0, 0.0],
                [0.0, 0.75, 0.25],
                [0.0, 0.50, 0.50],
                [0.0, 0.25, 0.75],
                [0.0, 0.0, 1.0],
            ]
        )

        self.assertEqual(
            list(self.generate_values(1, 5)),
            [[1.0]]
        )
        self.assertEqual(list(self.generate_values(2, 5)), [
            [1.0, 0.0],
            [0.75, 0.25],
            [0.50, 0.50],
            [0.25, 0.75],
            [0.0, 1.0],
        ])

        self.assertEqual(
            list(self.generate_values(3, 9)),
            [[1.0, 0.0, 0.0], [0.875, 0.125, 0.0], [0.875, 0.0, 0.125],
             [0.75, 0.25, 0.0], [0.75, 0.125, 0.125], [0.75, 0.0, 0.25],
             [0.625, 0.375, 0.0], [0.625, 0.25, 0.125], [0.625, 0.125, 0.25],
             [0.625, 0.0, 0.375], [0.5, 0.5, 0.0], [0.5, 0.375, 0.125],
             [0.5, 0.25, 0.25], [0.5, 0.125, 0.375], [0.5, 0.0, 0.5],
             [0.375, 0.625, 0.0], [0.375, 0.5, 0.125], [0.375, 0.375, 0.25],
             [0.375, 0.25, 0.375], [0.375, 0.125, 0.5], [0.375, 0.0, 0.625],
             [0.25, 0.75, 0.0], [0.25, 0.625, 0.125], [0.25, 0.5, 0.25],
             [0.25, 0.375, 0.375], [0.25, 0.25, 0.5], [0.25, 0.125, 0.625],
             [0.25, 0.0, 0.75], [0.125, 0.875, 0.0], [0.125, 0.75, 0.125],
             [0.125, 0.625, 0.25], [0.125, 0.5, 0.375], [0.125, 0.375, 0.5],
             [0.125, 0.25, 0.625], [0.125, 0.125, 0.75], [0.125, 0.0, 0.875],
             [0.0, 1.0, 0.0], [0.0, 0.875, 0.125], [0.0, 0.75, 0.25],
             [0.0, 0.625, 0.375], [0.0, 0.5, 0.5], [0.0, 0.375, 0.625],
             [0.0, 0.25, 0.75], [0.0, 0.125, 0.875], [0.0, 0.0, 1.0]]
        )

        self.assertEqual(
            list(self.generate_values(2, 5, zero_values=False)),
            [
                [0.75, 0.25],
                [0.50, 0.50],
                [0.25, 0.75],
            ]
        )

        self.assertEqual(
            list(self.generate_values(3, 9, zero_values=False)),
            [[0.75, 0.125, 0.125],
             [0.625, 0.25, 0.125],
             [0.625, 0.125, 0.25],
             [0.5, 0.375, 0.125],
             [0.5, 0.25, 0.25],
             [0.5, 0.125, 0.375],
             [0.375, 0.5, 0.125],
             [0.375, 0.375, 0.25],
             [0.375, 0.25, 0.375],
             [0.375, 0.125, 0.5],
             [0.25, 0.625, 0.125],
             [0.25, 0.5, 0.25],
             [0.25, 0.375, 0.375],
             [0.25, 0.25, 0.5],
             [0.25, 0.125, 0.625],
             [0.125, 0.75, 0.125],
             [0.125, 0.625, 0.25],
             [0.125, 0.5, 0.375],
             [0.125, 0.375, 0.5],
             [0.125, 0.25, 0.625],
             [0.125, 0.125, 0.75]]
        )
