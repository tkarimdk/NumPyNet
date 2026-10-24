import unittest

from numpynet.experiments import train_xor


class ExperimentTests(unittest.TestCase):
    def test_training_solves_xor_with_fixed_seed(self) -> None:
        report = train_xor()
        self.assertGreater(report.losses[0], report.losses[-1])
        self.assertEqual(report.accuracy, 1.0)
        self.assertLess(report.losses[-1], 0.05)

    def test_training_is_reproducible(self) -> None:
        first = train_xor(epochs=10, seed=3)
        second = train_xor(epochs=10, seed=3)
        self.assertEqual(first.losses, second.losses)


if __name__ == "__main__":
    unittest.main()
