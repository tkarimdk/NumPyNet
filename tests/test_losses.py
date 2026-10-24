import unittest

import numpy as np

from numpynet.losses import BinaryCrossEntropy


class LossTests(unittest.TestCase):
    def test_loss_for_good_predictions_is_small(self) -> None:
        loss = BinaryCrossEntropy().forward(
            np.array([[0.01], [0.99]]), np.array([[0.0], [1.0]])
        )
        self.assertLess(loss, 0.02)

    def test_loss_gradient_matches_finite_difference(self) -> None:
        criterion = BinaryCrossEntropy()
        target = np.array([[1.0], [0.0]])
        probabilities = np.array([[0.8], [0.3]])
        criterion.forward(probabilities, target)
        analytic = criterion.backward()
        delta = 1e-6
        for index in np.ndindex(probabilities.shape):
            plus = probabilities.copy()
            minus = probabilities.copy()
            plus[index] += delta
            minus[index] -= delta
            numeric = (criterion.forward(plus, target) - criterion.forward(minus, target)) / (2 * delta)
            self.assertAlmostEqual(analytic[index], numeric, places=7)

    def test_invalid_targets_raise(self) -> None:
        with self.assertRaises(ValueError):
            BinaryCrossEntropy().forward(np.array([[0.5]]), np.array([[2.0]]))


if __name__ == "__main__":
    unittest.main()
