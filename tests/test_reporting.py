"""Experiment logs must have unambiguous epoch semantics and render offline."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

from numpynet.experiments import train_xor
from numpynet.reports import write_comparison_csv, write_history_csv, write_loss_svg


class ReportingTests(unittest.TestCase):
    def test_history_includes_epoch_zero_and_every_update(self) -> None:
        result = train_xor(epochs=7, learning_rate=0.7)
        self.assertEqual([item.epoch for item in result.history], list(range(8)))
        self.assertEqual([item.loss for item in result.history], result.losses)
        self.assertEqual(result.history[-1].accuracy, result.accuracy)

    def test_export_csv_contains_epoch_zero(self) -> None:
        result = train_xor(epochs=3)
        with TemporaryDirectory() as directory:
            path = write_history_csv(Path(directory) / "sub" / "train.csv", result.history)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(list(rows[0]), ["epoch", "loss", "accuracy"])
            self.assertEqual([int(row["epoch"]) for row in rows], [0, 1, 2, 3])

    def test_svg_can_be_parsed_as_xml(self) -> None:
        first = train_xor(epochs=10, learning_rate=0.01)
        second = train_xor(epochs=10, learning_rate=0.7)
        comparisons = [(0.01, first.history), (0.7, second.history)]
        with TemporaryDirectory() as directory:
            plot = write_loss_svg(Path(directory) / "comparison.svg", comparisons)
            root = ET.parse(plot).getroot()
            self.assertTrue(root.tag.endswith("svg"))
            self.assertEqual(sum(node.tag.endswith("polyline") for node in root.iter()), 2)
            summary = write_comparison_csv(Path(directory) / "summary.csv", comparisons)
            with summary.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[0]["initial_loss"]), float(rows[1]["initial_loss"]))

    def test_reject_empty_svg_input(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_loss_svg(Path(directory) / "plot.svg", [])

    def test_svg_preserves_labels_with_xml_characters(self) -> None:
        history = train_xor(epochs=2).history
        label = "clip & decay <baseline>"
        with TemporaryDirectory() as directory:
            plot = write_loss_svg(Path(directory) / "plot.svg", [(label, history)])
            root = ET.parse(plot).getroot()
            self.assertIn(label, [node.text for node in root.iter()])


if __name__ == "__main__":
    unittest.main()
