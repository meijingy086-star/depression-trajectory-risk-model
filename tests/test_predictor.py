import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from predictor import ModelError, TrajectoryModel


def leaf(node_id, value):
    return {"nodeid": node_id, "leaf": value}


def split(feature, threshold, left_value, right_value):
    return {
        "nodeid": 0, "split": feature, "split_condition": threshold,
        "yes": 1, "no": 2, "missing": 2,
        "children": [leaf(1, left_value), leaf(2, right_value)],
    }


def softmax(values):
    maximum = max(values)
    values = [math.exp(value - maximum) for value in values]
    return [value / sum(values) for value in values]


def fixture():
    data = {
        "format": "charls-xgboost-v1",
        "classes": ["Class1", "Class2", "Class3", "Class4"],
        "inputs": [
            {"name": "sleep", "label_en": "Sleep", "type": "number",
             "observed_min": 2, "observed_max": 10},
            {"name": "gender", "label_en": "Gender", "type": "category",
             "options": [{"value": "0"}, {"value": "1"}]},
        ],
        "features": [
            {"name": "sleep", "type": "number", "original": "sleep"},
            {"name": "gender1", "type": "category", "original": "gender",
             "values": {"0": 0, "1": 1}},
        ],
        "base_margin": [0, 0, 0, 0],
        "trees": [
            split("sleep", 6, 1, 0),
            split("gender1", 0.5, 1, 0),
            leaf(0, 0.5),
            leaf(0, -0.5),
        ],
        "n_trees": 4,
        "trees_per_class": 1,
    }
    expected = softmax([1, 1, 0.5, -0.5])
    data["verification_cases"] = [{
        "inputs": {"sleep": 5, "gender": "0"},
        "encoded": [5, 0],
        "probabilities": expected,
    }]
    return data


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.model = TrajectoryModel(fixture())

    def test_xgboost_threshold_uses_strict_less_than(self):
        left = self.model.predict({"sleep": 5.999999, "gender": "0"})
        boundary = self.model.predict({"sleep": 6, "gender": "0"})
        self.assertGreater(left["probabilities"][0], boundary["probabilities"][0])

    def test_encoding_and_r_parity_case(self):
        self.assertEqual(self.model.encode({"sleep": 7, "gender": "1"})[0], [7, 1])
        self.assertEqual(self.model.verify_export(), 1)

    def test_probabilities_sum_to_one(self):
        probabilities = self.model.predict({"sleep": 7, "gender": "1"})["probabilities"]
        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_missing_unknown_and_nonfinite_inputs_fail(self):
        with self.assertRaises(ModelError):
            self.model.predict({"gender": "0"})
        with self.assertRaises(ModelError):
            self.model.predict({"sleep": 6, "gender": "2"})
        with self.assertRaises(ModelError):
            self.model.predict({"sleep": float("nan"), "gender": "0"})

    def test_outside_training_range_warns(self):
        self.assertEqual(len(self.model.predict({"sleep": 12, "gender": "0"})["warnings"]), 1)

    def test_invalid_tree_is_rejected(self):
        data = fixture()
        data["trees"][0]["yes"] = 99
        with self.assertRaises(ModelError):
            TrajectoryModel(data)

    def test_gzip_roundtrip_and_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.json.gz"
            with gzip.open(str(path), "wt", encoding="utf-8") as handle:
                json.dump(fixture(), handle)
            loaded = TrajectoryModel.load(path)
            self.assertEqual(len(loaded.digest), 64)

    def test_corrupt_expected_prediction_fails(self):
        data = fixture()
        data["verification_cases"][0]["probabilities"] = [1, 0, 0, 0]
        with self.assertRaises(ModelError):
            TrajectoryModel(data).verify_export()


if __name__ == "__main__":
    unittest.main()
