"""Portable inference for the final R XGBoost model; no R runtime is needed."""
import gzip
import hashlib
import json
import math
from pathlib import Path


class ModelError(ValueError):
    pass


def finite_number(value):
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


class TrajectoryModel:
    def __init__(self, data, digest=""):
        self.data = data
        self.digest = digest
        self._validate()
        self.classes = data["classes"]
        self.inputs = data["inputs"]
        self.features = data["features"]
        self.feature_index = {item["name"]: index for index, item in enumerate(self.features)}
        self.trees = data["trees"]
        self.base_margin = [float(value) for value in data["base_margin"]]

    @classmethod
    def load(cls, path):
        path = Path(path)
        with path.open("rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        with gzip.open(str(path), "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        model = cls(data, digest)
        model.verify_export()
        return model

    def _validate_tree(self, node, feature_names, seen):
        if not isinstance(node, dict) or "nodeid" not in node:
            raise ModelError("XGBoost tree node is invalid.")
        node_id = int(node["nodeid"])
        if node_id in seen:
            raise ModelError("XGBoost tree contains duplicate or cyclic nodes.")
        seen.add(node_id)
        if "leaf" in node:
            if not finite_number(node["leaf"]):
                raise ModelError("XGBoost leaf value is invalid.")
            return
        if node.get("split") not in feature_names or not finite_number(node.get("split_condition")):
            raise ModelError("XGBoost split definition is invalid.")
        children = node.get("children")
        if not isinstance(children, list) or not children:
            raise ModelError("XGBoost split node has no children.")
        child_ids = {int(child.get("nodeid", -1)) for child in children}
        if int(node.get("yes", -1)) not in child_ids or int(node.get("no", -1)) not in child_ids:
            raise ModelError("XGBoost branch points to an unknown child.")
        for child in children:
            self._validate_tree(child, feature_names, seen)

    def _validate(self):
        data = self.data
        if data.get("format") != "charls-xgboost-v1":
            raise ModelError("Unsupported model format. Re-run export_model.R.")
        classes = data.get("classes", [])
        if classes != ["Class1", "Class2", "Class3", "Class4"]:
            raise ModelError("The model must contain Class1 through Class4 in order.")
        specs = data.get("inputs", [])
        features = data.get("features", [])
        trees = data.get("trees", [])
        if not specs or not features or not trees:
            raise ModelError("Model inputs, encoded features, or trees are empty.")
        if data.get("n_trees") != len(trees) or len(trees) % len(classes):
            raise ModelError("The multiclass tree count is inconsistent.")
        if data.get("trees_per_class") != len(trees) // len(classes):
            raise ModelError("The trees-per-class metadata is inconsistent.")
        base_margin = data.get("base_margin", [])
        if len(base_margin) != len(classes) or not all(finite_number(value) for value in base_margin):
            raise ModelError("The XGBoost base margin is invalid.")

        names = [spec.get("name") for spec in specs]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ModelError("Raw input names are duplicated or empty.")
        spec_map = {spec["name"]: spec for spec in specs}
        for spec in specs:
            if spec.get("type") not in ("number", "category"):
                raise ModelError("Unsupported raw input type.")
            if spec["type"] == "category":
                options = spec.get("options", [])
                codes = [option.get("value") for option in options]
                if not codes or len(set(codes)) != len(codes) or any(not isinstance(code, str) for code in codes):
                    raise ModelError("Category options are invalid.")

        feature_names = [feature.get("name") for feature in features]
        if len(set(feature_names)) != len(feature_names):
            raise ModelError("Encoded feature names are duplicated.")
        for feature in features:
            original = feature.get("original")
            if original not in spec_map or feature.get("type") != spec_map[original]["type"]:
                raise ModelError("Encoded feature is not linked to a compatible raw input.")
            if feature["type"] == "category":
                values = feature.get("values", {})
                expected = {option["value"] for option in spec_map[original]["options"]}
                if set(values) != expected or not all(finite_number(value) for value in values.values()):
                    raise ModelError("Category encoding is incomplete.")
        for tree in trees:
            self._validate_tree(tree, set(feature_names), set())

    def validate_inputs(self, raw):
        if not isinstance(raw, dict):
            raise ModelError("Inputs must be provided as a field mapping.")
        cleaned, warnings = {}, []
        for spec in self.inputs:
            name = spec["name"]
            label = spec.get("label_en") or spec.get("label", name)
            value = raw.get(name)
            if value is None or value == "":
                raise ModelError("Please complete: " + label)
            if spec["type"] == "category":
                if isinstance(value, bool):
                    raise ModelError(label + " has an invalid category.")
                code = str(value)
                if code not in [option["value"] for option in spec["options"]]:
                    raise ModelError(label + " is outside the development-set coding range.")
                cleaned[name] = code
            else:
                if not finite_number(value):
                    raise ModelError(label + " must be a finite number.")
                value = float(value)
                cleaned[name] = value
                low, high = spec.get("observed_min"), spec.get("observed_max")
                if low is not None and high is not None and not float(low) <= value <= float(high):
                    warnings.append(
                        "{0} is outside the development-set range ({1:g}–{2:g} {3}). "
                        "Please check the value; performance outside this range is unknown.".format(
                            label, float(low), float(high), spec.get("unit", "")
                        )
                    )
        return cleaned, warnings

    def encode(self, raw):
        cleaned, warnings = self.validate_inputs(raw)
        vector = []
        for feature in self.features:
            value = cleaned[feature["original"]]
            if feature["type"] == "category":
                value = feature["values"][value]
            vector.append(float(value))
        return vector, warnings

    def _tree_value(self, tree, vector):
        node = tree
        while "leaf" not in node:
            value = vector[self.feature_index[node["split"]]]
            target = int(node["yes"] if value < float(node["split_condition"]) else node["no"])
            children = {int(child["nodeid"]): child for child in node["children"]}
            if target not in children:
                raise ModelError("XGBoost traversal reached an unknown child.")
            node = children[target]
        return float(node["leaf"])

    def predict_encoded(self, vector):
        if len(vector) != len(self.features) or not all(finite_number(value) for value in vector):
            raise ModelError("Encoded feature length or values are invalid.")
        margins = list(self.base_margin)
        class_count = len(self.classes)
        for tree_index, tree in enumerate(self.trees):
            margins[tree_index % class_count] += self._tree_value(tree, vector)
        maximum = max(margins)
        exponentials = [math.exp(value - maximum) for value in margins]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def predict(self, raw):
        vector, warnings = self.encode(raw)
        probabilities = self.predict_encoded(vector)
        top_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "class": self.classes[top_index],
            "classes": list(self.classes),
            "probabilities": probabilities,
            "warnings": warnings,
            "tie": sum(abs(value - probabilities[top_index]) < 1e-12 for value in probabilities) > 1,
        }

    def verify_export(self):
        cases = self.data.get("verification_cases", [])
        if not cases:
            raise ModelError("The model lacks R parity verification profiles.")
        for case in cases:
            vector, _ = self.encode(case["inputs"])
            expected_vector = case.get("encoded", [])
            if len(expected_vector) != len(vector) or any(
                abs(actual - expected) > 1e-10
                for actual, expected in zip(vector, expected_vector)
            ):
                raise ModelError("Website encoding does not match the R encoding.")
            actual_probability = self.predict_encoded(vector)
            expected_probability = case.get("probabilities", [])
            if len(expected_probability) != len(actual_probability) or any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(actual_probability, expected_probability)
            ):
                raise ModelError("Website prediction does not match the R XGBoost model.")
        return len(cases)
