"""Check shared evidence, not equality of two different forest implementations."""

import csv
import math
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def table(edition, name):
    with (ROOT / "outputs" / edition / f"{name}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def compare(name, keys, numbers, tolerance=1e-8):
    def indexed(edition):
        rows = table(edition, name)
        result = {tuple(row[k] for k in keys): row for row in rows}
        if len(result) != len(rows):
            raise AssertionError(f"Duplicate keys in {edition}/{name}")
        return result

    r_rows, py_rows = indexed("r"), indexed("python")
    if r_rows.keys() != py_rows.keys():
        raise AssertionError(f"R/Python keys differ for {name}")
    maximum = 0.0
    for key, r_row in r_rows.items():
        for field in numbers:
            a, b = float(r_row[field]), float(py_rows[key][field])
            if not math.isfinite(a) or not math.isfinite(b):
                raise AssertionError(f"Nonfinite {name}/{key}/{field}")
            maximum = max(maximum, abs(a - b))
            if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
                raise AssertionError(f"R/Python mismatch {name}/{key}/{field}: {a} vs {b}")
    print(f"{name}: {len(r_rows)} matching rows; maximum absolute difference {maximum:.3g}")


def check_evaluation():
    with (ROOT / "data" / "split.csv").open(newline="") as handle:
        split = list(csv.DictReader(handle))
    with (ROOT / "data" / "passengers.csv").open(newline="") as handle:
        outcomes = {row["PassengerId"]: int(row["Survived"]) for row in csv.DictReader(handle)}
    test_ids = {row["PassengerId"] for row in split if row["subset"] == "test"}
    for edition in ("r", "python"):
        predictions = table(edition, "predictions")
        metrics = table(edition, "metrics")
        for model in ("Prevalence baseline", "Random forest", "Bayesian logistic"):
            rows = [row for row in predictions if row["model"] == model]
            if len(rows) != len(test_ids) or {r["PassengerId"] for r in rows} != test_ids:
                raise AssertionError(f"Held-out prediction IDs differ: {edition}/{model}")
            actual = [int(row["Survived"]) for row in rows]
            if actual != [outcomes[row["PassengerId"]] for row in rows]:
                raise AssertionError(f"Outcome labels differ: {edition}/{model}")
            probability = [float(row["probability"]) for row in rows]
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in probability):
                raise AssertionError(f"Invalid probability: {edition}/{model}")
            expected = {
                "roc_auc": roc_auc_score(actual, probability),
                "brier": brier_score_loss(actual, probability),
                "log_loss": log_loss(actual, probability, labels=[0, 1]),
                "balanced_accuracy": balanced_accuracy_score(actual, [p >= .5 for p in probability]),
            }
            recorded = [r for r in metrics if r["phase"] == "test" and r["model"] == model]
            if len(recorded) != 1:
                raise AssertionError(f"Missing/duplicate test metrics: {edition}/{model}")
            for field, value in expected.items():
                if not math.isclose(value, float(recorded[0][field]), abs_tol=1e-8, rel_tol=1e-8):
                    raise AssertionError(f"Metric/prediction mismatch: {edition}/{model}/{field}")
    print("Both editions: held-out IDs/outcomes and all exported test metrics verified.")


def main():
    check_evaluation()
    features = [
        "age_z", "log_fare_z", "family_size_z", "age_missing", "sex_male",
        "class_2", "class_3", "embarked_Q", "embarked_S",
    ]
    compare("design_train", ["PassengerId"], features)
    compare("design_test", ["PassengerId"], features)
    compare("coefficients", ["term"],
            ["estimate", "sd", "lower", "upper", "odds_ratio", "or_lower", "or_upper"],
            tolerance=2e-5)
    compare("rules", ["antecedent", "consequent"],
            ["support", "confidence", "lift", "count", "antecedent_count",
             "consequent_count", "baseline", "N"])
    compare("bayesian_rates", ["sex", "pclass"],
            ["n", "survived", "observed_rate", "posterior_mean", "lower", "upper"])
    print("Forest probability parity is intentionally not asserted: the implementations differ.")


if __name__ == "__main__":
    main()
