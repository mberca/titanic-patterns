"""Bundle the independently generated R/Python evidence into an offline story."""

import csv
import json
import math
from pathlib import Path

from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[1]
MODELS = {"Prevalence baseline", "Random forest", "Bayesian logistic"}


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        converted = {}
        for field, value in row.items():
            try:
                number = float(value)
            except ValueError:
                converted[field] = value
            else:
                if not math.isfinite(number):
                    raise ValueError(f"Nonfinite value in {path}: {field}")
                converted[field] = number
        result.append(converted)
    return result


def load_edition(folder):
    tables = {
        name: read_table(folder / f"{name}.csv")
        for name in ("metrics", "predictions", "coefficients", "rules", "bayesian_rates")
    }
    metrics = tables["metrics"]
    if len(metrics) != 18 or {r["model"] for r in metrics} != MODELS:
        raise ValueError(f"Expected 15 validation and 3 test metric rows in {folder}")
    for model in MODELS:
        cv = [r for r in metrics if r["model"] == model and r["phase"] == "cv"]
        test = [r for r in metrics if r["model"] == model and r["phase"] == "test"]
        if {r["fold"] for r in cv} != set(range(5)) or len(test) != 1 or test[0]["n"] != 179:
            raise ValueError(f"Evaluation schema differs from the shared contract: {folder}")
    if len(tables["predictions"]) != 179 * 3:
        raise ValueError(f"Expected one held-out prediction per passenger/model: {folder}")
    for row in tables["predictions"]:
        if not 0 <= row["probability"] <= 1 or row["Survived"] not in (0, 1):
            raise ValueError(f"Invalid held-out probability or label in {folder}")
    for row in tables["rules"]:
        if row["N"] != 712 or "Outcome=" in row["antecedent"]:
            raise ValueError(f"Training-only rule contract failed in {folder}")
    if len(tables["bayesian_rates"]) != 6:
        raise ValueError(f"Expected the six observed sex/class strata in {folder}")
    return tables


def main():
    payload = {
        "provenance": json.loads((ROOT / "data" / "provenance.json").read_text()),
        "languages": {
            "R": load_edition(ROOT / "outputs" / "r"),
            "Python": load_edition(ROOT / "outputs" / "python"),
        },
    }
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = (ROOT / "web" / "story.html").read_text(encoding="utf-8")
    html = template.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", encoded)
    output = ROOT / "docs"
    output.mkdir(exist_ok=True)
    (output / "index.html").write_text(html, encoding="utf-8")
    print(f"Built {output / 'index.html'} ({len(html):,} characters)")


if __name__ == "__main__":
    main()
