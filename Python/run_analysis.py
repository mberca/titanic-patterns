"""Run with ``python -m Python.run_analysis`` from the repository root."""

import json
from .analysis import run_analysis


if __name__ == "__main__":
    _, tables, summary = run_analysis()
    print(json.dumps(summary, indent=2))
    print(tables["metrics"].query("phase == 'test'").to_string(index=False))
