# R-first historical analysis

Open `titanic-patterns.Rproj` in RStudio, or run R from the repository root.
The R analysis is self-contained and never reads Python outputs. It uses only
the committed public `data/passengers.csv` and grouped `data/split.csv`.
Read [the shared analysis contract](../METHODOLOGY.md) before interpreting results.

## Run

```r
source("R/install.R")         # Installs missing packages only; no upgrades.
source("R/run_analysis.R")    # Complete analysis, CSVs, and static PNGs.
source("R/test-analysis.R")   # Focused unit + committed-data integration tests.
source("R/render_analysis.R") # Execute the narrative and save Markdown.
```

On Windows, the equivalent commands are:

```powershell
& 'C:\Program Files\R\R-4.4.3\bin\Rscript.exe' R\run_analysis.R
& 'C:\Program Files\R\R-4.4.3\bin\Rscript.exe' R\test-analysis.R
& 'C:\Program Files\R\R-4.4.3\bin\Rscript.exe' R\render_analysis.R
```

All commands run from the repository root. No download, Python, Shiny server,
LaTeX, or Pandoc is needed to execute the analysis. Rendering uses
`github_document` when Pandoc is available; otherwise `knitr::knit` executes
all R Markdown chunks directly. Both routes persist `outputs/r/analysis.md`,
with relative links to the neighboring PNGs. No standalone HTML is generated.

## Files and responsibilities

| File | Purpose |
| --- | --- |
| `analysis.R` | Reusable preprocessing, models, scoring, rules, beta updates, plots |
| `run_analysis.R` | Root-relative executable entry point |
| `test-analysis.R` | Hand-calculated metrics, leakage/group separation, analytic derivative and known posterior fixture tests, rule recounts, export integration |
| `analysis.Rmd` | Runnable narrative with computed tables and uncertainty limitations |
| `render_analysis.R` | Execute the narrative, gracefully avoiding a Pandoc dependency |
| `install.R` | Missing-package-only installation |

## Outputs

All artifacts are regenerated under `outputs/r/`:

- `metrics.csv`: each model on five CV folds and one test subset (18 rows).
  `fold=-1` marks the test subset; CV folds are 0–4.
- `predictions.csv`: held-out records only, one probability per model.
- `coefficients.csv`: intercept, then the nine ordered features; mode,
  Laplace SD, approximate central 95% credible interval, odds ratios and
  corresponding intervals, and prior SD.
- `design_train.csv`, `design_test.csv`: IDs followed by the shared nine-feature
  matrix in original data order. IDs are audit keys, never predictors.
- `preprocessing.json`: full-training medians, alphabetical-tie mode,
  numeric means/sample-SD scales, and ordered feature names.
- `rules.csv`: the complete retained training-only rule set, with raw joint,
  antecedent, and consequent counts, true training denominator, and baseline.
- `bayesian_rates.csv`: full-sample sex-by-class Beta(1,1) posterior summaries.
- `missingness.csv`: count and fraction missing for every retained field.
- `calibration.csv`: all five fixed bins per test model; empty bins are
  explicitly retained with count zero and unavailable estimates.
- Seven PNGs: `overview`, `performance`, `roc`, `calibration`,
  `bayesian_effects`, `bayesian_rates`, and `rules`.
- Executed `analysis.md` and `session-info.txt`.

The CSV writer uses 17-digit precision; all predictors and preprocessing
values remain unrounded. No fitted model or personal passenger simulator
is serialized.

## Interpretation guardrails

This is a selected 891-record historical subset, not the complete manifest.
Preprocessing and Apriori see training records only; every CV fold relearns
its transform. Forest settings are fixed at 500 trees, `mtry=3`, `nodesize=5`,
factor outcome levels `0,1`, positive probability `"1"`, seed `20261014`.

The logistic model uses independent Normal priors (intercept SD 5; others
2.5), explicit analytic-gradient BFGS checks with analytic Newton polishing
to a stationary mode, and an inverse analytic Hessian Laplace covariance.
Predictive probabilities are **MAP plug-in estimates**,
not integrated or exact posterior predictions. Intervals are approximate
marginal credible intervals, not confidence intervals.

Descriptive Beta(1,1) rates deliberately use the full sample and are clearly
separate from the holdout exercise. Association rules are exploratory,
overlapping training patterns, not significance tests. Shared features
and Bayesian modes should closely match the Python edition; forests need
not. No causal claim, rescue advice, universal algorithm ranking, or
individual decision-use is justified.
