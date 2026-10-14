# Python / Titanic patterns

A complete offline companion to the R-first historical case study. Read the
root `METHODOLOGY.md` and data provenance before interpreting results. The input
is a pinned public 891-record labeled sample, **not the full ship manifest**.

## Run on Windows (Python 3.11)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m Python.run_analysis
.\.venv\Scripts\python.exe -m pytest tests\test_python_analysis.py
.\.venv\Scripts\python.exe -m Python.make_notebook
.\.venv\Scripts\python.exe -m Python.execute_notebook
```

No Kaggle account, API key, or runtime internet access is needed after installing
dependencies. The notebook calls the same tested functions as the CLI; rerunning
it deterministically regenerates the outputs. Forest parallelism is bounded at
two workers. `requirements-resolved.txt` under `outputs/python/` records the
actual environment used for the checked-in execution.

## What is estimated?

- **Describe:** full-sample sex-by-class Beta(1,1) posterior rates, exact central
  beta intervals, observed counts and missingness. These are not test results.
- **Discover:** Apriori itemsets up to length three, only in 712 training
  records. Survival is the singleton RHS; maximum two predictor items on LHS.
  Thresholds are support .04, confidence .60, joint count 20, and lift 1.0.
  They are exploratory filters, not multiple-testing-adjusted significance.
- **Predict:** prevalence, a fixed 500-tree forest, and normal-prior Bayesian
  logistic regression on the committed grouped folds and 179 held-out records.
  Imputation and sample-SD standardization are relearned inside each fold.
- **Quantify:** logistic posterior mode plus inverse analytic Hessian (Laplace);
  approximate marginal 95% coefficient credible intervals, not simultaneous
  intervals, MCMC draws or exact-posterior intervals. Predictions use the
  posterior-mode plug-in, **not** integrated posterior prediction.

`outputs/python/` contains high-precision CSVs: fold/test metrics, test-only
predictions, coefficients, train/test design matrices, rules, descriptive rates,
missingness, plus preprocessing metadata and convergence summary. Additional
`cv_predictions.csv` and five-bin `calibration.csv` make diagnostics auditable.
The notebook and four static PNG exhibits are generated from actual results.

## Limits

Sex, class, age and fare are historically correlated. Adjusted coefficients and
association rules cannot establish causes or explain individual fates.
Selection, missing records, small subgroups and shared-ticket dependence limit
generalization. Ticket groups are not perfect families/social groups. The test
set is small: metric differences do not establish a universal winner. Forests
differ across language implementations; Bayesian design and mode should agree
numerically. No personal prediction simulator, deployment or rescue
recommendations are provided, and these historical differences do not justify
discriminatory decisions.
