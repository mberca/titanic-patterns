# Analysis contract

This project is a historical observational case study, not a rescue-prioritization
tool or evidence about causal effects. It uses the 891 labeled passenger records
distributed by the CC0-declared `titanic` R package, not the complete ship manifest.

## Shared data and evaluation

- `data/passengers.csv` is a minimal extract of a pinned public source. Names,
  cabin strings, and ticket strings are not retained in that file.
- `data/split.csv` is the single source of truth for both languages. Shared
  tickets remain within the same train/test subset and within each validation
  fold. Ticket grouping is not a perfect family or social-group identifier.
- The first of five stratified, grouped outer folds is the held-out test set.
  Five grouped folds within the training set provide development estimates.
  Only the training data are used for fitting, imputation, scaling, and rule
  discovery. No hyperparameters are selected from test performance.
- Forest settings and Bayesian priors are fixed in advance. The final test set
  is for a one-time comparison, not further tuning.

## Features

Both languages use this ordered matrix:

`age_z`, `log_fare_z`, `family_size_z`, `age_missing`, `sex_male`,
`class_2`, `class_3`, `embarked_Q`, `embarked_S`.

Age and fare are imputed with training medians. Embarkation is imputed with the
training mode, resolving ties alphabetically. `log_fare = log1p(fare)` and
`family_size = SibSp + Parch + 1`. The first three columns are centered and scaled
using **training sample** standard deviations (`ddof=1` in Python). A zero scale
is replaced by one. The reference levels are female, first class, and embarkation
at C. Names, IDs, tickets, cabins, groups, and the outcome are not predictors.

Every cross-validation fold relearns these preprocessing statistics from that
fold's training rows; the final model relearns them from the full training subset.

## Models

1. **Prevalence baseline:** constant probability equal to the training survival
   rate. It is refitted inside every validation fold.
2. **Random forest:** 500 trees, `mtry/max_features=3`, minimum terminal/leaf size
   5, no class weighting or test-driven tuning. R uses `randomForest`, Python
   uses scikit-learn. Their algorithms and probability definitions differ;
   equal forest predictions are not expected.
3. **Bayesian logistic regression:** independent zero-centered normal priors,
   intercept SD 5 and coefficient SD 2.5. Fit the posterior mode with analytic
   gradient, then approximate the coefficient posterior with the inverse
   negative-log-posterior Hessian (Laplace approximation). Show approximate
   marginal 95% credible intervals and odds ratios. Evaluated probabilities
   are **posterior-mode plug-in probabilities**, not a fully integrated
   posterior predictive. No MCMC convergence or exact-posterior claim is made.

Report training-fold validation and held-out ROC AUC, Brier score, log loss,
and balanced accuracy at a prespecified 0.5 threshold. Lower Brier/log loss is
better. Include ROC and five-bin reliability curves with bin counts; small test
sets make differences uncertain. Do not declare a universal best model or
statistically significant improvement.

## Association rules

Mine **training rows only**, using fixed categories: sex; passenger class;
age `Child (<16)`, `Adult (16–59)`, `Older (60+)`, or `Unknown`; family size
`Alone`, `2–4`, `5+`; fare `Low (<=10)`, `Middle (10–30)`, `High (>30)`;
and survival outcome. Embarkation is excluded from the rule exhibit for clarity.

Apriori discovers itemsets up to size three, with survival confined to the
single-item consequent and no outcome in the antecedent. Default thresholds:
support 0.04, confidence 0.60, joint count 20, lift 1.0. All are exploratory
filters, not significance tests. Counts and conditional baselines are computed
within the training population. Export the complete retained rule set.

## Bayesian descriptive rates

For sex-by-class strata, show observed counts and a **Beta(1,1)** prior updated
by the historical sample's survived/did-not-survive counts. Report posterior
mean and central 95% beta credible interval. These conjugate summaries describe
the **observed labeled sample**, not population causal effects; they are
separate from the train/test prediction exercise. Reveal group counts alongside
intervals. Do not treat a credible interval as a frequentist confidence interval.

## Reproducibility and limits

Predeclare seed `20261014`; consume the committed split in both languages.
Report missingness and sample coverage. The data are curated, incomplete
historical records, not a random sample of a modern deployment population.
Sex and class differences are descriptive and conditioned on this selected
sample; they do not identify why a person survived. Modeling correlated
historical characteristics does not justify discriminatory decisions.
