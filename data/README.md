# Public historical data

The source is `titanic_train` from Paul Hendricks' **titanic** R package,
whose pinned [`DESCRIPTION`](https://github.com/paulhendricks/titanic/blob/53be2fd391100044594de004d559858077a10c43/DESCRIPTION)
declares **CC0**. It packages the labeled training sample used in the
[Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

The analysis uses **891 labeled passenger records**, not the complete passenger
and crew manifest. These are real historical observations, not synthetic data
and not a representative sample of a modern population.

## Files

| File | Purpose |
|---|---|
| [`passengers.csv`](passengers.csv) | Minimal shared input for R and Python |
| [`split.csv`](split.csv) | Frozen training/test membership and inner validation folds |
| [`provenance.json`](provenance.json) | Source URL, immutable revision, SHA-256 checksum, scope, and split counts |

`passengers.csv` includes `PassengerId`, `Survived` (1 means survived),
`Pclass` (1/2/3), `Sex` (source labels female/male), `Age`, `SibSp`, `Parch`,
`Fare`, and `Embarked` (C/Q/S). Missing age and embarkation values are retained.
`PassengerId` is only a join key, never a model feature.

Raw passenger names, ticket strings, and cabin strings are **not committed**.
The preparation script uses shared tickets only to create group identifiers,
then discards the original strings. These groups do not perfectly identify
families or other social relationships.

The shared split has **712 training rows and 179 held-out test rows**.
Training records have validation folds 0–4; held-out records have fold -1.
The same ticket group never spans the test boundary or two validation folds.
Neither group identifiers nor fold membership are predictors.

## Recreate the extract

With the Python dependencies installed, run from the repository root:

```powershell
python scripts/prepare_data.py
```

The source revision and expected checksum are pinned. This is the only step
requiring a download. Normal R/Python analysis runs use the committed extract
without internet access or a Kaggle account.

Read the [analysis contract](../METHODOLOGY.md) before interpreting the results.
