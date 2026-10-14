"""Prepare a minimal historical dataset and a shared, ticket-grouped split."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
REVISION = "53be2fd391100044594de004d559858077a10c43"
SOURCE = (
    "https://raw.githubusercontent.com/paulhendricks/titanic/"
    f"{REVISION}/inst/data-raw/train.csv"
)
SEED = 20261014
EXPECTED_SHA256 = "7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f"


def main():
    with urlopen(SOURCE, timeout=30) as response:
        raw = response.read()
    if sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError("The source checksum differs from the pinned dataset")
    source = pd.read_csv(BytesIO(raw))
    columns = ["PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
               "Fare", "Embarked"]
    frame = source[columns].copy()
    if len(frame) != 891 or frame["PassengerId"].duplicated().any():
        raise ValueError("The pinned source no longer matches the expected 891-row dataset")
    tickets = source["Ticket"].str.strip()
    if tickets.isna().any():
        raise ValueError("Ticket groups must be observed before splitting")
    group_numbers = {ticket: f"G{i:04d}" for i, ticket in enumerate(sorted(tickets.unique()))}
    groups = tickets.map(group_numbers)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(splitter.split(frame, frame["Survived"], groups))
    split = pd.DataFrame({
        "PassengerId": frame["PassengerId"], "subset": "train",
        "cv_fold": -1, "ticket_group": groups,
    })
    split.loc[test, "subset"] = "test"
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    for fold, (_, validation) in enumerate(inner.split(
        frame.iloc[train], frame.iloc[train]["Survived"], groups.iloc[train]
    )):
        split.loc[train[validation], "cv_fold"] = fold
    train_groups = set(split.loc[split.subset == "train", "ticket_group"])
    test_groups = set(split.loc[split.subset == "test", "ticket_group"])
    if train_groups & test_groups:
        raise AssertionError("Ticket groups overlap the holdout boundary")
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    frame.to_csv(data / "passengers.csv", index=False, float_format="%.10g")
    split.to_csv(data / "split.csv", index=False)
    metadata = {
        "source_url": SOURCE, "source_revision": REVISION,
        "source_sha256": sha256(raw).hexdigest(),
        "source_package_license": "CC0",
        "license_declaration": (
            "https://github.com/paulhendricks/titanic/blob/" + REVISION + "/DESCRIPTION"
        ),
        "upstream_context": "https://www.kaggle.com/competitions/titanic",
        "rows": len(frame), "survived": int(frame.Survived.sum()),
        "split_seed": SEED, "split_method": "StratifiedGroupKFold; first outer fold held out",
        "grouping": "Shared ticket strings, replaced by invented group identifiers",
        "train_rows": len(train), "test_rows": len(test),
        "train_survived": int(frame.iloc[train].Survived.sum()),
        "test_survived": int(frame.iloc[test].Survived.sum()),
        "retained_columns": columns,
        "excluded_at_ingest": ["Name", "Ticket", "Cabin"],
        "scope": "891 labeled historical passenger records, not the full ship manifest",
    }
    (data / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
