"""Training: deterministic, sampled correctly, thresholds chosen by F1, test split closed."""

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from statementscreen import splits, train
from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.writer import write_batch


def _toy(seed: int, n: int = 3000):
    """A small problem: the label depends on two of 30 sparse features."""
    rng = np.random.default_rng(seed)
    X = sp.random(n, 30, density=0.3, format="csr", random_state=seed, dtype=np.float32)
    dense = X.toarray()
    y = ((dense[:, 0] + dense[:, 1] + rng.normal(0, 0.1, n)) > 0.5).astype(np.int8)
    return X, y


def test_training_is_deterministic() -> None:
    X, y = _toy(1)
    Xv, yv = _toy(2, 500)
    names = [f"f{i}" for i in range(30)]
    setting = train.GRID[0]
    runs = []
    for _ in range(2):
        booster = train.fit(X, y, np.ones(len(y)), Xv, yv, np.ones(len(yv)), setting, names)
        runs.append(booster.predict(Xv, num_iteration=booster.best_iteration))
    assert np.array_equal(runs[0], runs[1])  # identical, not just close
    assert train.BASE_PARAMS["deterministic"] and train.BASE_PARAMS["num_threads"] == 2


def test_best_threshold_on_a_hand_example() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    # cut at 0.35: 3 flagged, 2 right -> P 2/3, R 1, F1 0.8 (beats 0.4: F1 0.5, 0.8: F1 2/3)
    best = train.best_threshold(y, scores)
    assert best["threshold_edge"] == pytest.approx(0.35) and best["gap_below"] == 0.1
    # placed halfway in log-odds between 0.1 and 0.35: same predictions on these rows
    logit = np.log([0.1 / 0.9, 0.35 / 0.65]).mean()
    assert best["threshold"] == pytest.approx(1 / (1 + np.exp(-logit)))  # about 0.197
    assert 0.1 < best["threshold"] < 0.35
    assert (best["precision"], best["recall"], best["f1"]) == pytest.approx((2 / 3, 1.0, 0.8))
    assert train.best_threshold(np.zeros(3), np.ones(3))["f1"] is None


def test_a_perfect_split_gets_its_threshold_in_the_gap() -> None:
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.001, 0.01, 0.2, 0.9990, 0.9999])
    best = train.best_threshold(y, scores)
    assert best["threshold_edge"] == pytest.approx(0.9990)  # the weakest positive
    assert best["gap_below"] == pytest.approx(0.2) and 0.2 < best["threshold"] < 0.999
    assert best["f1"] == 1.0


def test_ties_go_to_pr_auc_then_the_earlier_setting() -> None:
    first = {"f1": 0.99, "pr_auc": 0.95}
    assert train.better(first, None)
    assert train.better({"f1": 0.99, "pr_auc": 0.97}, first)
    assert not train.better({"f1": 0.99, "pr_auc": 0.95}, first)  # full tie: keep earlier
    assert not train.better({"f1": 0.98, "pr_auc": 1.0}, first)


def test_label_rows_keep_every_positive_and_weight_sampled_negatives() -> None:
    rows = pl.DataFrame({
        "is_salary": [True, True, False, False, False, True],
        "sampled":   [False, True, True, False, True, False],
        "holdout":   [False, False, False, False, True, True],
    })  # fmt: skip
    use, y, w = train.label_rows(rows, "is_salary", holdout=False, neg_rate=0.1)
    assert use.tolist() == [True, True, True, False, False, False]  # row 3: unsampled negative
    assert y.tolist() == [1, 1, 0]
    assert w.tolist() == pytest.approx([1, 1, 10])  # 1 / 0.1 keeps probabilities calibrated
    use, y, w = train.label_rows(rows, "is_salary", holdout=True, neg_rate=0.1)
    assert use.tolist() == [False, False, False, False, True, True] and y.tolist() == [0, 1]


def test_holdout_is_by_statement_and_fixed() -> None:
    ids = [f"stmt_{i:04d}" for i in range(1, 201)]
    a = train.holdout_statements(ids)
    assert a == train.holdout_statements(list(reversed(ids)))  # order doesn't matter
    assert len(a) == 20 and a <= set(ids)


def test_labels_come_from_train_or_dev_only(tmp_path, monkeypatch) -> None:
    batch = tmp_path / "batch"
    write_batch(plan_batch(BatchConfig(seed=11), 20), batch)
    split_dir = tmp_path / "splits"
    splits.write(*splits.build(batch / "manifest.csv"), split_dir)
    labels = train.load_labels(batch, "train", split_dir)
    truth = pl.read_parquet(batch / "transactions.parquet")
    ids = splits.load_split("train", batch / "manifest.csv", split_dir)
    expected = truth.filter(pl.col("statement_id").is_in(ids))
    assert labels.height == expected.height
    assert labels["is_rent"].sum() == (expected["category"] == "rent").sum()
    monkeypatch.delenv(splits.OPEN_TEST_ENV, raising=False)
    with pytest.raises(splits.SplitError, match="closed"):
        train.load_labels(batch, "test", split_dir)
