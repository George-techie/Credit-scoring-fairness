"""Tests for the credit_scoring CLI."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from credit_scoring.cli import main


def test_audit_command(tmp_path, capsys):
    csv = tmp_path / "scored.csv"
    pd.DataFrame({
        "prediction": [1, 1, 1, 0, 0, 0],
        "gender": ["F", "F", "F", "M", "M", "M"],
    }).to_csv(csv, index=False)
    rc = main(["audit", "--input", str(csv), "--sensitive-col", "gender"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "disparate_impact_ratio" in out
    assert "four_fifths_rule" in out


def test_audit_with_truth_reports_odds(tmp_path, capsys):
    csv = tmp_path / "scored.csv"
    pd.DataFrame({
        "prediction": [1, 0, 1, 0],
        "truth": [1, 1, 1, 1],
        "grp": ["A", "A", "B", "B"],
    }).to_csv(csv, index=False)
    rc = main(["audit", "--input", str(csv), "--sensitive-col", "grp",
               "--pred-col", "prediction", "--truth-col", "truth"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "equalized_odds_difference" in out


def test_drift_command_detects_shift(tmp_path, capsys):
    base = tmp_path / "base.csv"
    cur = tmp_path / "cur.csv"
    pd.DataFrame({"EXT_SOURCE_2": [0.5] * 50}).to_csv(base, index=False)
    pd.DataFrame({"EXT_SOURCE_2": [0.9] * 50}).to_csv(cur, index=False)
    rc = main(["drift", "--baseline", str(base), "--current", str(cur)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Drift detected" in out
    assert "EXT_SOURCE_2" in out


def test_drift_command_no_shift(tmp_path, capsys):
    base = tmp_path / "base.csv"
    cur = tmp_path / "cur.csv"
    frame = pd.DataFrame({"EXT_SOURCE_2": [0.4, 0.5, 0.6] * 20})
    frame.to_csv(base, index=False)
    frame.to_csv(cur, index=False)
    rc = main(["drift", "--baseline", str(base), "--current", str(cur)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No drift detected" in out


def test_predict_command(tmp_path, capsys):
    import joblib
    import lightgbm as lgb
    import numpy as np

    from credit_scoring.schema import NUMERIC_FEATURES

    rng = np.random.default_rng(0)
    X = pd.DataFrame({f: rng.uniform(0, 1, 80) for f in NUMERIC_FEATURES})
    y = (X["EXT_SOURCE_2"] < 0.5).astype(int)
    model = lgb.LGBMClassifier(n_estimators=20, random_state=0, verbose=-1).fit(X, y)
    model_path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "feature_names": list(NUMERIC_FEATURES)}, model_path)

    record = {f: 0.5 for f in NUMERIC_FEATURES}
    rec_path = tmp_path / "rec.json"
    rec_path.write_text(json.dumps(record))

    rc = main(["predict", "--model", str(model_path), "--input", str(rec_path)])
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert "default_probability" in parsed
    assert parsed["default_prediction"] in (0, 1)


def test_no_command_errors():
    with pytest.raises(SystemExit):
        main([])
