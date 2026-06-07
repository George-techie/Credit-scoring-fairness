"""Command-line interface for the credit-scoring service.

Exposes the package's operations from the shell:

    python -m credit_scoring audit  --input scored.csv --pred-col pred --sensitive-col gender
    python -m credit_scoring drift   --baseline base.csv --current live.csv
    python -m credit_scoring predict --model model/lgb_model.joblib --input applicant.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import pandas as pd

from .drift import detect_drift
from .fairness import audit_fairness, passes_four_fifths_rule


def _cmd_audit(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input)
    truth = df[args.truth_col].to_numpy() if args.truth_col else None
    report = audit_fairness(
        df[args.pred_col].to_numpy(),
        df[args.sensitive_col].to_numpy(),
        y_true=truth,
    )
    print(f"Groups audited on '{args.sensitive_col}':")
    for g in report.by_group:
        print(
            f"  {g.group}: n={g.n} selection_rate={g.selection_rate:.3f} "
            f"tpr={g.tpr:.3f} fpr={g.fpr:.3f}"
        )
    print(f"demographic_parity_difference: {report.demographic_parity_difference:.3f}")
    print(f"disparate_impact_ratio:        {report.disparate_impact_ratio:.3f}")
    print(f"equalized_odds_difference:     {report.equalized_odds_difference:.3f}")
    verdict = "PASS" if passes_four_fifths_rule(report) else "FAIL"
    print(f"four_fifths_rule: {verdict}")
    return 0


def _cmd_drift(args: argparse.Namespace) -> int:
    baseline = pd.read_csv(args.baseline)
    current = pd.read_csv(args.current)
    report = detect_drift(baseline, current)
    if report.any_drift:
        print("Drift detected in: " + ", ".join(report.drifted_features))
    else:
        print("No drift detected.")
    for r in report.results:
        flag = "DRIFT" if r.drift_detected else "ok"
        print(f"  {r.feature}: p={r.p_value:.4f} ks={r.ks_statistic:.4f} [{flag}]")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    import joblib

    from .features import prepare_features
    from .inference import decide, predict_default_proba
    from .schema import NUMERIC_FEATURES

    with open(args.input) as fh:
        record = pd.DataFrame([json.load(fh)])
    artefact = joblib.load(args.model)
    if isinstance(artefact, dict) and "model" in artefact:
        model, feature_names = artefact["model"], artefact.get("feature_names")
    else:
        model, feature_names = artefact, list(NUMERIC_FEATURES)
    X = prepare_features(record, feature_names)
    prob = float(predict_default_proba(model, X)[0])
    print(json.dumps({
        "default_probability": round(prob, 4),
        "default_prediction": int(decide([prob])[0]),
    }))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credit_scoring", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="group fairness audit over a scored CSV")
    a.add_argument("--input", required=True)
    a.add_argument("--pred-col", default="prediction")
    a.add_argument("--sensitive-col", required=True)
    a.add_argument("--truth-col", default=None)
    a.set_defaults(func=_cmd_audit)

    d = sub.add_parser("drift", help="KS drift between a baseline and a current CSV")
    d.add_argument("--baseline", required=True)
    d.add_argument("--current", required=True)
    d.set_defaults(func=_cmd_drift)

    p = sub.add_parser("predict", help="score a single applicant from a JSON file")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.set_defaults(func=_cmd_predict)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
