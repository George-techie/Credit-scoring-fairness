"""Group fairness metrics for credit decisions.

Given model decisions and a protected attribute (e.g. CODE_GENDER), compute the
standard group-fairness diagnostics: selection rate per group, demographic
parity difference, disparate-impact ratio (the "80% rule"), and — when ground
truth is available — equalized-odds and equal-opportunity differences.

Pure numpy/pandas; no model or network dependency. "Selection" here means the
positive model decision (predicted default = 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupRates:
    group: object
    n: int
    selection_rate: float
    tpr: float  # true-positive rate; nan when the group has no positives
    fpr: float  # false-positive rate; nan when the group has no negatives


@dataclass(frozen=True)
class FairnessReport:
    by_group: tuple
    demographic_parity_difference: float
    disparate_impact_ratio: float
    equalized_odds_difference: float
    equal_opportunity_difference: float

    def group(self, name) -> GroupRates:
        for g in self.by_group:
            if g.group == name:
                return g
        raise KeyError(name)


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else float("nan")


def _group_rates(y_true, y_pred, mask) -> GroupRates:
    yp = y_pred[mask]
    n = int(mask.sum())
    sel = float(np.mean(yp)) if n else float("nan")
    tpr = fpr = float("nan")
    group_label = None
    if y_true is not None:
        yt = y_true[mask]
        pos = yt == 1
        neg = yt == 0
        tpr = _safe_rate(int(((yp == 1) & pos).sum()), int(pos.sum()))
        fpr = _safe_rate(int(((yp == 1) & neg).sum()), int(neg.sum()))
    return GroupRates(group=group_label, n=n, selection_rate=sel, tpr=tpr, fpr=fpr)


def audit_fairness(
    y_pred: Sequence[int],
    sensitive: Sequence,
    y_true: Sequence[int] | None = None,
) -> FairnessReport:
    """Audit decisions for group fairness across the values of ``sensitive``.

    ``y_true`` is optional: without it, only selection-based metrics
    (demographic parity, disparate impact) are populated and the odds-based
    metrics are ``nan``.
    """
    y_pred = np.asarray(y_pred)
    sensitive = pd.Series(np.asarray(sensitive, dtype=object)).reset_index(drop=True)
    yt = None if y_true is None else np.asarray(y_true)

    rates = []
    for value in sorted(sensitive.unique(), key=lambda v: str(v)):
        mask = (sensitive == value).to_numpy()
        gr = _group_rates(yt, y_pred, mask)
        rates.append(GroupRates(value, gr.n, gr.selection_rate, gr.tpr, gr.fpr))

    sel = [r.selection_rate for r in rates if not np.isnan(r.selection_rate)]
    if len(sel) >= 1:
        dpd = float(max(sel) - min(sel))
        di = float(min(sel) / max(sel)) if max(sel) > 0 else float("nan")
    else:
        dpd = di = float("nan")

    def _gap(attr: str) -> float:
        vals = [getattr(r, attr) for r in rates if not np.isnan(getattr(r, attr))]
        return float(max(vals) - min(vals)) if len(vals) >= 1 else float("nan")

    tpr_gap = _gap("tpr")
    fpr_gap = _gap("fpr")
    eo_diff = (
        float(np.nanmax([tpr_gap, fpr_gap]))
        if not (np.isnan(tpr_gap) and np.isnan(fpr_gap))
        else float("nan")
    )

    return FairnessReport(
        by_group=tuple(rates),
        demographic_parity_difference=dpd,
        disparate_impact_ratio=di,
        equalized_odds_difference=eo_diff,
        equal_opportunity_difference=tpr_gap,
    )


def passes_four_fifths_rule(report: FairnessReport, threshold: float = 0.8) -> bool:
    """The disparate-impact ratio meets the regulatory 4/5ths guideline."""
    di = report.disparate_impact_ratio
    return (not np.isnan(di)) and di >= threshold
