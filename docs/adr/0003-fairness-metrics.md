# ADR 0003: Group fairness metrics

## Status
Accepted

## Context
The service makes lending decisions across protected groups (gender, education,
income type, housing). We need to measure whether decisions are equitable, both
at decision time (parity of approvals) and against outcomes (error-rate parity).

## Decision
`credit_scoring.fairness.audit_fairness` reports, per protected group: selection
rate, plus demographic parity difference and the disparate-impact ratio with the
4/5ths-rule check. When ground truth is available it also reports equalized-odds
and equal-opportunity differences. Metrics that are undefined for a group (e.g.
TPR with no positives, disparate impact with an all-zero max rate) are reported
as `nan` rather than silently zeroed.

## Consequences
Fairness is measurable and testable on real decisions; the 4/5ths check gives a
single regulatory pass/fail, while parity and odds differences expose where and
how a model is inequitable.
