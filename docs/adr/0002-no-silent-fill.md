# ADR 0002: Missing model features must fail loudly

## Status
Accepted

## Context
Feature alignment used `DataFrame.reindex(..., fill_value=0)`, which silently
created any absent feature with the value 0. For the bureau scores (normalised
to [0, 1], higher = safer) 0 is the maximum-risk extreme, so applicants whose
upstream integration omitted an optional field were silently scored as if they
had the worst possible credit history.

## Decision
`prepare_features` raises `MissingFeatureError` naming the absent feature(s)
instead of fabricating a value. Complete requests behave unchanged.

## Consequences
Missing data surfaces as a catchable 422 at the API boundary rather than a
silent, systematically pessimistic score.
