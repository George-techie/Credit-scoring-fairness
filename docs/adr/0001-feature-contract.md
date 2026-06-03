# ADR 0001: Centralised feature contract

## Status
Accepted

## Context
The mapping between request fields, predictive model features, and protected
demographic attributes was duplicated across the API, the drift monitor and the
retraining pipeline, and drifted out of sync.

## Decision
Define the contract once in `credit_scoring.schema` (predictive numeric
features, bureau-score features and their [0, 1] higher-is-safer scale,
demographic columns, drift features) and have every other module import it.

## Consequences
Feature handling is consistent across serving, monitoring and retraining;
demographic attributes are never fed to the model as predictors.
