# Contributing

## Workflow
1. Branch from `develop` (`feat/...`, `fix/...`, `data/...`). Never commit to `main` directly.
2. Run `make ci` locally - it executes the same steps as the GitHub pipeline (lint, hygiene, tests, data validation, training, evaluation, quality gates, packaging, smoke test).
3. Open a pull request. Required status checks (set in branch protection): `code-quality` and `ml-pipeline`.
4. Releases: merge to `main`, then tag `vMAJOR.MINOR.PATCH`. The `Release` workflow re-trains, re-runs the gates, publishes the versioned model + model card + container image.

## Commit messages
`type(scope): summary` (<= 72 chars) followed by a body that says **what changed and why**, plus any metric impact for model/data changes.
Types: feat, fix, data, model, ci, docs, test, chore, refactor. Example:

    model(score): ship gradient-boosting score regressor with score bands

    Out-of-fold macro-F1 0.798 vs 0.793 for the rule baseline; tree classifiers overfit (0.42-0.55).
    risk_flags_count excluded (derived feature, leakage). High Risk recall 0.63 is the weak spot.

## Changing the model or data
* Any change to `data/raw`, `data/labelled`, `src/coverwise/scoring.py`, `features.py` or `scripts/train_model.py` re-runs the quality gates in CI.
* If metrics move on purpose, review `reports/quality_gate_result.json`, then promote the baseline in the same PR:
  `python scripts/ci_quality_gate.py --update-baseline` (refuses to promote a model that fails the gates).
* Gate thresholds live in `ci/quality_gates.json` - loosening them needs a reviewer's explicit approval.

## Rules that are never negotiable
* No API keys or `.env` in git (CI scans for them). Use environment variables or AWS Secrets Manager.
* Only fictional or anonymised policy documents in the repo; never real policyholder data.
* The product never tells a user to buy, switch, renew or cancel a policy (CI safety gates enforce this).
