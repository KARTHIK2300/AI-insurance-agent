# Model card - CoverWise coverage clarity scorer v1.0.0

**Intended use.** Estimate how clear / restrictive an Indian health-insurance policy's wording looks (score 0-100 and one of
four labels) to support *human* review. It is **not** a claim-settlement predictor and must not be used to tell anyone to
buy, switch or cancel a policy.

**Method.** Gradient Boosting regressor -> score bands. Reasons shown to users come from a transparent linear rule set, not from the ML model.
`risk_flags_count` is excluded from the features because it is derived from the other features (leakage).

**Training data.** 480 rows (0 human-reviewed real rows), mostly a *synthetic* dataset whose
score is a function of its own features; the model learns that recipe, not real claim outcomes. Data hash `2a93f15deec97bcd`.

**Evaluation (5-fold out-of-fold).** macro-F1 0.798,
accuracy 0.812, score MAE 4.27, R2 0.922,
High Risk recall 0.63 (weakest class - treat borderline scores as "needs advisor review").

**Quality gates.** passed (see `reports/quality_gate_result.json`).

**Provenance.** git `090fb5bb786d70245c5669e7806ac5fa47e12c2a`, scikit-learn 1.8.0,
Python 3.12.3, artifact sha256 `6715846667c76f4d`.

**Known limitations.** Small data; tree models overfit; rules tuned on two fictional policies; real documents may need the LLM path or human review.
