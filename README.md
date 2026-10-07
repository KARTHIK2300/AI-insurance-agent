# CoverWise AI - Coverage Insight and Risk Intelligence System

AI-powered insurance policy decoder, coverage comparison and risk explanation assistant (GUVI x HCL capstone).
Upload one or two policy PDFs -> extract clauses with page evidence -> plain-English summary -> Coverage Clarity Score and
risk level -> current-vs-proposed comparison -> grounded Q&A with citations -> advisor questions -> downloadable PDF report.

> **Responsible AI.** CoverWise is a policy *understanding and decision-support* tool. It never says "buy" or "don't buy",
> refuses advice-style questions, declines questions the document does not answer, and always points to the insurer / a
> licensed advisor for the final decision. Scores describe how restrictive or unclear the wording looks - they do **not**
> predict claim settlement.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install tesseract-ocr poppler-utils      # only needed for scanned PDFs (OCR)
python -m coverwise.sample_policies                   # (run from ./src) regenerates the 2 fictional sample PDFs
python scripts/train_model.py                         # trains + evaluates the risk scorer -> models/, reports/figures/
python scripts/evaluate.py                            # extraction / Q&A / safety evaluation -> reports/
pytest                                                # 41 tests
make ci                                               # the full CI/CD pipeline locally (lint, tests, gates, packaging, smoke test)
streamlit run app/streamlit_app.py                    # tick "Use bundled fictional sample policies" to try it instantly
```
The app works **fully offline** (rule-based extraction + TF-IDF retrieval). Add an LLM to improve extraction and answers:
copy `.env.example` to `.env` and set `GEMINI_API_KEY` or `GROQ_API_KEY` (or paste a key in the sidebar).

## How it works
| Stage | Module | What it does |
|---|---|---|
| Ingest | `pdf_extract.py` | pdfplumber text, OCR fallback per page, header/footer/page-number removal, **PII masking** |
| Structure | `document.py` | section detection (coverage, exclusions, waiting, limits, claims, renewal), sentence units, RAG chunks with page / section metadata |
| Extract | `extractor.py` | ~35 fields via transparent rules; optional Gemini/Groq JSON extraction. LLM values are accepted **only if their supporting quote is found in the document**; rules win on conflict; values are range-validated |
| Features | `features.py` | the brief's feature set (premium ratio, waiting periods, co-pay, caps, sub-limits, exclusions, restoration, NCB, child/maternity, claim clarity, ambiguity, citation coverage). Missing clauses are imputed conservatively **and reported as unclear** |
| Score | `scoring.py`, `scripts/train_model.py` | rule baseline (weights rounded from a linear fit, fully explainable) + gradient-boosting score model; label from score bands; reasons from the transparent rules |
| Compare | `comparison.py` | improved / worse / unclear per item, premium-vs-coverage, switching (waiting-period) risks, profile-specific cautions |
| Q&A | `rag.py` | TF-IDF retrieval at sentence level, topic gating, term-coverage check, citations (page + section), refusal on unsupported or advice questions; optional LLM answer |
| Report | `report.py`, `summary.py`, `risk.py` | 10-20 line summary, flags, advisor questions, limitations note, PDF |
| UI | `app/streamlit_app.py` | upload, profile, tabs for score, clauses, comparison, Q&A, report, model/eval |

## CI/CD for the ML model
The scoring model is a versioned, gated artifact: nothing is merged, packaged or released unless it passes the same checks the pipeline runs.

```
push / PR ──► code-quality ───────────────► ml-pipeline ──────────────────────────────────────────────► artifact
              ruff (PEP 8) · hygiene          validate data → train → evaluate → QUALITY GATES → package (MANIFEST + MODEL_CARD) → smoke test
              (secrets, PII, files) · pytest
tag vX.Y.Z ─► release: same gates ─► GitHub Release (model + manifest + card) ─► build image ─► smoke-test container ─► push to GHCR
manual     ─► deploy (optional): Hugging Face Space  |  AWS EC2 over SSH ─► post-deploy health check
runtime    ─► app verifies model hash + feature schema + "gates passed" before loading it; otherwise falls back to the rule-based scorer
```

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | push, PR | lint, hygiene, tests; then data validation, training, evaluation, quality gates, packaging, smoke test; uploads the model bundle |
| `.github/workflows/release.yml` | tag `v*.*.*` | re-runs gates, publishes versioned model assets, builds and smoke-tests the container, pushes it to GHCR |
| `.github/workflows/deploy.yml` | manual | optional deploy to Hugging Face Spaces or AWS EC2, protected by the `production` environment |

**Quality gates** (`ci/quality_gates.json`, failing any one blocks the merge / release):
macro-F1 >= 0.75, accuracy >= 0.77, score MAE <= 5.5, R2 >= 0.88, High Risk recall >= 0.55; the leaky `risk_flags_count` feature must not be used;
no regression vs `ci/baseline_metrics.json` (macro-F1 drop <= 0.03, MAE rise <= 0.5); extraction accuracy >= 0.95; OCR accuracy >= 0.90;
Q&A correctness and citation accuracy >= 0.95; **0** hallucinated answers to unsupported questions; **0** buy/switch language; advice questions refused 100%.

**Other pipeline safeguards:** schema / range / label / class-size validation of the training data; deterministic training (identical model hash on every run);
`MANIFEST.json` (version, git SHA, data hash, sklearn version, metrics, gate result, SHA-256) and `MODEL_CARD.md`; the packager refuses to run when gates fail;
a model that is tampered with, mismatches the feature schema or was never gated is rejected at load time.

**Run it locally:** `make ci`. **Promote new baseline metrics:** `python scripts/ci_quality_gate.py --update-baseline`. See `CONTRIBUTING.md` for the branch and release flow.

## Optional modules (switch on when needed)
All are off by default and do not affect the core app.

| Module | How to switch on | What it gives you |
|---|---|---|
| **Human review** | sidebar toggle, or `COVERWISE_HUMAN_REVIEW=1` | a Human review tab: confirm or correct 16 extracted fields, re-score instantly, record a final label, save to `data/labelled/` (document hash + numeric features only, no text); shows real-world field-correction rates |
| **Larger real labelled set** | `python scripts/train_model.py --include-labelled` (needs 20+ reviews) | retrains on reviewed real policies plus the synthetic data; the quality gates and baseline check still apply |
| **Real-policy evaluation** | `python scripts/evaluate_real_policies.py --pdf-dir data/real_policies [--truth truth.csv] [--llm gemini]` | per-PDF field coverage, unclear clauses, OCR use, timing, Q&A safety, accuracy against your truth file, and a "needs human review" flag |
| **Docker** | `docker compose up --build` (copy `.env.example` to `.env`) | non-root image with health check; secrets are never baked in |
| **AWS EC2** | `IMAGE=ghcr.io/<owner>/<repo>:1.0.0 SECRET_ID=coverwise/prod bash deploy/ec2_setup.sh` or the `deploy` workflow | runs the container with an IAM role; keys read from AWS Secrets Manager (`COVERWISE_AWS_SECRET_ID`), no keys on disk |
| **Hugging Face Space** | `HF_TOKEN=... python scripts/deploy_hf_space.py --repo-id user/coverwise-ai` or the `deploy` workflow | Streamlit Space with fictional samples only; add LLM keys as Space secrets |
| **LLM extraction / answers** | `.env` keys or the sidebar | Gemini / Groq on top of the offline rules (grounded quotes only) |
| **Premium benchmark** | sidebar toggle, `COVERWISE_PREMIUM_BENCHMARK=1` | practice premium model trained on the Acko-style data |

## Results (all reproducible with the scripts above)
**Risk scoring** - synthetic 500-row dataset, cleaned to 480 rows (20 implausible maternity rows dropped; 165 rows with PED < initial waiting are flagged and kept). 5-fold out-of-fold:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Rule baseline (weights fitted on the same rows - optimistic) | 0.817 | 0.793 |
| **Production: Gradient-boosting score regressor -> score bands** (MAE 4.3, R2 0.92) | **0.813** | **0.798** |
| Logistic Regression (second opinion / probabilities) | 0.762 | 0.730 |
| Gradient Boosting / Random Forest / Decision Tree classifiers | 0.650 / 0.625 / 0.473 | 0.550 / 0.473 / 0.424 |
| Logistic Regression **with** `risk_flags_count` (leakage demo) | 0.840 | 0.806 |

* ML did **not** clearly beat the transparent rules on 480 rows; the tree classifiers overfit. We ship the regressor + bands because it keeps score and label consistent and generalises at least as well, and keep rules for explanations.
* `risk_flags_count` is derived from the other features (R2 0.84, corr -0.88 with the score), so the production model excludes it.
* Weak spot: **High Risk recall is 0.63** (30 rows); 11 of 30 are predicted "Caution Required". Treat borderline scores as "needs advisor review".

**Pipeline evaluation on the two fictional sample policies** (`reports/evaluation_results.json`):
extraction field accuracy 100% on text PDFs and 96% on a scanned/OCR copy (OCR loses the numbered exclusion list: 6 of 14 found);
26 policy-specific questions: 100% correct, 100% correct page citation, 3/3 unsupported questions refused; 0 directive ("buy/switch") language;
all 12 unique questions of the provided 60-row test set answered with a citation or an honest "not clearly found"; advice questions refused 100%.
**Caveat:** the rules were developed on these same two documents, so this is a regression benchmark, not a generalisation estimate. Test on real
HDFC ERGO / Star / Care / Niva Bupa documents before quoting accuracy.

**Optional premium benchmark** (`scripts/train_premium_model.py`, Acko/Kaggle-style data, 300k labelled rows, log target): holdout RMSLE 0.934 vs 1.089 for a mean-only baseline.
The data is synthetic with weak signal and has no policy clauses, so it is a practice benchmark only and is not used for scoring.

## Dataset resource links
* Resource pack (synthetic coverage dataset + Q&A test set): https://drive.google.com/file/d/1csdcu8VjGlw_4hiRhCP9T51tBxysIzaC/view?usp=drive_link
* Kaggle - Insurance Premium Forecasting Challenge: https://www.kaggle.com/competitions/insurance-premium-forecasting-challenge/data (invitation may be required)
* Kaggle - Regression with an Insurance Dataset: https://www.kaggle.com/competitions/regression-with-an-insurance-dataset
* Policy documents: IRDAI https://irdai.gov.in/health-insurance-products - HDFC ERGO https://www.hdfcergo.com/download/policy-wordings/health -
  Star Health https://www.starhealth.in/downloads/ - Care Health https://www.careinsurance.com/other-downloads.html -
  Niva Bupa https://transaction.nivabupa.com/pages/downloads.aspx
* Optional Q&A: https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset , https://huggingface.co/datasets/deccan-ai/insuranceQA-v2

The 219 MB Acko CSV is not in the repo (`.gitignore`). Only anonymised / fictional documents are bundled. **Never upload real policyholder documents to a public demo.**

## Repository layout
```
app/streamlit_app.py      dashboard                 .github/workflows/   ci.yml, release.yml, deploy.yml
src/coverwise/            library (ingest, extract, score, compare, rag, report, review, mlops, secrets)
scripts/                  prepare_data, validate_data, train_model, evaluate, ci_quality_gate, package_model,
                          check_repo_hygiene, smoke_test, evaluate_real_policies, train_premium_model, deploy_hf_space
ci/                       quality_gates.json, baseline_metrics.json, pii_allowlist.txt
models/                   coverwise_model.joblib + MANIFEST.json + MODEL_CARD.md + metrics.json
data/raw|processed        datasets      data/samples   fictional sample PDFs      data/eval   ground truth + Q&A sheet
data/labelled|real_policies   optional-module data (git-ignored contents)
deploy/                   ec2_setup.sh, huggingface/README.md       Dockerfile, docker-compose.yml, Makefile
notebooks/                profiling + model review (executed)       prompts/   LLM prompts
reports/                  sample PDF reports, evaluation + gate results, figures      tests/   41 pytest tests
```

## Configuration notes
* LLM calls use plain REST (`src/coverwise/llm.py`). Model names change over time - set `GEMINI_MODEL` / `GROQ_MODEL` to a currently available model.
* Embedding retrieval (`PolicyIndex(doc, backend="embeddings")`) needs `sentence-transformers`; default is TF-IDF.
* Keep API keys in environment variables / secrets only.

## Deployment
See "Optional modules" above. Keep secrets in environment variables, GitHub Actions secrets, Hugging Face Space secrets or AWS Secrets Manager - never in the repo.
For EC2: expose only the app port (or 443 behind Nginx), and stop unused instances to avoid cost.

## Honest limitations
* The live Gemini / Groq code paths are unit-tested with mocked responses only - they have not been run against the real APIs here.
* The GitHub Actions workflows were validated as YAML and every step was executed locally through `make ci` and individual scripts, but they have **not** run on GitHub.
  Docker build / push, the Hugging Face and EC2 deployments and AWS Secrets Manager against a real account were not run (no Docker or cloud access here); the secret lookup is tested with a fake boto3 client.
  Third-party action versions (checkout, setup-python, docker/*, softprops, appleboy) should be checked against their current releases.
* Rule extraction is tuned to common Indian health-policy phrasing; unusual layouts, tables and multi-column PDFs will need more rules or the LLM path.
* Real-world waiting-period / portability rules (IRDAI) change; switching-risk text is generic guidance to verify, not legal advice.
* The synthetic training data encodes its own scoring recipe; the model learns that recipe, not real claim outcomes.
