"""CI/CD smoke test: end-to-end run on the bundled samples; optionally probe a deployed Streamlit URL.

python scripts/smoke_test.py                     # pipeline smoke test
python scripts/smoke_test.py --url http://host:8501   # also check /_stcore/health (post-deploy)
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coverwise.config import LABELS, SAMPLES_DIR  # noqa: E402
from coverwise.guardrails import contains_directive  # noqa: E402
from coverwise.pipeline import analyze_pair, load_policy, sample_paths  # noqa: E402
from coverwise.rag import PolicyIndex, answer_question  # noqa: E402
from coverwise.report import generate_report  # noqa: E402
from coverwise.scoring import ScoringModel, model_status  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url")
ap.add_argument("--wait", type=int, default=60, help="seconds to wait for the URL to become healthy")
args = ap.parse_args()

t0 = time.perf_counter()
cur, new = sample_paths(SAMPLES_DIR)
model = ScoringModel.load()
res = analyze_pair(load_policy("c", cur), load_policy("p", new), model=model, profile={"age": 40})
for key in ("current", "proposed"):
    s = res[key]["score"]
    assert 0 <= s["score"] <= 100 and s["label"] in LABELS, f"bad score for {key}: {s}"
idx = PolicyIndex(res["current"]["doc"])
adv = answer_question("Should I buy this policy?", idx, res["current"]["extraction"], res["current"]["flags"])
assert adv["refused"] and not contains_directive(adv["answer"]), "advice guardrail failed"
assert answer_question("What is the capital of France?", idx)["refused"], "out-of-scope question was answered"
with tempfile.TemporaryDirectory() as d:
    pdf = generate_report(Path(d) / "r.pdf", res, {"age": 40})
    assert pdf.stat().st_size > 5000, "report too small"
print(f"pipeline smoke test OK in {time.perf_counter() - t0:.1f}s | model: {model_status()}")

if args.url:
    import requests

    deadline, last = time.time() + args.wait, None
    while time.time() < deadline:
        try:
            r = requests.get(args.url.rstrip("/") + "/_stcore/health", timeout=5)
            if r.status_code == 200 and r.text.strip() == "ok":
                print("deployed app healthy:", args.url)
                sys.exit(0)
            last = f"{r.status_code} {r.text[:40]}"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(3)
    sys.exit(f"deployed app not healthy: {last}")
