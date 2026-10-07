import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from coverwise.config import SAMPLES_DIR  # noqa: E402
from coverwise.pipeline import analyze, load_policy, sample_paths  # noqa: E402
from coverwise.sample_policies import generate_samples  # noqa: E402
from coverwise.scoring import ScoringModel  # noqa: E402


@pytest.fixture(scope="session")
def samples():
    cur, new = sample_paths(SAMPLES_DIR)
    if not cur.exists():
        generate_samples(SAMPLES_DIR)
    return cur, new


@pytest.fixture(scope="session")
def analyses(samples):
    model = ScoringModel.load()
    return {
        "current": analyze(load_policy("Current", samples[0]), model=model, profile={"age": 62, "planning_child": True, "n_children": 3}),
        "proposed": analyze(load_policy("Proposed", samples[1]), model=model, profile={"age": 62, "planning_child": True, "n_children": 3}),
    }
