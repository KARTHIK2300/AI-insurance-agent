"""Central configuration: paths, labels, disclaimers, thresholds."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PROMPTS_DIR = ROOT / "prompts"
LABELLED_DIR = DATA_DIR / "labelled"
CI_DIR = ROOT / "ci"

LABELS = ["High Risk", "Caution Required", "Moderate Coverage", "Strong Coverage"]

DISCLAIMER = (
    "CoverWise AI is a policy understanding and decision-support assistant. It does not "
    "provide financial, legal, medical or insurance advice and never recommends buying, "
    "switching or cancelling a policy. Extracted values may be incomplete or wrong. Always "
    "confirm every clause with the insurer or a licensed insurance advisor before deciding."
)

# RAG
RAG_MIN_SENTENCE_SCORE = 0.10  # below this the assistant refuses (not found in document)
RAG_TOP_K = 4
CHUNK_MAX_CHARS = 900


# ---------------------------------------------------------------- optional modules (switch on when needed)
OPTIONAL_MODULES = {
    "human_review": "COVERWISE_HUMAN_REVIEW",  # reviewer corrections + labelled-set export
    "premium_benchmark": "COVERWISE_PREMIUM_BENCHMARK",  # practice premium model (synthetic data)
}


def env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def feature_enabled(module: str) -> bool:
    """Optional modules are OFF by default and read from the environment at call time."""
    return env_flag(OPTIONAL_MODULES[module], default=module == "premium_benchmark")
