"""OPTIONAL DEPLOY - publish the app to a Hugging Face Space (Streamlit SDK). Needs HF_TOKEN in the environment.

  HF_TOKEN=... python scripts/deploy_hf_space.py --repo-id your-user/coverwise-ai

Only fictional / anonymised documents are included. Add LLM keys as Space *secrets* in the HF UI - never in the repo.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = shutil.ignore_patterns(
    ".git", ".github", "__pycache__", ".pytest_cache", ".venv", "tests", "data/real_policies", "data/labelled", ".env", "acko*.csv", "*.log"
)

ap = argparse.ArgumentParser()
ap.add_argument("--repo-id", required=True)
ap.add_argument("--private", action="store_true")
args = ap.parse_args()
token = os.getenv("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN is not set")
from huggingface_hub import HfApi  # noqa: E402  (optional dependency: requirements-deploy.txt)

with tempfile.TemporaryDirectory() as tmp:
    staging = Path(tmp) / "space"
    shutil.copytree(ROOT, staging, ignore=IGNORE)
    shutil.copy(ROOT / "deploy/huggingface/README.md", staging / "README.md")  # Space front matter
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="space", space_sdk="streamlit", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(staging), repo_id=args.repo_id, repo_type="space", commit_message="Deploy CoverWise AI")
print(f"Deployed: https://huggingface.co/spaces/{args.repo_id}")
