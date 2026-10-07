"""CI step: no committed secrets, no real PII in bundled documents, no oversized / forbidden files. Exit 1 on findings."""

import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coverwise.pdf_extract import _PII_RULES  # noqa: E402

SECRET_PATTERNS = {
    "Google API key": r"AIza[0-9A-Za-z_\-]{35}",
    "Groq key": r"gsk_[A-Za-z0-9]{20,}",
    "AWS access key": r"AKIA[0-9A-Z]{16}",
    "HF token": r"hf_[A-Za-z0-9]{30,}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "Private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "Hard-coded key assignment": r"(?i)(api[_-]?key|secret|token)\s*=\s*['\"][A-Za-z0-9_\-]{24,}['\"]",
}
TEXT_EXT = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".csv", ".sh", ".cfg", ".ini", ".env", ".example", ".jsonl", ".js"}
PII_SCAN_DIRS = ("data/samples", "data/eval")
MAX_BYTES = 5 * 1024 * 1024


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        files = [ROOT / f for f in out if f]
        if files:
            return files
    except Exception:
        pass
    skip = {".git", "__pycache__", ".venv", ".pytest_cache"}
    return [p for p in ROOT.rglob("*") if p.is_file() and not (skip & set(p.relative_to(ROOT).parts))]


def pdf_text(path: Path) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(path.read_bytes())) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


def main() -> int:
    allow = [a.strip() for a in (ROOT / "ci/pii_allowlist.txt").read_text().splitlines() if a.strip()]
    findings = []
    for f in tracked_files():
        rel = f.relative_to(ROOT).as_posix()
        if rel == "scripts/check_repo_hygiene.py":
            continue
        if rel == ".env" or re.search(r"(^|/)acko.*\.csv$", rel):
            findings.append((rel, "forbidden file tracked"))
        if f.stat().st_size > MAX_BYTES:
            findings.append((rel, f"file larger than {MAX_BYTES // 2**20} MB"))
        text = ""
        if f.suffix.lower() in TEXT_EXT or f.name in {".gitignore", "Dockerfile", "Makefile"}:
            text = f.read_text(errors="ignore")
            for name, pat in SECRET_PATTERNS.items():
                if re.search(pat, text) and not rel.startswith("tests/"):
                    findings.append((rel, f"possible secret: {name}"))
        if rel.startswith(PII_SCAN_DIRS):
            blob = pdf_text(f) if f.suffix.lower() == ".pdf" else text
            for pat, _ in _PII_RULES[:4]:  # email, phone, aadhaar, PAN
                for m in pat.finditer(blob):
                    if not any(a in m.group(0) for a in allow):
                        findings.append((rel, f"possible real PII: {m.group(0)[:20]}"))
    print(
        json.dumps([{"file": a, "issue": b} for a, b in findings], indent=2)
        if findings
        else "hygiene OK: no secrets, forbidden files or unlisted PII"
    )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
