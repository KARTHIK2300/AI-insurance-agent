"""Secret lookup: environment variables first, then (optionally) AWS Secrets Manager.

Set COVERWISE_AWS_SECRET_ID to a secret holding JSON such as {"GEMINI_API_KEY": "..."}; on EC2 use an IAM role so no
keys are stored on disk. boto3 is only needed when this is enabled (requirements-deploy.txt).
"""

from __future__ import annotations

import json
import os

_cache: dict[str, dict] = {}


def get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    secret_id = os.getenv("COVERWISE_AWS_SECRET_ID")
    if not secret_id:
        return None
    if secret_id not in _cache:
        try:
            import boto3

            client = boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
            _cache[secret_id] = json.loads(client.get_secret_value(SecretId=secret_id)["SecretString"])
        except Exception:  # missing boto3 / permissions / bad JSON -> behave as "no secret"
            _cache[secret_id] = {}
    return _cache[secret_id].get(name)
