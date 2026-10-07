"""Thin LLM client for Gemini and Groq over REST (no SDK needed). Keys come from environment variables.

Model names change over time - set GEMINI_MODEL / GROQ_MODEL in .env to a currently available model.
"""

from __future__ import annotations

import json
import os
import re
import time

import requests

from .secrets import get_secret


class LLMError(RuntimeError):
    pass


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise LLMError("model did not return valid JSON")


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str | None = None, session=None, timeout: int = 60):
        provider = provider.lower()
        if provider not in {"gemini", "groq"}:
            raise ValueError("provider must be 'gemini' or 'groq'")
        self.provider, self.api_key, self.timeout = provider, api_key, timeout
        self.model = model or (
            os.getenv("GEMINI_MODEL", "gemini-2.0-flash") if provider == "gemini" else os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        )
        self.session = session or requests

    # -- raw call -----------------------------------------------------------------
    def _post(self, system: str, prompt: str, json_mode: bool) -> str:
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, **({"responseMimeType": "application/json"} if json_mode else {})},
            }
            headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        else:
            url = "https://api.groq.com/openai/v1/chat/completions"
            body = {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last = None
        for attempt in range(3):
            try:
                r = self.session.post(url, headers=headers, json=body, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(1.5 * (attempt + 1))
                    last = f"HTTP {r.status_code}"
                    continue
                r.raise_for_status()
                data = r.json()
                if self.provider == "gemini":
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                return data["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                last = str(exc)
                time.sleep(1.0 * (attempt + 1))
        raise LLMError(f"{self.provider} request failed: {last}")

    def generate_text(self, system: str, prompt: str) -> str:
        return self._post(system, prompt, json_mode=False).strip()

    def generate_json(self, system: str, prompt: str) -> dict:
        return _parse_json(self._post(system, prompt, json_mode=True))


def get_llm(provider: str | None = None, api_key: str | None = None, model: str | None = None) -> LLMClient | None:
    """Build a client from arguments or env / AWS secrets (LLM_PROVIDER, GEMINI_API_KEY, GROQ_API_KEY). None if no key."""
    provider = (provider or get_secret("LLM_PROVIDER") or "").lower()
    if not provider:
        provider = "gemini" if get_secret("GEMINI_API_KEY") else "groq" if get_secret("GROQ_API_KEY") else ""
    if not provider or provider == "none":
        return None
    key = api_key or get_secret("GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY")
    return LLMClient(provider, key, model) if key else None
