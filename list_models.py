"""List the chat models your key can use for LLM_PROVIDER, so you can set GEN_MODEL / JUDGE_MODEL in .env."""
import requests

import config

if config.LLM_PROVIDER == "nebius":
    key = config.require("NEBIUS_API_KEY")
    resp = requests.get(config.NEBIUS_BASE_URL.rstrip("/") + "/models",
                        headers={"Authorization": f"Bearer {key}"}, timeout=30)
    resp.raise_for_status()
    for m in sorted(resp.json().get("data", []), key=lambda m: m["id"]):
        print(m["id"])
else:
    key = config.require("GOOGLE_API_KEY")
    resp = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                        headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=30)
    resp.raise_for_status()
    for m in resp.json().get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            print(m["name"].removeprefix("models/"))
