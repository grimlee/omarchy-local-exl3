#!/usr/bin/env python3
"""Short real GPU smoke for both profiles; no long-context benchmark."""

import base64
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin/local-exl3"


def call(path, body=None):
    request = Request("http://127.0.0.1:8881" + path,
                      data=json.dumps(body).encode() if body is not None else None,
                      headers={"Content-Type": "application/json"} if body is not None else {},
                      method="POST" if body is not None else "GET")
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main(image_path):
    image = Path(image_path).expanduser().resolve()
    if not image.is_file():
        raise SystemExit("Provide a small existing JPEG or PNG image path")
    mime = mimetypes.guess_type(image.name)[0]
    if mime not in ("image/jpeg", "image/png"):
        raise SystemExit("Image must be JPEG or PNG")
    encoded = base64.b64encode(image.read_bytes()).decode()
    image_url = f"data:{mime};base64,{encoded}"
    for profile, expected_context, expected_kv in (("balanced", 117760, 4), ("max-context", 150016, 3)):
        print(f"Starting {profile}...", flush=True)
        subprocess.run([str(CLI), "start", "--profile", profile], check=True)
        try:
            status = json.loads(subprocess.check_output([str(CLI), "status", "--json"], text=True))
            assert status["health"] and status["stable_mm"] and status["context"] == expected_context
            assert status["kv_k_bits"] == status["kv_v_bits"] == expected_kv
            models = call("/v1/models")
            assert any(row.get("id") == "exllama-v3" for row in models.get("data", []))
            text = call("/v1/chat/completions", {"model": "exllama-v3", "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 8})
            assert text.get("choices"), "No text response"
            picture = call("/v1/chat/completions", {"model": "exllama-v3", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this image briefly."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}], "max_tokens": 32})
            assert picture.get("choices"), "No image response"
            print(f"{profile}: /v1/models, text, image PASS", flush=True)
        finally:
            subprocess.run([str(CLI), "stop"], check=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/smoke.py /path/to/small-test-image.jpg")
    main(sys.argv[1])
