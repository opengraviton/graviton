#!/usr/bin/env python3
"""
Test Graviton API chat: load a model (if needed), send "hello how are you",
collect streaming response, and verify a coherent answer.

To get correct responses (especially Mistral/Mixtral):
  1. Install latest engine:  cd graviton && pip install -e ".[api,huggingface]"
  2. Start API:  python -m uvicorn graviton.api.server:app --host 127.0.0.1 --port 7862
  3. Run this script:  python graviton/scripts/test_api_chat.py --port 7862 [--model MODEL]

Usage:
  python graviton/scripts/test_api_chat.py [--port 7862] [--model TinyLlama/...]
"""
import argparse
import json
import re
import sys
import time

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

DEFAULT_PORT = 7860
LOAD_POLL_INTERVAL = 2
LOAD_TIMEOUT = 600  # 10 min for large models
CHAT_TIMEOUT = 120


def main():
    p = argparse.ArgumentParser(description="Test Graviton API chat")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="API port")
    p.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model to load if none loaded (default: TinyLlama for fast test)",
    )
    p.add_argument("--message", default="hello how are you", help="Message to send")
    p.add_argument("--max-tokens", type=int, default=128, help="Max tokens")
    args = p.parse_args()

    base = f"http://127.0.0.1:{args.port}"

    # Health
    try:
        r = requests.get(f"{base}/health", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"API not reachable at {base}: {e}", file=sys.stderr)
        print("Start the API first: graviton-api", file=sys.stderr)
        sys.exit(1)

    data = r.json()
    if not data.get("model_loaded"):
        print(f"No model loaded. Loading {args.model}...")
        load_r = requests.post(
            f"{base}/api/models/load",
            json={"model_id": args.model, "bits": 4},
            timeout=10,
        )
        load_r.raise_for_status()
        if load_r.json().get("status") != "loading":
            print("Load did not return status=loading", file=sys.stderr)
            sys.exit(1)
        deadline = time.monotonic() + LOAD_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(LOAD_POLL_INTERVAL)
            st = requests.get(f"{base}/api/models/status", timeout=5).json()
            if st.get("error"):
                print(f"Load failed: {st['error']}", file=sys.stderr)
                sys.exit(1)
            if st.get("loaded"):
                print(f"Model loaded: {st.get('model_id')}")
                break
        else:
            print("Load timed out", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Using already loaded model: {data.get('model_id')}")

    # Chat (streaming)
    print(f"Sending: {args.message!r}")
    resp = requests.post(
        f"{base}/api/chat",
        json={
            "message": args.message,
            "temperature": 0.7,
            "max_tokens": args.max_tokens,
            "system_prompt": "You are a helpful assistant.",
        },
        stream=True,
        timeout=CHAT_TIMEOUT,
    )
    resp.raise_for_status()

    full_text = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if "token" in data:
            full_text.append(data["token"])
        if data.get("done"):
            print(f"Tokens: {data.get('total_tokens')}, TPS: {data.get('tps')}")
        if "error" in data:
            print(f"API error: {data['error']}", file=sys.stderr)
            sys.exit(1)

    response = "".join(full_text).strip()
    print("\n--- Response ---")
    print(response)
    print("--- End ---\n")

    if not response:
        print("FAIL: empty response", file=sys.stderr)
        sys.exit(1)
    if len(response) < 5:
        print("FAIL: response too short", file=sys.stderr)
        sys.exit(1)
    # Expect a coherent reply: mostly normal words, not code/gibberish
    ascii_ok = sum(
        1 for c in response
        if c.isascii() and (c.isalpha() or c.isspace() or c in ".,!?'-")
    )
    ratio = ascii_ok / len(response) if response else 0
    if len(response) > 30 and ratio < 0.65:
        print(
            "FAIL: response is gibberish (too many non-ASCII/symbols).",
            file=sys.stderr,
        )
        sys.exit(1)
    # Greeting reply should contain at least one normal reply word
    reply_words = {"good", "well", "fine", "great", "thanks", "thank", "hello", "hi", "doing", "help", "you", "how", "am", "are", "i'm", "i am"}
    words_lower = set(re.split(r"\W+", response.lower()))
    if len(response) > 20 and not (words_lower & reply_words):
        print(
            "FAIL: response does not look like a greeting reply (gibberish).",
            file=sys.stderr,
        )
        sys.exit(1)
    from collections import Counter
    words = [w.strip().lower() for w in response.split() if len(w.strip()) > 1]
    if len(words) >= 8:
        cnt = Counter(words)
        top_count = cnt.most_common(1)[0][1] if cnt else 0
        if top_count > max(8, len(words) // 3):
            print("FAIL: response looks repetitive/gibberish.", file=sys.stderr)
            sys.exit(1)

    print("OK: got a coherent response.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
