"""Entry-point: ``graviton-api`` CLI — headless inference server for AI agents."""

from __future__ import annotations

# MPS memory limit bypass (for large models on Apple Silicon)
import os
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import signal
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="graviton-api",
        description="Start the Graviton inference API server (headless, for agents).",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to serve on (default: 7860)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    if sys.platform != "win32":
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    url = f"http://{args.host}:{args.port}"
    print(
        f"\n"
        f"  \033[1;35mGraviton API\033[0m ready at \033[1;36m{url}\033[0m\n"
        f"\n"
        f"  Endpoints:\n"
        f"    GET  /health                Liveness check\n"
        f"    POST /api/models/load       Load a model\n"
        f"         body: {'{'}\"model_id\": \"Qwen/Qwen2.5-72B-Instruct\", \"bits\": 4{'}'}\n"
        f"    GET  /api/models/status      Check loading progress\n"
        f"    POST /api/chat               Send a message (SSE stream)\n"
        f"         body: {'{'}\"message\": \"Hello\", \"temperature\": 0.7{'}'}\n"
        f"    POST /api/models/unload      Unload the current model\n"
        f"\n"
        f"  Example:\n"
        f"    curl -X POST {url}/api/models/load \\\n"
        f"      -H 'Content-Type: application/json' \\\n"
        f"      -d '{'{'}\"model_id\": \"TinyLlama/TinyLlama-1.1B-Chat-v1.0\", \"bits\": 4{'}'}'\n"
        f"\n"
    )

    import uvicorn
    uvicorn.run(
        "graviton.api.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
