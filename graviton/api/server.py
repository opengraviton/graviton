"""Graviton API — headless inference server for AI agents.

No HTML, no static files, no browser. Pure REST API over the Graviton engine.
Designed to be consumed programmatically by AI agents, scripts, and pipelines.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("graviton-api")

app = FastAPI(
    title="Graviton API",
    version="0.1.0",
    description="Headless inference API for AI agents. Load 70B+ models on consumer hardware.",
)


# ── Engine state ────────────────────────────────────────────────────

class _EngineState:
    def __init__(self):
        self.engine = None
        self.model_id: Optional[str] = None
        self.loading: bool = False
        self.load_stage: str = ""
        self.error: Optional[str] = None
        self.gen_lock = threading.Lock()
        self.config_summary: dict = {}

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def reset(self):
        self.engine = None
        self.model_id = None
        self.loading = False
        self.load_stage = ""
        self.error = None
        self.config_summary = {}


state = _EngineState()


# ── Request / response models ──────────────────────────────────────

class LoadRequest(BaseModel):
    model_id: str
    hf_token: str = ""
    bits: float = 4.0
    no_quantize: bool = False
    no_mixed: bool = False
    speculative: bool = False
    spec_tokens: int = 4


class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 256
    system_prompt: str = ""
    history: list = []


# ── Routes ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness check for agents and orchestrators."""
    return {
        "status": "ok",
        "model_loaded": state.loaded,
        "model_id": state.model_id,
    }


@app.post("/api/models/load")
async def load_model(req: LoadRequest):
    """Load a model from HuggingFace. Streams layer-by-layer for 70B+ models."""
    if state.loading:
        raise HTTPException(409, "A model is already loading")

    if state.loaded:
        state.reset()

    state.loading = True
    state.error = None
    state.load_stage = "Initializing..."

    def _load():
        try:
            from graviton.core.config import GravitonConfig, QuantMode
            from graviton.core.engine import GravitonEngine

            if req.hf_token:
                os.environ["HF_TOKEN"] = req.hf_token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = req.hf_token

            state.load_stage = "Building config..."
            config = GravitonConfig(
                model_path=req.model_id,
                quant_bits=req.bits,
                memory=GravitonConfig().memory,
                decoding=GravitonConfig().decoding,
                use_speculative=req.speculative,
            )

            if req.no_quantize:
                config.quantization.mode = QuantMode.NONE
            if req.no_mixed:
                config.quantization.use_mixed_precision = False
            if req.speculative:
                config.decoding.num_speculative_tokens = req.spec_tokens

            state.load_stage = "Creating engine..."
            engine = GravitonEngine(config=config)
            engine.progress_callback = lambda msg: setattr(state, "load_stage", msg)

            state.load_stage = "Downloading & loading weights..."
            engine.load_model()

            if req.no_quantize:
                qlabel = "FP16"
            elif req.no_mixed:
                qlabel = f"INT{int(req.bits)}"
            else:
                qlabel = f"Mixed (critical=8bit, other={int(req.bits)}bit)"

            state.engine = engine
            state.model_id = req.model_id
            state.config_summary = {
                "quantization": qlabel,
                "speculative": req.speculative,
                "bits": req.bits,
            }
        except Exception as exc:
            logger.exception("Model load failed")
            state.error = str(exc)
        finally:
            state.loading = False
            state.load_stage = ""

    threading.Thread(target=_load, daemon=True).start()
    return {"status": "loading", "model_id": req.model_id}


@app.get("/api/models/status")
async def model_status():
    """Check model loading progress. Poll this after /api/models/load."""
    return {
        "loading": state.loading,
        "load_stage": state.load_stage,
        "loaded": state.loaded,
        "model_id": state.model_id,
        "error": state.error,
        "config": state.config_summary,
    }


@app.post("/api/models/unload")
async def unload_model():
    """Unload the current model and free memory."""
    state.reset()
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Send a message and receive a streaming SSE response."""
    if not state.loaded:
        raise HTTPException(400, "No model loaded")

    prompt = _format_prompt(req.system_prompt, req.history, req.message)

    engine = state.engine
    engine.config.decoding.temperature = req.temperature
    engine.config.decoding.max_tokens = req.max_tokens

    def generate():
        token_count = 0
        start = time.time()
        acquired = state.gen_lock.acquire(timeout=30)
        if not acquired:
            yield f"data: {json.dumps({'error': 'Another generation is in progress'})}\n\n"
            return
        try:
            for chunk in engine.generate(prompt, stream=True):
                token_count += 1
                elapsed = time.time() - start
                tps = token_count / max(elapsed, 0.001)
                yield f"data: {json.dumps({'token': chunk, 'tps': round(tps, 1)})}\n\n"

            elapsed = time.time() - start
            tps = token_count / max(elapsed, 0.001)
            yield f"data: {json.dumps({'done': True, 'total_tokens': token_count, 'elapsed': round(elapsed, 2), 'tps': round(tps, 1)})}\n\n"
        except GeneratorExit:
            pass
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.debug("Client disconnected during streaming")
        except Exception as exc:
            try:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        finally:
            state.gen_lock.release()

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Helpers ─────────────────────────────────────────────────────────

def _format_prompt(system: str, history: list, message: str) -> str:
    """Format using ChatML template."""
    parts = []
    sys_text = system or "You are a friendly and helpful assistant."
    parts.append(f"<|system|>\n{sys_text}</s>")
    for msg in history:
        role = msg.get("role", "user")
        parts.append(f"<|{role}|>\n{msg['content']}</s>")
    parts.append(f"<|user|>\n{message}</s>")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)
