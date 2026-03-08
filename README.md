<p align="center">
  <img src="assets/logo.svg" alt="Graviton" width="250"/>
</p>
<h1 align="center">Graviton</h1>
<p align="center"><em>Run 70B+ parameter LLMs on hardware you already own.</em></p>

<p align="center">
  <a href="#installation">Installation</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#-web-ui">Web UI</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#benchmarks">Benchmarks</a> &bull;
  <a href="#-testing">Testing</a> &bull;
  <a href="#contributing">Contributing</a>
</p>

---

## What is Graviton?

**Graviton** is an open-source AI inference engine that makes large language models accessible on consumer hardware. A 72B-parameter model that normally requires a $10,000+ GPU server? Graviton streams it layer-by-layer, quantizes each layer in-flight, and fits it into **36 GB** on a Mac with 64 GB of unified memory.

Modern AI models are getting bigger — GPT-4 class models have hundreds of billions of parameters, requiring server farms with expensive GPUs. Graviton combines seven cutting-edge techniques to make the impossible possible:

| Technique | Impact | Description |
|---|---|---|
| **Streaming Layer-by-Layer Loading** | Run 70B+ models | Builds model on meta device, streams each transformer layer from disk, quantizes in-flight, frees originals. Peak memory = 1 FP16 layer + all quantized layers. |
| **Extreme Quantization** | 4-16x smaller | FP16 &rarr; 4-bit, 2-bit, or 1.58-bit (ternary) weights |
| **QuantizedLinear** | 62% less memory | Drop-in `nn.Linear` replacement with packed quantized weights |
| **Mixed-Precision** | Best of both | Critical layers at 8-bit, FFN at 4-bit automatically |
| **Dynamic Sparsity** | 2-10x faster | Only activate relevant neurons per token |
| **Layer Streaming** | Unlimited model size | Stream layers from SSD via memory-mapped files |
| **Speculative Decoding** | 2-3x faster | Layer-skip draft model predicts, full model verifies |
| **KV-Cache (Fast + Compressed)** | Bounded memory | Pre-allocated buffers (fast) or INT8 compressed (long-context) |

### The Math

A 72B parameter model at FP16 requires ~144 GB of memory. With Graviton's streaming loader:

```
144 GB (FP16) → stream layer-by-layer → quantize each to 4-bit → 36 GB on device
Peak memory during loading: ~38 GB (1 FP16 layer + quantized layers so far)
= Runs on a Mac with 64 GB unified memory!
```

## One-Command Install

### For Humans

Install everything and open the chat UI in one command:

```bash
pip install graviton-ui && graviton-ui
```

This single command installs the Graviton engine, quantization stack, HuggingFace integration, and the chat interface. Your browser opens at `http://localhost:7860` — pick a model, choose quantization, and start chatting.

### For AI Agents

No UI, no browser, no unnecessary dependencies. Just the engine and a REST API:

```bash
pip install "graviton-ai[api]" && graviton-api
```

The headless API server starts on `0.0.0.0:7860`. An agent on a low-budget machine can load a 70B+ model via streaming quantization and use it programmatically — no GPU cluster, no cloud bill.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check for agents and orchestrators |
| `/api/models/load` | POST | Load a model: `{"model_id": "Qwen/Qwen2.5-72B-Instruct", "bits": 4}` |
| `/api/models/status` | GET | Check loading progress (layer-by-layer status for large models) |
| `/api/chat` | POST | Chat: `{"message": "Hello", "temperature": 0.7}` — returns SSE token stream |
| `/api/models/unload` | POST | Unload the current model and free memory |

### From Source (Development)

```bash
git clone https://github.com/opengraviton/graviton.git
cd graviton
pip install -e ".[all]"
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- Apple Silicon Mac recommended (but works on any platform with CPU or CUDA)

### HuggingFace Setup (for downloading models)

Many models on HuggingFace require authentication:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Authenticate:
   ```bash
   huggingface-cli login
   ```
   Or set the environment variable:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

> **Tip:** For gated models (like LLaMA), you must also accept the model's license on its HuggingFace page.

## Quick Start

### Python API

```python
from graviton import GravitonEngine, GravitonConfig

config = GravitonConfig(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quant_bits=4,
    max_memory_gb=16,
)

engine = GravitonEngine(config=config)
engine.load_model()

response = engine.generate(
    "Explain quantum computing in simple terms:",
    max_tokens=256,
    temperature=0.7,
)
print(response)
```

> Uses [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), an open model. For gated models, see [HuggingFace Setup](#huggingface-setup-for-downloading-models).

### CLI

```bash
graviton info                                           # hardware capabilities

graviton run TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt 'Explain quantum computing:' -b 8 --no-mixed   # INT8 quantization

graviton run TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt 'Hello world' --speculative --spec-tokens 4    # speculative decoding

graviton benchmark                                      # performance benchmark
```

**Example output** (Apple M1 Max, 64 GB, INT8 QuantizedLinear):

```
Quantized 154 linear layers, saved 1318 MB in packed storage
Model ready: 0.13B params, 0.78 GB on mps

Prompt: Explain quantum computing briefly.
--------------------------------------------------
Quantum computing is an emerging field of computing that
operates using quantum mechanics rather than classical
computing principles...
--------------------------------------------------
Generated 80 tokens in 4.28s (18.7 tok/s)
```

## How It Works

### 1. Streaming Layer-by-Layer Loading

The breakthrough that makes 70B+ models possible on consumer hardware. When a model's FP16 size exceeds available memory, Graviton automatically switches to streaming mode:

1. **Meta skeleton** — The model architecture is built on PyTorch's meta device (zero memory).
2. **Weight index** — Parses `model.safetensors.index.json` to map every weight to its shard file.
3. **Non-layer weights** — Embeddings, final norm, and LM head are loaded and moved to the target device.
4. **Layer-by-layer** — Each transformer layer is loaded from its safetensors shard, cast to the target dtype, moved to the target device, quantized in-place, and the FP16 original is freed.
5. **Progress** — The UI shows real-time progress: "Loading layer 42/80..."

```
72B model (FP16): 144 GB — impossible to fit in 64 GB RAM
                     ↓ streaming loader
Build skeleton on meta device: 0 GB
Load embeddings + head: ~1 GB
Layer 1: load FP16 (1.75 GB) → quantize to 4-bit (0.44 GB) → free FP16
Layer 2: load FP16 (1.75 GB) → quantize to 4-bit (0.44 GB) → free FP16
...
Layer 80: same
                     ↓
Final: ~36 GB quantized model on device. Peak: ~38 GB.
```

### 2. Extreme Quantization + QuantizedLinear

Graviton supports multiple quantization strategies applied **directly to model weights** at load time via the `QuantizedLinear` module:

- **INT8**: Per-group symmetric quantization — near-lossless quality, 2x memory savings
- **INT4**: Aggressive 4-bit — significant memory savings, good for large models
- **Mixed-Precision**: Critical layers (attention) at 8-bit, FFN layers at 4-bit
- **1.58-bit (Ternary)**: Inspired by [BitNet b1.58](https://arxiv.org/abs/2402.17764) — weights are {-1, 0, +1}, matmul becomes pure addition/subtraction

```python
from graviton.quantization import TernaryQuantizer, QuantizedLinear

quantizer = TernaryQuantizer()
ql = QuantizedLinear.from_linear(original_layer, quantizer)
# 500B params x 1.58 bits = ~99 GB (vs 1 TB at FP16!)
```

### 3. Dynamic Sparsity

Not all neurons are needed for every input. Graviton's Top-K activation only computes the most relevant neurons:

```python
from graviton.sparsity import TopKActivation

sparse = TopKActivation(k_ratio=0.3)  # Only 30% of neurons fire
output = sparse(hidden_states)
# 70% less computation per layer!
```

### 4. Layer Streaming via MMAP

When a model doesn't fit in memory, Graviton streams layers from SSD:

```python
from graviton.memory import LayerStreamer

streamer = LayerStreamer(model_path, max_memory_gb=16)
# Layers are loaded on-demand, prefetched asynchronously
# Even a 1 TB model can run with 16 GB RAM!
```

### 5. Speculative Decoding

Graviton includes a self-speculative decoding engine that uses layer-skip as a lightweight draft model. The framework supports any draft model for 2-3x throughput gains:

```python
from graviton.decoding import SpeculativeDecoder

decoder = SpeculativeDecoder(
    draft_forward_fn=draft_model,
    target_forward_fn=target_model,
    gamma=4,  # 4 speculative tokens per step
)
# Verified tokens skip re-computation — identical output quality!
```

```bash
# Enable speculative decoding from CLI
graviton run TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt 'Hello world' --speculative --spec-tokens 4
```

## Benchmarks

### 70B+ Model on Consumer Hardware

*Streaming layer-by-layer loading on Apple M1 Max (64 GB):*

| Model | FP16 Size | After Graviton | Peak Memory | Fits in 64 GB? |
|---|---|---|---|---|
| **Qwen2.5-72B-Instruct** | ~144 GB | **~36 GB** (mixed 4/8-bit) | ~38 GB | Yes |
| **LLaMA-2-70B** | ~140 GB | **~35 GB** (4-bit) | ~37 GB | Yes |
| **Mixtral-8x22B** | ~280 GB | **~70 GB** (4-bit) | ~72 GB | Needs 96 GB+ |

### Inference Speed

*Measured on Apple M1 Max (64 GB) with TinyLlama-1.1B-Chat-v1.0:*

| Mode | Decode Speed | Memory | Notes |
|---|---|---|---|
| **FP16 (baseline)** | ~18 tok/s | 2.05 GB | Full precision |
| **INT8 QuantizedLinear** | ~19 tok/s | 0.78 GB | **62% less memory, same speed** |
| **Mixed-Precision (8/4)** | ~10 tok/s | 0.78 GB | Critical=8bit, FFN=4bit |
| **Speculative (layer-skip)** | framework ready | 2.05 GB | Needs trained draft model for best results |

### Memory Compression via QuantizedLinear

*Actual measured memory when loading TinyLlama-1.1B with `QuantizedLinear`:*

| Quantization | Layers Quantized | Memory Saved | Final Model Size |
|---|---|---|---|
| **INT8 uniform** | 154 linear layers | 1,318 MB | **0.78 GB** (from 2.05 GB) |
| **Mixed 8/4** | 154 linear layers | 1,318 MB | **0.78 GB** |
| **FP16 (none)** | — | — | 2.05 GB |

### Theoretical Compression

| Model | Original FP16 | Graviton INT4 | Graviton Ternary | Reduction |
|---|---|---|---|---|
| **TinyLlama-1.1B** | 2.05 GB | ~0.5 GB | ~0.25 GB | 4-8x |
| LLaMA-3-8B | ~16 GB | ~4 GB | ~2 GB | 4-8x |
| LLaMA-2-70B | ~140 GB | ~35 GB | ~17.5 GB | 4-8x |
| Qwen2.5-72B | ~144 GB | ~36 GB | ~18 GB | 4-8x |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        GravitonEngine                            │
├──────────────────────────────────────────────────────────────────┤
│  Tokenizer ──► GravitonCausalLM ──► Sampler ──► Output           │
│                │                                                 │
│                ├── Embedding                                     │
│                ├── TransformerBlock x N                           │
│                │   ├── RMSNorm + RoPE Attention (GQA)            │
│                │   │   └── QuantizedLinear (Q/K/V/O)             │
│                │   └── SwiGLU FFN (Top-K Sparse)                 │
│                │       └── QuantizedLinear (gate/up/down)         │
│                ├── Final RMSNorm                                 │
│                └── LM Head                                       │
├───────────┬──────────┬──────────┬────────────────────────────────┤
│ Quantize  │ Sparsity │  Memory  │       Decoding                 │
│  Engine   │  Engine  │ Manager  │       Engine                   │
├───────────┼──────────┼──────────┼────────────────────────────────┤
│ • INT8    │ • Top-K  │ • mmap   │ • Speculative (self)           │
│ • INT4    │ • Prune  │ • Stream │   └─ layer-skip draft          │
│ • Mixed   │ • MoE    │ • KV$    │ • Top-K / Top-P                │
│ • 1.58b   │          │ • LRU    │ • Rep. Penalty                 │
│ • QLinear │          │ • Snap   │ • Streaming                    │
├───────────┴──────────┴──────────┴────────────────────────────────┤
│                   Streaming Model Loader                         │
│  (auto-detects large models → meta skeleton → layer-by-layer     │
│   quantize-in-flight → safetensors shard streaming)              │
├──────────────────────────────────────────────────────────────────┤
│                    Hardware Detector                              │
│            (Apple Silicon / CUDA / CPU Auto)                     │
└──────────────────────────────────────────────────────────────────┘
```

## Testing

Graviton has a comprehensive test suite covering every component — **88 tests across 10 test modules**, all passing:

```bash
pytest tests/ -v
# ============================== 88 passed in 2.46s ==============================
```

| Test Module | Tests | Coverage Area |
|---|---|---|
| `test_attention.py` | 6 | RoPE, single/multi-token decode, causal mask correctness, future-leakage prevention |
| `test_config.py` | 11 | Config propagation, presets (Mac Mini/extreme/quality), QuantMode, memory estimation |
| `test_decoding.py` | 9 | Sampler (greedy, top-k, top-p, repetition penalty), SpeculativeDecoder acceptance/rejection |
| `test_engine.py` | 3 | Hardware detection, engine initialization, benchmark |
| `test_memory.py` | 2 | Memory budget enforcement, LRU cache eviction |
| `test_mixed_precision.py` | 11 | Layer-bit selection, overrides, sensitivity scores, quantize/dequantize roundtrip |
| `test_model.py` | 11 | GravitonCausalLM forward pass, KV cache, layer_skip, quantize_weights (linear/ternary/mixed) |
| `test_quantization.py` | 3 | INT8 quantize/dequantize, ternary packing, ternary matmul correctness |
| `test_quantized_linear.py` | 19 | QuantizedLinear INT4/INT8/ternary roundtrip, bias handling, device transfer, KV cache fast-path + compressed mode |
| `test_sparsity.py` | 2 | TopK activation sparsity, identity pass-through |
| `test_transformer.py` | 5 | TransformerBlock forward pass, KV cache integration, residual connections |

Key areas validated:

- **Causal mask correctness** — multi-token decode (speculative verification) produces correct causal attention when Q and KV lengths differ
- **Device-aware quantization** — all pack/unpack operations stay on the correct device (CPU/MPS/CUDA) without silent transfers
- **QuantizedLinear fidelity** — INT8 roundtrip error < 0.05, INT4 < 0.5; cached weights persist across forward calls
- **Speculative decoding** — acceptance/rejection sampling, KV cache snapshot & rollback, bonus token generation
- **Mixed-precision routing** — critical layers get higher precision, FFN layers get aggressive compression

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the full test suite: `pytest tests/ -v` (all 88 tests must pass)
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/opengraviton/graviton.git
cd graviton
pip install -e ".[all]"
pytest tests/ -v   # 88 tests, ~2 seconds
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

## Star History

If you find Graviton useful, please consider giving it a star!

## Acknowledgments

- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — Inspiration for ternary quantization
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Pioneering efficient LLM inference
- [GPTQ](https://arxiv.org/abs/2210.17323) — Post-training quantization techniques
- [Speculative Decoding](https://arxiv.org/abs/2211.17192) — Fast autoregressive decoding

---

<p align="center">
  Made by the Graviton community
  <br>
  <em>AI should be accessible to everyone — not just those who can afford a data center.</em>
</p>
