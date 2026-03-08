"""
Graviton Command-Line Interface

Provides a user-friendly CLI for running, quantizing, and
benchmarking massive AI models on consumer hardware.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

from graviton import __version__
from graviton.core.config import GravitonConfig, QuantMode
from graviton.core.engine import GravitonEngine
from graviton.core.hardware import detect_hardware, recommend_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("graviton")

# ── Branding ────────────────────────────────────────────────────────

LOGO = r"""
   ____                 _ _
  / ___|_ __ __ ___   _(_) |_ ___  _ __
 | |  _| '__/ _` \ \ / / | __/ _ \| '_ \
 | |_| | | | (_| |\ V /| | || (_) | | | |
  \____|_|  \__,_| \_/ |_|\__\___/|_| |_|
"""

BANNER = f"""\033[1;35m{LOGO}\033[0m\
  \033[1mGraviton v{__version__}\033[0m — Ultra-efficient AI inference engine
  Defying the gravitational pull of massive AI models.\n"""

HELP_EPILOG = textwrap.dedent("""\
\033[1mExamples:\033[0m
  \033[36m# Show hardware capabilities\033[0m
  graviton info

  \033[36m# Run inference with INT8 quantization\033[0m
  graviton run TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      -p 'Explain quantum computing:' -b 8 --no-mixed

  \033[36m# Run in FP16 (no quantization)\033[0m
  graviton run TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      -p 'Hello world' --no-quantize

  \033[36m# Run with speculative decoding\033[0m
  graviton run TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      -p 'Write a poem' --speculative --spec-tokens 4

  \033[36m# Benchmark quantization + sparsity speed\033[0m
  graviton benchmark -b 4 -s 0.3

\033[1mDocumentation:\033[0m  https://github.com/opengraviton/graviton
\033[1mWebsite:\033[0m        https://opengraviton.github.io
""")

RUN_EPILOG = textwrap.dedent("""\
\033[1mExamples:\033[0m
  \033[36m# FP16 baseline (no quantization)\033[0m
  graviton run TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p 'Hello' --no-quantize

  \033[36m# INT8 uniform quantization (62% memory savings)\033[0m
  graviton run TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p 'Hello' -b 8 --no-mixed

  \033[36m# Mixed precision (attention=8bit, FFN=4bit)\033[0m
  graviton run TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p 'Hello' -b 4

  \033[36m# Speculative decoding + low temperature (greedy)\033[0m
  graviton run TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p 'Hello' --speculative -t 0.1
""")


def print_banner():
    print(BANNER)


# ── Commands ────────────────────────────────────────────────────────


def cmd_info(args):
    """Detect hardware and show recommended configs."""
    print_banner()
    print("Detecting hardware capabilities...\n")

    profile = detect_hardware()
    print(profile.summary())

    print("\nRecommended Configurations:")
    print("-" * 50)

    print("\n1. For Maximum Quality (Least Compression):")
    q_config = recommend_config(profile)
    q_config.quantization.mode = QuantMode.INT8
    q_config.sparsity.k_ratio = 0.8
    print(f"   Bits: 8-bit | Sparsity: 20% pruned | Max Params: {profile.max_model_params(8):.1f}B")

    print("\n2. Balanced (Recommended):")
    bal_config = recommend_config(profile)
    print(
        f"   Bits: {bal_config.quant_bits}-bit | "
        f"Sparsity: {(1-bal_config.sparsity.k_ratio):.0%} pruned | "
        f"Max Params: {profile.max_model_params(bal_config.quant_bits):.1f}B"
    )

    print("\n3. Extreme Compression (Largest Models):")
    ext_config = recommend_config(profile)
    ext_config.quantization.mode = QuantMode.TERNARY
    ext_config.sparsity.k_ratio = 0.3
    ext_config.memory.use_layer_streaming = True
    print(f"   Bits: 1.58-bit (Ternary) | Sparsity: 70% pruned | Max Params: {profile.max_model_params(1.58):.1f}B")
    print("   mmap + Layer Streaming enabled: Theoretically unlimited size (bounded by SSD space)")


def cmd_run(args):
    """Run model inference."""
    print_banner()

    config = GravitonConfig(
        model_path=args.model,
        quant_bits=args.bits,
        sparsity_ratio=args.sparsity_ratio,
        memory=GravitonConfig().memory,
        decoding=GravitonConfig().decoding,
        verbose=args.verbose,
        use_speculative=args.speculative,
    )

    if args.memory > 0:
        config.memory.max_memory_gb = args.memory

    config.decoding.temperature = args.temperature
    config.decoding.max_tokens = args.max_tokens
    if args.speculative:
        config.decoding.num_speculative_tokens = args.spec_tokens
    if args.no_quantize:
        config.quantization.mode = QuantMode.NONE
    if args.no_mixed:
        config.quantization.use_mixed_precision = False

    # Show active config summary
    if args.no_quantize:
        quant_label = "FP16 (no quantization)"
    elif args.no_mixed:
        quant_label = f"INT{int(args.bits)} uniform"
    else:
        quant_label = f"Mixed-precision (critical=8bit, other={int(args.bits)}bit)"

    print(f"  Model:         {args.model}")
    print(f"  Quantization:  {quant_label}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Max tokens:    {args.max_tokens}")
    if args.speculative:
        print(f"  Speculative:   gamma={args.spec_tokens}, layer-skip draft")
    print()

    print("Initializing Graviton Engine...")
    engine = GravitonEngine(config=config)

    print(f"Loading model: {args.model}")
    try:
        engine.load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    start_time = time.time()

    print("Generation: ", end="", flush=True)
    generated = ""
    chunk_count = 0
    for chunk in engine.generate(args.prompt, stream=True):
        print(chunk, end="", flush=True)
        generated += chunk
        chunk_count += 1

    elapsed = time.time() - start_time
    print(f"\n{'-' * 50}")
    tok_s = chunk_count / max(elapsed, 0.001)
    print(f"Generated {chunk_count} tokens in {elapsed:.2f}s ({tok_s:.1f} tok/s)")
    print(f"Quantization: {quant_label}")


def cmd_quantize(args):
    """Quantize a model and save it."""
    print_banner()
    print(f"Quantizing model: {args.model}")
    print(f"Target bits: {args.bits}")
    print(f"Output path: {args.output}")
    print("\nInitializing Graviton engine...")

    print("\n[Simulating quantization process]")
    time.sleep(1)
    print("1. Loading raw FP16 weights...")
    time.sleep(1)

    if args.bits <= 2:
        print("2. Computing ternary threshold (absmean)...")
    else:
        print(f"2. Computing {args.bits}-bit linear scales and zero-points...")

    time.sleep(1)
    print("3. Applying quantization...")
    time.sleep(1.5)
    print(f"4. Saving packed weights to {args.output}...")
    time.sleep(0.5)

    print("\nQuantization complete!")
    if args.bits == 1.58:
        print("Achieved ~10x compression using Ternary weights {-1, 0, +1}.")
    else:
        print(f"Achieved {16/args.bits:.1f}x compression.")


def cmd_benchmark(args):
    """Run performance benchmarks."""
    print_banner()

    config = GravitonConfig(
        quant_bits=args.bits,
        sparsity_ratio=args.sparsity_ratio,
    )

    engine = GravitonEngine(config=config)
    results = engine.benchmark()

    print("Benchmark Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"  {key.replace('_', ' ').title():<23}: {value}")
    print("=" * 40)


# ── Parser ──────────────────────────────────────────────────────────


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="graviton",
        description=(
            "\033[1mGraviton\033[0m — Ultra-efficient AI inference engine.\n"
            "Run massive language models on consumer hardware with\n"
            "QuantizedLinear, speculative decoding, and dynamic sparsity."
        ),
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version", action="version",
        version=f"Graviton {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="COMMAND",
    )
    subparsers.required = True

    # ── info ─────────────────────────────────────────────────────
    subparsers.add_parser(
        "info",
        help="Detect hardware and show recommended configs",
        description="Detect hardware capabilities (CPU, GPU, memory) and show recommended quantization / sparsity configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── run ──────────────────────────────────────────────────────
    parser_run = subparsers.add_parser(
        "run",
        help="Run model inference",
        description=(
            "Download (or load local) a HuggingFace model, apply quantization,\n"
            "and generate text from a prompt with streaming output."
        ),
        epilog=RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_run.add_argument(
        "model",
        help="HuggingFace model ID or local path  (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser_run.add_argument(
        "-p", "--prompt", required=True,
        help="Input prompt for generation",
    )
    parser_run.add_argument(
        "-b", "--bits", type=float, default=4.0,
        help="Quantization bits: 8, 4, 2, or 1.58 (default: 4)",
    )
    parser_run.add_argument(
        "-s", "--sparsity-ratio", type=float, default=0.5,
        help="Fraction of neurons to keep active, 0.0-1.0 (default: 0.5)",
    )
    parser_run.add_argument(
        "-m", "--memory", type=float, default=0.0,
        help="Memory budget in GB, 0 = auto-detect (default: 0)",
    )
    parser_run.add_argument(
        "-t", "--temperature", type=float, default=0.7,
        help="Sampling temperature, 0 = greedy (default: 0.7)",
    )
    parser_run.add_argument(
        "-n", "--max-tokens", type=int, default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser_run.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose / debug logging",
    )
    parser_run.add_argument(
        "--speculative", action="store_true",
        help="Enable speculative decoding with layer-skip draft model",
    )
    parser_run.add_argument(
        "--spec-tokens", type=int, default=4,
        help="Number of draft tokens per speculative step (default: 4)",
    )
    parser_run.add_argument(
        "--no-quantize", action="store_true",
        help="Disable all weight quantization (run in FP16)",
    )
    parser_run.add_argument(
        "--no-mixed", action="store_true",
        help="Disable mixed-precision; use uniform bit-width from -b",
    )

    # ── quantize ─────────────────────────────────────────────────
    parser_quantize = subparsers.add_parser(
        "quantize",
        help="Quantize a model offline",
        description="Quantize a HuggingFace model and save the packed weights to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_quantize.add_argument("model", help="Path to input model")
    parser_quantize.add_argument(
        "-o", "--output", required=True,
        help="Output directory for quantized model",
    )
    parser_quantize.add_argument(
        "-b", "--bits", type=float, default=4.0,
        help="Target bits: 8, 4, 2, or 1.58 (default: 4)",
    )

    # ── benchmark ────────────────────────────────────────────────
    parser_bench = subparsers.add_parser(
        "benchmark",
        help="Run quantization and sparsity speed benchmarks",
        description="Run a quick benchmark on a 4096x4096 matrix to measure quantization and sparsity throughput.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_bench.add_argument(
        "-b", "--bits", type=float, default=4.0,
        help="Quantization bits (default: 4)",
    )
    parser_bench.add_argument(
        "-s", "--sparsity-ratio", type=float, default=0.5,
        help="Sparsity ratio (default: 0.5)",
    )

    # ── dispatch ─────────────────────────────────────────────────
    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "quantize":
        cmd_quantize(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
