# Contributing to Graviton 🌌

First off, thank you for considering contributing to OpenGraviton! It's people like you that make Graviton such a highly optimized, universally accessible inference engine.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We expect all contributors to maintain a welcoming, respectful, and collaborative environment.

## 🚀 How Can I Contribute?

### 1. Reporting Bugs
* Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/opengraviton/graviton/issues).
* Open a new Issue and use the `Bug Report` template. 
* Please provide reproducible steps, your exact hardware configuration, and the model (or synthetic scale) being tested.

### 2. Suggesting Enhancements
* Open an issue with the label `enhancement`.
* Explain the expected behavior and why it would improve the performance (latency/memory) or UX of the engine.
* We actively welcome new Quantization, Pruning, or Layer Streaming techniques!

### 3. Submitting Pull Requests
1. Fork the repo and create your branch from `main`.
2. Make sure you install the developer requirements: `pip install -e ".[dev]"`
3. Write clear, documented code. If you are modifying the core engines (`memory/`, `quantization/`, etc.), ensure backward compatibility.
4. **Mandatory:** Run the full PyTest suite locally before pushing: `pytest tests/ -v`.
5. Update the `README.md` or OpenGraviton documentation if you change CLI arguments or configuration schemas.
6. Open a Pull Request using our `PULL_REQUEST_TEMPLATE.md`.

## 🧠 Architectural Guidelines

If you are modifying the core execution flow:
* **Memory first.** Graviton is designed to defy hardware gravity. Never assume the user has unlimited memory. Always respect the `MemoryManager`.
* **C++ & Metal Kernels.** If introducing optimizations, attempt to keep a Python fallback path for platforms that don't support MPS or CUDA natively.
* **Apple Silicon:** We deeply care about the unified memory architecture. Ensure your changes do not break zero-copy assumptions during `mmap` streaming.

## Setting Up the Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/graviton.git
cd graviton

# Install for local development
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

Thank you!
— The OpenGraviton Team
