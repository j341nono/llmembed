[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llemb?logo=pypi&style=flat&color=blue)](https://pypi.org/project/llemb/)
[![PyPI - Package Version](https://img.shields.io/pypi/v/llemb?logo=pypi&style=flat&color=orange)](https://pypi.org/project/llemb/)
[![License](https://img.shields.io/github/license/j341nono/llemb?logo=github&style=flat&color=green)](https://github.com/j341nono/llemb/blob/main/LICENSE)
# llemb: Unified Embedding Extraction from Decoder-only LLMs

**llemb** is a lightweight framework designed to extract high-quality sentence embeddings from Decoder-only Large Language Models (LLMs) like Llama, Mistral, and others. It unifies various state-of-the-art pooling strategies and efficiency optimizations into a simple, coherent interface.

With `llemb`, you can easily leverage powerful LLMs for embedding tasks using advanced techniques like **PromptEOL** and **PCoTEOL**, with built-in support for quantization to run on consumer hardware.

## Features

- **Flexible Backends**: Seamless support for Hugging Face Transformers.
- **Advanced Pooling Strategies**:
    - Standard: `mean`, `last_token`, `eos_token`
    - Research-grade: `prompt_eol`, `pcoteol` (Pretended Chain of Thought), `ke` (Knowledge Enhancement)
- **Efficient Inference**: Native support for **4-bit and 8-bit quantization** via `bitsandbytes`.
- **Granular Control**: Extract embeddings from any layer (defaults to recommended layers based on research).

## Installation

Install via PyPI using `pip` or `uv`.

**Basic Installation**

```bash
pip install llemb
# or
uv add llemb
```

**With Quantization Support**

To enable 4-bit/8-bit quantization (recommended for large models):

```bash
pip install "llemb[quantization]"
# or
uv add llemb[quantization]
```

## Getting Started

Initialize the encoder and start extracting embeddings in just a few lines of code.

### Basic Usage

```python
import llemb

# 1. Initialize the encoder (defaults to auto-device detection)
enc = llemb.Encoder("meta-llama/Llama-3.1-8B")

# 2. Extract embeddings using mean pooling
embeddings = enc.encode("Hello world", pooling="mean")

print(embeddings.shape)
# => (1, 4096)
```

### Advanced Usage (Quantization & Research Strategies)

Use quantization to reduce memory usage and apply advanced pooling strategies like `pcoteol` for better representation.

```python
import llemb

# Initialize with 4-bit quantization and force CUDA
enc = llemb.Encoder(
    model_name="meta-llama/Llama-3.1-8B",
    backend="transformers",
    device="cuda",
    quantization="4bit"
)

# Extract using "Pretended Chain of Thought" strategy
# Note: Automatically uses the second-to-last layer (layer -2) as recommended
embeddings = enc.encode("Hello world", pooling="pcoteol")
```

## Configuration & Optimization

`llemb` passes arguments directly to the backend, allowing for deep customization.

**Using Flash Attention 2**

```python
import torch

encoder = llemb.Encoder(
    model_name="meta-llama/Llama-3.1-8B",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```

**Custom Quantization Config**

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

encoder = llemb.Encoder(
    model_name="meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config
)
```

## Supported Pooling Strategies

| Strategy | Description | Recommended Layer |
| --- | --- | --- |
| `mean` | Average pooling of all tokens (excluding padding). | -1 (Last) |
| `last_token` | Vector of the last generated token. | -1 (Last) |
| `eos_token` | Vector corresponding to the EOS token position. | -1 (Last) |
| `prompt_eol` | Embeddings extracted using a prompt template targeting the last token. | -1 (Last) |
| `pcoteol` | "Pretended Chain of Thought" - wraps input in a reasoning template. | -2 |
| `ke` | "Knowledge Enhancement" - wraps input in a context-aware template. | -2 |

## Development

Clone the repository and sync dependencies using `uv`:

```bash
git clone [https://github.com/j341nono/llemb.git](https://github.com/j341nono/llemb.git)
cd llemb
uv sync --all-extras --dev
```

**Run Tests**

```bash
uv run pytest
```

**Static Analysis**

```bash
uv run ruff check src
uv run mypy src
```

## Citations

If you use the advanced pooling strategies implemented in this library, please cite the respective original papers:

**PromptEOL:**

```bibtex
@inproceedings{jiang-etal-2024-scaling,
    title = "Scaling Sentence Embeddings with Large Language Models",
    author = "Jiang, Ting and Huang, Shaohan and Luan, Zhongzhi and Wang, Deqing and Zhuang, Fuzhen",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    year = "2024"
}
```

**PCoTEOL and KE:**

```bibtex
@article{zhang2024simple,
    title={Simple Techniques for Enhancing Sentence Embeddings in Generative Language Models},
    author={Zhang, Bowen and Chang, Kehua and Li, Chunping},
    journal={arXiv preprint arXiv:2404.03921},
    year={2024}
}
```

## License

This project is open source and available under the [Apache-2.0 license](LICENSE).
