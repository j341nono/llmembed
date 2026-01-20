# llemb

Unified embedding extraction from Decoder-only LLMs.

## Features

- **Backends**: Support for Hugging Face Transformers.
- **Pooling Strategies**:
    - `mean`: Average pooling of all tokens (excluding padding).
    - `last_token`: Vector of the last token.
    - `eos_token`: Vector corresponding to the EOS token position.
    - `prompt_eol`: Embeddings extracted using a prompt template targeting the last token.
    - `pcoteol`: "Pretended Chain of Thought" - wraps input in a reasoning template.
    - `ke`: "Knowledge Enhancement" - wraps input in a context-aware template.
- **Quantization**: Support for 4-bit and 8-bit quantization via `bitsandbytes`.
- **Layer Selection**: Extract embeddings from any layer.
    - Defaults to **-1** (last layer) for standard strategies.
    - Defaults to **-2** (second-to-last layer) for `pcoteol` and `ke` (as recommended by research).

## Installation

Install using `uv`:

```bash
uv add llemb
```

To include quantization support:

```bash
uv add llemb[quantization]
```

## Quick Start

Initialize the encoder with minimal setup (defaults to transformers, no quantization, cpu/cuda auto-detect):

```python
import llemb

# Minimal setup
enc = llemb.Encoder("meta-llama/Llama-3.1-8B")

# Extract embeddings
embeddings = enc.encode("Hello world", pooling="mean")
print(embeddings.shape)
```

## Advanced Usage

Initialize with specific options:

```python
import llemb

# Initialize encoder with specific backend and configuration
enc = llemb.Encoder(
    model_name="meta-llama/Llama-3.1-8B",
    backend="transformers",
    device="cuda", # Force CUDA
    quantization="4bit" # Use 4-bit quantization
)

# Extract embeddings using pcoteol strategy (automatically uses layer -2)
embeddings = enc.encode("Hello world", pooling="pcoteol")
```

## Transformers Backend Configuration

When using the `transformers` backend, you can pass standard Hugging Face `AutoModel` arguments directly to the `Encoder`.

**Example 1: Using Flash Attention 2**

```python
import torch

encoder = Encoder(
    model_name="meta-llama/Llama-3.1-8B",
    backend="transformers",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```

**Example 2: Custom Quantization Config**

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

encoder = Encoder(
    model_name="meta-llama/Llama-3.1-8B",
    backend="transformers",
    quantization_config=bnb_config
)
```

## References

- **PromptEOL:**

    Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing Wang, and Fuzhen Zhuang. 2024. Scaling Sentence Embeddings with Large Language Models. Findings of the Association for Computational Linguistics: EMNLP 2024.

- **PCoTEOL and KE:**

    Bowen Zhang, Kehua Chang, and Chunping Li. 2024. Simple Techniques for Enhancing Sentence Embeddings in Generative Language Models. arXiv preprint arXiv:2404.03921.

## Development

Clone the repository and sync dependencies:

```bash
git clone https://github.com/j341nono/llemb.git
cd llemb
uv sync --all-extras --dev
```

Run tests:

```bash
uv run pytest
```

Run static analysis:

```bash
uv run ruff check src
uv run mypy src
```
