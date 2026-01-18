# llmembed

Unified embedding extraction from Decoder-only LLMs.

## Features

- **Backends**: Support for Hugging Face Transformers and VLLM.
- **Pooling Strategies**:
    - `mean`: Average pooling of all tokens (excluding padding).
    - `last_token`: Vector of the last token.
    - `eos_token`: Vector corresponding to the EOS token position.
    - `prompt_eol`: Embeddings extracted using a prompt template targeting the last token.
    - `pcoteol`: "Pretended Chain of Thought" - wraps input in a reasoning template.
    - `ke`: "Knowledge Enhancement" - wraps input in a context-aware template.
- **Quantization**: Support for 4-bit and 8-bit quantization via `bitsandbytes`.
- **Layer Selection**: Extract embeddings from any layer (default: last hidden state).

## Installation

Install using `uv`:

```bash
uv add llmembed
```

To include VLLM support:

```bash
uv add llmembed[vllm]
```

To include quantization support:

```bash
uv add llmembed[quantization]
```

## Quick Start

Initialize the encoder with minimal setup (defaults to transformers, no quantization, cpu/cuda auto-detect):

```python
import llmembed

# Minimal setup
enc = llmembed.Encoder("sshleifer/tiny-gpt2")

# Extract embeddings
embeddings = enc.encode("Hello world", pooling="mean")
print(embeddings.shape)
```

## Advanced Usage

Initialize with specific options:

```python
import llmembed

# Initialize encoder with specific backend and configuration
enc = llmembed.Encoder(
    model_name="sshleifer/tiny-gpt2",
    backend="transformers",
    device="cuda", # Force CUDA
    quantization="4bit" # Use 4-bit quantization
)

# Extract embeddings using pcoteol strategy (recommended with layer_index=-2)
embeddings = enc.encode("Hello world", pooling="pcoteol", layer_index=-2)
```

## References

**For `prompt_eol`:**

```bibtex
@inproceedings{jiang-etal-2024-scaling,
    title = "Scaling Sentence Embeddings with Large Language Models",
    author = "Jiang, Ting  and
      Huang, Shaohan  and
      Luan, Zhongzhi  and
      Wang, Deqing  and
      Zhuang, Fuzhen",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.181/",
    doi = "10.18653/v1/2024.findings-emnlp.181",
    pages = "3182--3196",
}
```

**For `pcoteol` and `ke`:**

```bibtex
@misc{zhang2024simpletechniquesenhancingsentence,
      title={Simple Techniques for Enhancing Sentence Embeddings in Generative Language Models}, 
      author={Bowen Zhang and Kehua Chang and Chunping Li},
      year={2024},
      eprint={2404.03921},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.03921}, 
}
```

## Development

Clone the repository and sync dependencies:

```bash
git clone https://github.com/j341nono/llmembed.git
cd llmembed
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
