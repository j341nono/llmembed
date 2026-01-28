import pytest
import torch

from llemb.backends.vllm_backend import VLLMBackend

MODEL = "facebook/opt-125m"

@pytest.fixture(scope="module")
def backend():
    """
    Initialize VLLMBackend once for the entire module to avoid re-init crashes.
    """
    try:
        return VLLMBackend(MODEL, gpu_memory_utilization=0.4, enforce_eager=True)
    except ImportError:
        pytest.skip("vLLM not installed")
    except Exception as e:
        pytest.skip(f"Failed to initialize vLLM (requires GPU?): {e}")

# --- Core Functionality Tests ---

def test_encode_mean(backend):
    emb = backend.encode("hello world", pooling_method="mean")
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (1, 768)

def test_encode_batch(backend):
    texts = ["hello", "world", "vllm"]
    emb = backend.encode(texts, pooling_method="mean")
    assert emb.shape == (3, 768)

def test_empty_input(backend):
    emb = backend.encode([], pooling_method="mean")
    assert emb.numel() == 0

# --- Pooling Strategy Tests ---

def test_encode_last_token(backend):
    emb = backend.encode("hello world", pooling_method="last_token")
    assert emb.shape == (1, 768)

def test_encode_eos_token(backend):
    # Ensure EOS exists
    text = f"hello{backend.tokenizer.eos_token}"
    emb = backend.encode(text, pooling_method="eos_token")
    assert emb.shape == (1, 768)
    
    # Fallback if no EOS
    emb_fallback = backend.encode("hello", pooling_method="eos_token")
    assert emb_fallback.shape == (1, 768)

def test_encode_prompt_strategies(backend):
    text = "complex reasoning task"
    # Test all prompt templates
    for template in ["pcoteol", "ke", "prompteol"]:
        emb = backend.encode(text, pooling_method="last_token", prompt_template=template)
        assert emb.shape == (1, 768)

# --- Warning/Error Tests ---

def test_layer_index_warning(backend, caplog):
    """Verify vLLM warns about unsupported layer_index."""
    with caplog.at_level(pytest.logging.WARNING):
        backend.encode("test", layer_index=-2)
    assert "layer_index" in caplog.text