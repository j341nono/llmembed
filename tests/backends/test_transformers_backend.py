import pytest
import torch

from llemb.backends.transformers_backend import TransformersBackend

# Use a small model for fast CPU testing
MODEL = "sshleifer/tiny-gpt2"

@pytest.fixture
def backend():
    return TransformersBackend(MODEL, device="cpu")

# --- Core Functionality Tests ---

def test_load_model(backend):
    assert backend.model is not None
    assert backend.tokenizer is not None

def test_auto_device_cpu():
    # Force auto-detection (should default to CPU/MPS in this env)
    backend = TransformersBackend(MODEL, device=None)
    assert backend.device in ["cpu", "cuda", "mps"]

def test_encode_mean(backend):
    emb = backend.encode("hello world", pooling="mean")
    assert isinstance(emb, torch.Tensor)
    assert emb.ndim == 2
    assert emb.shape[0] == 1
    assert emb.shape[1] > 0

def test_encode_batch(backend):
    emb = backend.encode(["hello", "world"], pooling="mean")
    assert emb.shape[0] == 2

def test_empty_input(backend):
    emb = backend.encode([], pooling="mean")
    assert emb.numel() == 0

# --- Pooling Strategy Tests ---

def test_encode_last_token(backend):
    emb = backend.encode("hello world", pooling="last_token")
    assert emb.shape[0] == 1

def test_encode_eos_token(backend):
    # Case 1: EOS exists
    text = "hello world" + backend.tokenizer.eos_token
    emb = backend.encode(text, pooling="eos_token")
    assert emb.shape[0] == 1
    
    # Case 2: Fallback (no EOS)
    text_no_eos = "hello world"
    emb_fallback = backend.encode(text_no_eos, pooling="eos_token")
    assert emb_fallback.shape[0] == 1

def test_encode_prompt_strategies(backend):
    """Test prompt engineering strategies (pcoteol, ke, prompt_eol)."""
    text = "hello world"
    for strategy in ["prompt_eol", "pcoteol", "ke"]:
        emb = backend.encode(text, pooling=strategy)
        assert emb.shape[0] == 1
        assert isinstance(emb, torch.Tensor)

# --- Layer Index Tests ---

def test_layer_index_selection(backend):
    # Layer 0 (embeddings) vs Layer -1 (last)
    emb_0 = backend.encode("hello", layer_index=0)
    emb_last = backend.encode("hello", layer_index=-1)
    assert not torch.allclose(emb_0, emb_last)

def test_layer_index_defaults(backend):
    """Verify default layer selection logic."""
    # 'pcoteol' defaults to layer -2
    emb_p_default = backend.encode("hello", pooling="pcoteol", layer_index=None)
    emb_p_explicit = backend.encode("hello", pooling="pcoteol", layer_index=-2)
    assert torch.allclose(emb_p_default, emb_p_explicit)

    # 'mean' defaults to layer -1
    emb_m_default = backend.encode("hello", pooling="mean", layer_index=None)
    emb_m_explicit = backend.encode("hello", pooling="mean", layer_index=-1)
    assert torch.allclose(emb_m_default, emb_m_explicit)