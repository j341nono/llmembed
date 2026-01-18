import pytest
import torch

from llmembed.backends.transformers_backend import TransformersBackend

MODEL = "sshleifer/tiny-gpt2"

@pytest.fixture
def backend():
    return TransformersBackend(MODEL, device="cpu")

def test_load_model(backend):
    assert backend.model is not None
    assert backend.tokenizer is not None

def test_encode_mean(backend):
    emb = backend.encode("hello world", pooling="mean")
    assert isinstance(emb, torch.Tensor)
    assert emb.ndim == 2
    assert emb.shape[0] == 1
    # tiny-gpt2 hidden size is 2 (config says n_embd=2? No, tiny-gpt2 is usually small
    # but 768 is regular GPT2. tiny-gpt2 has n_embd=2)
    # Actually sshleifer/tiny-gpt2: config.n_embd=64, n_layer=2, n_head=2.
    # Let's check shape[1] > 0.
    assert emb.shape[1] > 0

def test_encode_batch(backend):
    emb = backend.encode(["hello", "world"], pooling="mean")
    assert emb.shape[0] == 2

def test_encode_last_token(backend):
    emb = backend.encode(["hello", "world"], pooling="last_token")
    assert emb.shape[0] == 2

def test_encode_eos_token(backend):
    # Ensure EOS is present
    text = "hello world" + backend.tokenizer.eos_token
    emb = backend.encode(text, pooling="eos_token")
    assert emb.shape[0] == 1
    
    # Test fallback if no EOS
    text_no_eos = "hello world"
    emb_fallback = backend.encode(text_no_eos, pooling="eos_token")
    assert emb_fallback.shape[0] == 1

def test_encode_prompt_eol(backend):
    text = "hello world"
    emb = backend.encode(text, pooling="prompt_eol")
    assert emb.shape[0] == 1

def test_layer_index(backend):
    # Layer 0 (embeddings)
    emb_0 = backend.encode("hello", layer_index=0)
    # Layer -1 (last)
    emb_last = backend.encode("hello", layer_index=-1)
    
    assert not torch.allclose(emb_0, emb_last)
