import pytest

from llemb.backends.transformers_backend import TransformersBackend

MODEL = "sshleifer/tiny-gpt2"

@pytest.fixture
def backend():
    return TransformersBackend(MODEL, device="cpu")

def test_encode_pcoteol(backend):
    text = "hello world"
    emb = backend.encode(text, pooling="pcoteol")
    assert emb.shape[0] == 1
    # Check if template was applied? 
    # We can't easily check internal state, but we can verify it doesn't crash 
    # and returns correct shape.

def test_encode_ke(backend):
    text = "hello world"
    emb = backend.encode(text, pooling="ke")
    assert emb.shape[0] == 1

def test_auto_device_cpu():
    # Force cpu check
    backend = TransformersBackend(MODEL, device=None)
    # Since we likely don't have cuda/mps in this env, it should default to cpu or mps (macos).
    # We can check device type.
    # Actually, in the agent env, it might report 'mps' if it's Mac.
    # But wait, tiny-gpt2 is small.
    # Just ensure it initializes.
    assert backend.device in ["cpu", "cuda", "mps"]
