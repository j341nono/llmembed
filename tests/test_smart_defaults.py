import pytest
import torch

from llemb.backends.transformers_backend import TransformersBackend

MODEL = "sshleifer/tiny-gpt2"

@pytest.fixture
def backend():
    return TransformersBackend(MODEL, device="cpu")

def test_smart_default_with_template(backend):
    """Test that pooling_method defaults to last_token when template is provided."""
    # Without explicit pooling_method, should use last_token with template
    emb_implicit = backend.encode("hello", pooling_method=None, prompt_template="pcoteol")
    
    # Explicit last_token with template
    emb_explicit = backend.encode("hello", pooling_method="last_token", prompt_template="pcoteol")
    
    # Should produce the same result
    assert torch.allclose(emb_implicit, emb_explicit)

def test_smart_default_without_template(backend):
    """Test that pooling_method defaults to mean when no template is provided."""
    # Without explicit pooling_method or template, should use mean
    emb_implicit = backend.encode("hello", pooling_method=None, prompt_template=None)
    
    # Explicit mean without template
    emb_explicit = backend.encode("hello", pooling_method="mean", prompt_template=None)
    
    # Should produce the same result
    assert torch.allclose(emb_implicit, emb_explicit)

def test_explicit_pooling_overrides_default(backend):
    """Test that explicit pooling_method overrides the smart default."""
    # Explicit mean with template (overrides default last_token)
    emb_mean = backend.encode("hello", pooling_method="mean", prompt_template="pcoteol")
    
    # Default last_token with template
    emb_last = backend.encode("hello", pooling_method=None, prompt_template="pcoteol")
    
    # Should produce different results
    assert not torch.allclose(emb_mean, emb_last)
