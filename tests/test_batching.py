import pytest
import torch

from llemb.core import Encoder

MODEL = "sshleifer/tiny-gpt2"

@pytest.fixture
def encoder():
    return Encoder(model_name=MODEL, device="cpu")

def test_batching_equivalence(encoder):
    # Create a list of inputs (e.g. 10 sentences)
    inputs = [f"This is sentence number {i}" for i in range(10)]
    
    # 1. Encode all at once
    emb_full = encoder.encode(inputs, pooling_method="mean")
    
    # 2. Encode with batch_size=2
    emb_batched = encoder.encode(inputs, pooling_method="mean", batch_size=2)
    
    # Check shapes
    assert emb_full.shape == (10, 2) # tiny-gpt2 hidden size is 2
    assert emb_batched.shape == (10, 2)
    
    # Check values
    assert torch.allclose(emb_full, emb_batched, atol=1e-5)

def test_batching_odd_size(encoder):
    # Test with batch size that doesn't divide evenly
    inputs = [f"Sentence {i}" for i in range(5)]
    
    emb_batched = encoder.encode(inputs, pooling_method="mean", batch_size=2)
    assert emb_batched.shape == (5, 2)

def test_batching_single_batch(encoder):
    # Test with batch size >= len(inputs)
    inputs = ["One", "Two"]
    emb_batched = encoder.encode(inputs, pooling_method="mean", batch_size=5)
    assert emb_batched.shape == (2, 2)
